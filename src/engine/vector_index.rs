use crate::error::{VisionError, VisionResult};
use crate::pipeline::postprocess::{cosine_similarity_simd, l2_normalize_in_place};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A single similarity match returned from the contiguous SIMD vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    pub id: String,
    pub score: f32,
}

/// Zero-copy `rkyv` serializable representation of the contiguous matrix index.
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct PersistedIndexData {
    pub dim: u32,
    pub ids: Vec<String>,
    pub contiguous_vectors: Vec<f32>,
}

/// High-throughput contiguous flat vector index supporting sub-50-microsecond search over 1,000+ photos.
#[derive(Clone)]
pub struct FlatVectorIndex {
    dim: usize,
    ids: Vec<String>,
    contiguous_vectors: Vec<f32>,
}

impl FlatVectorIndex {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::with_capacity(1024),
            contiguous_vectors: Vec::with_capacity(1024 * dim),
        }
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Inserts an embedding vector (automatically L2-normalized) into the contiguous matrix.
    pub fn insert(&mut self, id: String, mut embedding: Vec<f32>) -> VisionResult<()> {
        if embedding.len() != self.dim {
            return Err(VisionError::DimensionMismatch {
                expected: self.dim,
                actual: embedding.len(),
            });
        }
        l2_normalize_in_place(&mut embedding);
        self.ids.push(id);
        self.contiguous_vectors.extend_from_slice(&embedding);
        Ok(())
    }

    /// Removes every indexed view associated with one photo or product ID.
    pub fn remove_id(&mut self, id: &str) {
        if !self.ids.iter().any(|existing| existing == id) {
            return;
        }
        let mut ids = Vec::with_capacity(self.ids.len());
        let mut vectors = Vec::with_capacity(self.contiguous_vectors.len());
        for (existing_id, vector) in self
            .ids
            .iter()
            .zip(self.contiguous_vectors.chunks_exact(self.dim))
        {
            if existing_id != id {
                ids.push(existing_id.clone());
                vectors.extend_from_slice(vector);
            }
        }
        self.ids = ids;
        self.contiguous_vectors = vectors;
    }

    /// Computes SIMD dot-products across all $N$ indexed vectors and extracts Top-K in $O(N)$ time.
    pub fn search(&self, query: &[f32], top_k: usize) -> VisionResult<Vec<SearchMatch>> {
        if self.dim == 0 {
            return Err(VisionError::Config(
                "vector index dimension must be non-zero".into(),
            ));
        }
        if query.len() != self.dim {
            return Err(VisionError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if self.ids.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let mut normalized_query = query.to_vec();
        l2_normalize_in_place(&mut normalized_query);

        // An ID may have several indexed views (full image and crops). Keep
        // only the strongest view per photo so one photo cannot occupy many
        // positions in the result list.
        let mut best_by_id: HashMap<&str, f32> = HashMap::new();
        for (id, row) in self
            .ids
            .iter()
            .zip(self.contiguous_vectors.chunks_exact(self.dim))
        {
            let score = cosine_similarity_simd(&normalized_query, row);
            best_by_id
                .entry(id.as_str())
                .and_modify(|best| *best = best.max(score))
                .or_insert(score);
        }

        let mut scores: Vec<(&str, f32)> = best_by_id.into_iter().collect();
        let k = top_k.min(scores.len());
        if k < scores.len() {
            let (_, _, _) = scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            scores.truncate(k);
        }
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        Ok(scores
            .into_iter()
            .map(|(id, score)| SearchMatch {
                id: id.to_owned(),
                score,
            })
            .collect())
    }

    /// Returns the number of distinct photo IDs represented in the index.
    #[must_use]
    pub fn photo_count(&self) -> usize {
        let mut ids = std::collections::HashSet::new();
        ids.extend(self.ids.iter().map(String::as_str));
        ids.len()
    }

    /// Iterates over photo IDs and their indexed vectors in insertion order.
    pub fn iter_vectors(&self) -> impl Iterator<Item = (&str, &[f32])> + '_ {
        self.ids.iter().enumerate().map(move |(index, id)| {
            let start = index * self.dim;
            let end = start + self.dim;
            (id.as_str(), &self.contiguous_vectors[start..end])
        })
    }

    /// Persists the index to disk using `rkyv` zero-copy binary archive format.
    pub fn save_to_disk(&self, path: &Path) -> VisionResult<()> {
        if self.dim == 0 || self.contiguous_vectors.len() != self.ids.len().saturating_mul(self.dim)
        {
            return Err(VisionError::Serialization(
                "vector index has inconsistent dimensions or storage".into(),
            ));
        }
        let dim = u32::try_from(self.dim).map_err(|_| {
            VisionError::Serialization("vector dimension exceeds archive format limit".into())
        })?;
        let payload = PersistedIndexData {
            dim,
            ids: self.ids.clone(),
            contiguous_vectors: self.contiguous_vectors.clone(),
        };
        let bytes = rkyv::to_bytes::<_, 4096>(&payload)
            .map_err(|e| VisionError::Serialization(e.to_string()))?;

        let file = File::create(path).map_err(|source| VisionError::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);
        writer.write_all(&bytes).map_err(|source| VisionError::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;
        writer.flush().map_err(|source| VisionError::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;
        Ok(())
    }

    /// Memory-maps an `.rkyv` file from disk and validates/loads the contiguous index.
    pub fn load_from_disk(path: &Path) -> VisionResult<Self> {
        let file = File::open(path).map_err(|source| VisionError::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;
        // SAFETY: The file handle is read-only and validated via `rkyv::from_bytes` immediately below.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| VisionError::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;

        let data: PersistedIndexData = rkyv::from_bytes(&mmap)
            .map_err(|e| VisionError::Serialization(format!("rkyv validation failed: {e}")))?;

        let dim = data.dim as usize;
        let expected_values =
            data.ids.len().checked_mul(dim).ok_or_else(|| {
                VisionError::Serialization("vector index dimensions overflow".into())
            })?;
        if dim == 0 || data.contiguous_vectors.len() != expected_values {
            return Err(VisionError::Serialization(
                "vector index contains inconsistent dimensions or storage".into(),
            ));
        }

        Ok(Self {
            dim,
            ids: data.ids,
            contiguous_vectors: data.contiguous_vectors,
        })
    }
}
