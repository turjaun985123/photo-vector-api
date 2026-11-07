use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use wide::f32x8;

/// Single class prediction with calibrated probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassPrediction {
    pub class_id: usize,
    pub label: String,
    pub confidence: f32,
}

/// Complete classification result including Shannon entropy and cascade tier metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub predictions: Vec<ClassPrediction>,
    pub entropy: f32,
    pub resolved_by_tier2: bool,
}

/// Bounding box prediction in original image pixel coordinates `[x1, y1, x2, y2]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: usize,
}

impl BoundingBox {
    #[inline]
    #[must_use]
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    #[inline]
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }
}

/// Detection result for a single image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub boxes: Vec<BoundingBox>,
}

/// Computes numerically stable temperature-scaled Softmax in-place and returns Shannon entropy:
/// $$H(p) = -\sum_{i=1}^C p_i \ln(p_i)$$
pub fn compute_softmax_and_entropy(logits: &mut [f32], temperature: f32) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let inv_temp = 1.0 / temperature.max(1e-4);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum_exp = 0.0_f32;
    for z in logits.iter_mut() {
        let e = ((*z - max_logit) * inv_temp).exp();
        *z = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let mut entropy = 0.0_f32;

    for p in logits.iter_mut() {
        *p *= inv_sum;
        let prob = *p;
        if prob > 1e-12 {
            entropy = (-prob).mul_add(prob.ln(), entropy);
        }
    }
    entropy.max(0.0)
}

/// Extracts the Top-K predictions in $O(C)$ average time using partial quickselect (`select_nth_unstable_by`).
#[must_use]
pub fn extract_top_k(probs: &[f32], k: usize, labels: &[String]) -> Vec<ClassPrediction> {
    if probs.is_empty() || k == 0 {
        return Vec::new();
    }
    let top_k = k.min(probs.len());
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();

    if top_k < indexed.len() {
        let (_, _, _) = indexed.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        indexed.truncate(top_k);
    }

    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    indexed
        .into_iter()
        .map(|(class_id, confidence)| {
            let label = labels
                .get(class_id)
                .cloned()
                .unwrap_or_else(|| format!("class_{class_id}"));
            ClassPrediction {
                class_id,
                label,
                confidence,
            }
        })
        .collect()
}

/// Vectorized Distance-IoU Non-Maximum Suppression (DIoU-NMS).
/// Accounts for both overlap area and normalized center-point distance between bounding boxes.
#[must_use]
pub fn diou_nms(
    mut candidates: Vec<BoundingBox>,
    diou_threshold: f32,
    max_detections: usize,
) -> Vec<BoundingBox> {
    if candidates.is_empty() {
        return Vec::new();
    }

    candidates.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let mut kept: Vec<BoundingBox> = Vec::with_capacity(max_detections.min(candidates.len()));
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        let anchor = candidates[i];
        kept.push(anchor);
        if kept.len() >= max_detections {
            break;
        }

        for j in (i + 1)..candidates.len() {
            if suppressed[j] || candidates[j].class_id != anchor.class_id {
                continue;
            }
            if compute_diou(&anchor, &candidates[j]) > diou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

#[inline]
fn compute_diou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let union_area = a.area() + b.area() - inter_area;
    if union_area <= 1e-7 {
        return 0.0;
    }
    let iou = inter_area / union_area;

    let (ax, ay) = a.center();
    let (bx, by) = b.center();
    let center_dist_sq = (ax - bx).mul_add(ax - bx, (ay - by) * (ay - by));

    let enclose_x1 = a.x1.min(b.x1);
    let enclose_y1 = a.y1.min(b.y1);
    let enclose_x2 = a.x2.max(b.x2);
    let enclose_y2 = a.y2.max(b.y2);
    let enclose_w = (enclose_x2 - enclose_x1).max(0.0);
    let enclose_h = (enclose_y2 - enclose_y1).max(0.0);
    let diag_sq = enclose_w.mul_add(enclose_w, enclose_h * enclose_h);

    if diag_sq <= 1e-7 {
        return iou;
    }

    iou - (center_dist_sq / diag_sq)
}

/// 8-lane portable SIMD L2-normalization in-place using `wide::f32x8`.
pub fn l2_normalize_in_place(vec: &mut [f32]) {
    let mut sum_sq_v = f32x8::ZERO;
    let chunks = vec.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: [f32; 8] = chunk.try_into().unwrap_or([0.0; 8]);
        let v = f32x8::from(arr);
        sum_sq_v = v.mul_add(v, sum_sq_v);
    }

    let mut sum_sq = sum_sq_v.reduce_add();
    for &x in remainder {
        sum_sq = x.mul_add(x, sum_sq);
    }

    let norm = sum_sq.sqrt();
    if norm > 1e-12 {
        let inv_norm = 1.0 / norm;
        let inv_v = f32x8::splat(inv_norm);
        let mut chunks_mut = vec.chunks_exact_mut(8);
        for chunk in &mut chunks_mut {
            let arr: [f32; 8] = (&*chunk).try_into().unwrap_or([0.0; 8]);
            let v = f32x8::from(arr) * inv_v;
            chunk.copy_from_slice(&v.to_array());
        }
        for x in chunks_mut.into_remainder() {
            *x *= inv_norm;
        }
    }
}

/// 8-lane SIMD dot-product between two L2-normalized vectors.
#[inline]
#[must_use]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    let mut acc = f32x8::ZERO;
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ca, cb) in a_chunks.zip(b_chunks) {
        let va = f32x8::from(<[f32; 8]>::try_from(ca).unwrap_or([0.0; 8]));
        let vb = f32x8::from(<[f32; 8]>::try_from(cb).unwrap_or([0.0; 8]));
        acc = va.mul_add(vb, acc);
    }

    let mut dot = acc.reduce_add();
    for (&xa, &xb) in a_rem.iter().zip(b_rem.iter()) {
        dot = xa.mul_add(xb, dot);
    }
    dot
}
