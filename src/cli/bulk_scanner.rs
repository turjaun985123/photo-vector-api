use crate::config::LatencyBreakdown;
use crate::engine::SpeculativeCascadeEngine;
use crate::error::{VisionError, VisionResult};
use crate::pipeline::postprocess::{compute_softmax_and_entropy, extract_top_k, ClassPrediction};
use crate::pipeline::preprocess::preprocess_batch_Slot;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;

/// Single NDJSON record emitted by the bulk directory scanner.
#[derive(Debug, Serialize)]
pub struct BulkImageRecord {
    pub file_path: String,
    pub predictions: Vec<ClassPrediction>,
    pub entropy: f32,
    pub latency: LatencyBreakdown,
}

/// Summary telemetry after scanning a directory of 1,000+ photos.
#[derive(Debug, Clone, Serialize)]
pub struct BulkScanSummary {
    pub total_files_discovered: usize,
    pub successful_images: usize,
    pub failed_images: usize,
    pub wall_time_ms: f64,
    pub throughput_images_per_sec: f64,
}

struct PreparedBatchSlot {
    path: String,
    decode_us: u64,
    resize_us: u64,
}

/// Two-stage Producer-Consumer Bulk Scanner:
/// 1. Producer (`rayon` + `memmap2` + `zune-jpeg` + `fast_image_resize` + FMA `HWC->NCHW`)
/// 2. Consumer (Dedicated inference runner + 256 KB buffered NDJSON writer)
pub fn run_bulk_directory_scan(
    engine: Arc<SpeculativeCascadeEngine>,
    input_dir: &Path,
    output_ndjson: &Path,
    top_k: usize,
) -> VisionResult<BulkScanSummary> {
    if engine.config().mode != crate::config::OperationalMode::SpeculativeCascade {
        return Err(VisionError::Config(
            "bulk classification requires speculative-cascade mode".into(),
        ));
    }
    let scan_start = Instant::now();

    let image_paths: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_supported_extension(p))
        .collect();

    let total_files_discovered = image_paths.len();
    let out_file = File::create(output_ndjson).map_err(|source| VisionError::Io {
        path: Some(output_ndjson.to_path_buf()),
        source,
    })?;
    let mut writer = BufWriter::with_capacity(256 * 1024, out_file);

    if total_files_discovered == 0 {
        return Ok(BulkScanSummary {
            total_files_discovered: 0,
            successful_images: 0,
            failed_images: 0,
            wall_time_ms: scan_start.elapsed().as_secs_f64() * 1000.0,
            throughput_images_per_sec: 0.0,
        });
    }

    let max_batch = engine.config().max_batch_size.max(1);
    let w = engine.config().tier1_width;
    let h = engine.config().tier1_height;
    let plane_len = (w as usize) * (h as usize);
    let slot_stride = 3 * plane_len;
    let max_px = engine.config().max_image_pixels;
    let norm = engine.config().normalization;
    let num_classes = engine.config().num_classes;

    let (tx, rx) = mpsc::sync_channel::<(Vec<f32>, Vec<PreparedBatchSlot>, usize)>(8);
    let producer_paths = image_paths;

    // Stage 1: Producer thread driving Rayon parallel memory-mapped decode & SIMD resize
    let producer_handle = std::thread::spawn(move || {
        let mut failed_count = 0_usize;

        for chunk in producer_paths.chunks(max_batch) {
            let decoded_items: Vec<Option<(String, Vec<f32>, u64, u64)>> = chunk
                .par_iter()
                .map(|path| {
                    let file = File::open(path).ok()?;
                    // SAFETY: Read-only memory map of local image file during decode pass.
                    let mmap = unsafe { Mmap::map(&file).ok()? };
                    let mut slot_buf = vec![0.0_f32; slot_stride];
                    let (r, rest) = slot_buf.split_at_mut(plane_len);
                    let (g, b) = rest.split_at_mut(plane_len);

                    let out =
                        preprocess_batch_Slot(&mmap, max_px, w, h, false, &norm, r, g, b).ok()?;
                    Some((
                        path.to_string_lossy().into_owned(),
                        slot_buf,
                        out.decode_us,
                        out.resize_and_transpose_us,
                    ))
                })
                .collect();

            let mut contiguous_batch = Vec::with_capacity(chunk.len() * slot_stride);
            let mut meta = Vec::with_capacity(chunk.len());

            for item in decoded_items {
                match item {
                    Some((path, buf, decode_us, resize_us)) => {
                        contiguous_batch.extend_from_slice(&buf);
                        meta.push(PreparedBatchSlot {
                            path,
                            decode_us,
                            resize_us,
                        });
                    }
                    None => {
                        failed_count += 1;
                    }
                }
            }

            if !meta.is_empty() {
                if tx.send((contiguous_batch, meta, failed_count)).is_err() {
                    break;
                }
                failed_count = 0;
            }
        }
        if failed_count > 0 {
            let _ = tx.send((Vec::new(), Vec::new(), failed_count));
        }
    });

    // Stage 2: Dedicated Consumer executing batched inference and streaming NDJSON
    let mut successful_images = 0_usize;
    let mut failed_images = 0_usize;

    while let Ok((batch_tensor, meta, chunk_failures)) = rx.recv() {
        failed_images += chunk_failures;
        let active_batch = meta.len();
        if active_batch == 0 {
            continue;
        }

        let t_inf = Instant::now();
        let logits = engine
            .tier1_backend()
            .run_batch(&batch_tensor, active_batch)?;
        let required_logits = active_batch.saturating_mul(num_classes);
        if logits.len() != required_logits {
            let _ = producer_handle.join();
            return Err(VisionError::Inference(format!(
                "Tier-1 model returned {} values for a batch requiring exactly {required_logits}",
                logits.len()
            )));
        }
        let inf_us_per_img = (t_inf.elapsed().as_micros() as u64) / (active_batch as u64).max(1);

        for (i, slot_meta) in meta.into_iter().enumerate() {
            let t_post = Instant::now();
            let start = i * num_classes;
            let end = start + num_classes;
            let mut img_logits = logits[start..end].to_vec();

            let entropy =
                compute_softmax_and_entropy(&mut img_logits, engine.config().cascade.temperature);
            let max_confidence = img_logits.iter().copied().fold(0.0_f32, f32::max);
            let (predictions, entropy, latency) = if engine.config().tier2_model_path.is_some()
                && (max_confidence < engine.config().cascade.min_confidence
                    || entropy > engine.config().cascade.max_entropy)
            {
                let raw_bytes =
                    std::fs::read(&slot_meta.path).map_err(|source| VisionError::Io {
                        path: Some(PathBuf::from(&slot_meta.path)),
                        source,
                    })?;
                let (result, latency) = engine.classify_single(&raw_bytes, top_k, 0)?;
                (result.predictions, result.entropy, latency)
            } else {
                let predictions = extract_top_k(&img_logits, top_k, engine.labels());
                let post_us = t_post.elapsed().as_micros() as u64;
                let latency = LatencyBreakdown {
                    queue_wait_us: 0,
                    decode_us: slot_meta.decode_us,
                    resize_and_transpose_us: slot_meta.resize_us,
                    inference_tier1_us: inf_us_per_img,
                    inference_tier2_us: None,
                    postprocess_us: post_us,
                    total_us: slot_meta.decode_us + slot_meta.resize_us + inf_us_per_img + post_us,
                };
                (predictions, entropy, latency)
            };

            let record = BulkImageRecord {
                file_path: slot_meta.path,
                predictions,
                entropy,
                latency,
            };

            serde_json::to_writer(&mut writer, &record)
                .map_err(|e| VisionError::Serialization(e.to_string()))?;
            writer.write_all(b"\n").map_err(|source| VisionError::Io {
                path: Some(output_ndjson.to_path_buf()),
                source,
            })?;

            successful_images += 1;
        }
    }

    let _ = producer_handle.join();
    writer.flush().map_err(|source| VisionError::Io {
        path: Some(output_ndjson.to_path_buf()),
        source,
    })?;

    let wall_secs = scan_start.elapsed().as_secs_f64();
    let throughput = if wall_secs > 1e-9 {
        successful_images as f64 / wall_secs
    } else {
        0.0
    };

    Ok(BulkScanSummary {
        total_files_discovered,
        successful_images,
        failed_images,
        wall_time_ms: wall_secs * 1000.0,
        throughput_images_per_sec: throughput,
    })
}

fn is_supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        })
        .unwrap_or(false)
}
