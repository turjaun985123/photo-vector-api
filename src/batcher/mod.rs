use crate::config::LatencyBreakdown;
use crate::engine::SpeculativeCascadeEngine;
use crate::error::{VisionError, VisionResult};
use crate::pipeline::postprocess::{
    compute_softmax_and_entropy, extract_top_k, ClassificationResult,
};
use crate::pipeline::preprocess::preprocess_batch_Slot;
use bytes::Bytes;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// Single request submitted to the adaptive dynamic batcher.
pub struct InferenceJob {
    pub image_bytes: Bytes,
    pub top_k: usize,
    pub enqueued_at: Instant,
    pub responder: oneshot::Sender<VisionResult<(ClassificationResult, LatencyBreakdown)>>,
}

/// Handle cloned across Axum HTTP workers to submit jobs with automatic backpressure.
#[derive(Clone)]
pub struct BatchHandle {
    sender: mpsc::Sender<InferenceJob>,
}

impl BatchHandle {
    pub async fn classify(
        &self,
        image_bytes: Bytes,
        top_k: usize,
    ) -> VisionResult<(ClassificationResult, LatencyBreakdown)> {
        let (tx, rx) = oneshot::channel();
        let job = InferenceJob {
            image_bytes,
            top_k,
            enqueued_at: Instant::now(),
            responder: tx,
        };

        self.sender
            .send(job)
            .await
            .map_err(|_| VisionError::BatcherClosed("BatchCoordinator channel closed".into()))?;

        rx.await
            .map_err(|_| VisionError::BatcherClosed("BatchCoordinator dropped responder".into()))?
    }
}

/// Asynchronous Dynamic Batch Coordinator with adaptive windowing.
pub struct BatchCoordinator {
    receiver: mpsc::Receiver<InferenceJob>,
    engine: Arc<SpeculativeCascadeEngine>,
}

impl BatchCoordinator {
    /// Spawns the background batcher loop and returns a clonable `BatchHandle`.
    pub fn spawn(engine: Arc<SpeculativeCascadeEngine>, queue_capacity: usize) -> BatchHandle {
        let (tx, rx) = mpsc::channel(queue_capacity.max(64));
        let coordinator = Self {
            receiver: rx,
            engine,
        };
        tokio::spawn(coordinator.run_loop());
        BatchHandle { sender: tx }
    }

    async fn run_loop(mut self) {
        let max_batch = self.engine.config().max_batch_size.max(1);
        let window_duration = Duration::from_micros(self.engine.config().max_batch_window_us);

        let mut pending: Vec<InferenceJob> = Vec::with_capacity(max_batch);

        while let Some(first_job) = self.receiver.recv().await {
            pending.push(first_job);

            // Adaptive Windowing: If the channel backlog exceeds 2 * max_batch, drain immediately
            // without waiting on a timer.
            let high_load = self.receiver.len() >= max_batch;
            if high_load {
                while pending.len() < max_batch {
                    match self.receiver.try_recv() {
                        Ok(job) => pending.push(job),
                        Err(_) => break,
                    }
                }
            } else {
                let deadline = tokio::time::Instant::now() + window_duration;
                while pending.len() < max_batch {
                    match tokio::time::timeout_at(deadline, self.receiver.recv()).await {
                        Ok(Some(job)) => pending.push(job),
                        Ok(None) | Err(_) => break,
                    }
                }
            }

            let batch_jobs: Vec<InferenceJob> = pending.drain(..).collect();
            let engine_ref = Arc::clone(&self.engine);

            // Offload SIMD preprocessing + GPU/ORT batch execution to blocking pool
            let _ = tokio::task::spawn_blocking(move || {
                Self::execute_batch(engine_ref, batch_jobs);
            })
            .await;
        }
    }

    fn execute_batch(engine: Arc<SpeculativeCascadeEngine>, jobs: Vec<InferenceJob>) {
        let batch_len = jobs.len();
        if batch_len == 0 {
            return;
        }

        if engine.config().mode != crate::config::OperationalMode::SpeculativeCascade {
            for job in jobs {
                let _ = job.responder.send(Err(VisionError::Config(
                    "classification requires speculative_cascade mode".into(),
                )));
            }
            return;
        }

        // Fast-path single item through full speculative cascade
        if batch_len == 1 {
            if let Some(job) = jobs.into_iter().next() {
                let wait_us = job.enqueued_at.elapsed().as_micros() as u64;
                let res = engine.classify_single(&job.image_bytes, job.top_k, wait_us);
                let _ = job.responder.send(res);
            }
            return;
        }

        let batch_start = Instant::now();
        let mut pool_guard = match engine.tier1_pool().try_acquire() {
            Ok(g) => g,
            Err(_) => {
                // Fallback to individual execution if pool is momentarily saturated
                for job in jobs {
                    let wait_us = job.enqueued_at.elapsed().as_micros() as u64;
                    let res = engine.classify_single(&job.image_bytes, job.top_k, wait_us);
                    let _ = job.responder.send(res);
                }
                return;
            }
        };

        let cfg = engine.config();
        let w = cfg.tier1_width;
        let h = cfg.tier1_height;
        let plane_len = (w as usize) * (h as usize);
        let slot_stride = 3 * plane_len;
        let max_px = cfg.max_image_pixels;
        let norm = cfg.normalization;

        let active_slice_len = batch_len * slot_stride;
        // Split the contiguous batch buffer into disjoint per-image mutable slices for Rayon parallelism
        let raw_ptr = pool_guard.slot_mut(0).map(|s| s.as_mut_ptr());
        let prep_results: Vec<VisionResult<(u64, u64)>> = match raw_ptr {
            Ok(ptr) => {
                // SAFETY: `active_slice_len <= max_batch * slot_stride`, and `chunks_exact_mut(slot_stride)`
                // yields non-overlapping mutable slices for each parallel Rayon worker.
                let full_batch_slice =
                    unsafe { std::slice::from_raw_parts_mut(ptr, active_slice_len) };

                full_batch_slice
                    .par_chunks_exact_mut(slot_stride)
                    .zip(jobs.par_iter())
                    .map(|(slot_slice, job)| {
                        let (r, rest) = slot_slice.split_at_mut(plane_len);
                        let (g, b) = rest.split_at_mut(plane_len);
                        let out = preprocess_batch_Slot(
                            &job.image_bytes,
                            max_px,
                            w,
                            h,
                            false,
                            &norm,
                            r,
                            g,
                            b,
                        )?;
                        Ok((out.decode_us, out.resize_and_transpose_us))
                    })
                    .collect()
            }
            Err(e) => {
                for job in jobs {
                    let _ = job
                        .responder
                        .send(Err(VisionError::MemoryLayout(e.to_string())));
                }
                return;
            }
        };

        let t_inf = Instant::now();
        let batch_slice = match pool_guard.active_batch_slice(batch_len) {
            Ok(s) => s,
            Err(e) => {
                for job in jobs {
                    let _ = job
                        .responder
                        .send(Err(VisionError::MemoryLayout(e.to_string())));
                }
                return;
            }
        };

        let raw_logits = match engine.tier1_backend().run_batch(batch_slice, batch_len) {
            Ok(out) => out,
            Err(e) => {
                for job in jobs {
                    let _ = job
                        .responder
                        .send(Err(VisionError::Inference(e.to_string())));
                }
                return;
            }
        };
        let batch_inf_us = t_inf.elapsed().as_micros() as u64;
        let num_classes = cfg.num_classes;
        let expected_logits = batch_len.saturating_mul(num_classes);
        if raw_logits.len() != expected_logits {
            let error = VisionError::Inference(format!(
                "Tier-1 model returned {} values for a batch requiring exactly {expected_logits}",
                raw_logits.len()
            ));
            for job in jobs {
                let _ = job
                    .responder
                    .send(Err(VisionError::Inference(error.to_string())));
            }
            return;
        }

        for (idx, (job, prep_res)) in jobs.into_iter().zip(prep_results.into_iter()).enumerate() {
            let queue_wait_us = job.enqueued_at.elapsed().as_micros() as u64;
            match prep_res {
                Err(err) => {
                    let _ = job.responder.send(Err(err));
                }
                Ok((decode_us, resize_us)) => {
                    let t_post = Instant::now();
                    let start = idx * num_classes;
                    let end = start + num_classes;
                    let mut item_logits = raw_logits[start..end].to_vec();
                    let entropy =
                        compute_softmax_and_entropy(&mut item_logits, cfg.cascade.temperature);
                    let max_confidence = item_logits.iter().copied().fold(0.0_f32, f32::max);
                    if cfg.tier2_model_path.is_some()
                        && (max_confidence < cfg.cascade.min_confidence
                            || entropy > cfg.cascade.max_entropy)
                    {
                        let wait_us = job.enqueued_at.elapsed().as_micros() as u64;
                        let result = engine.classify_single(&job.image_bytes, job.top_k, wait_us);
                        let _ = job.responder.send(result);
                        continue;
                    }
                    let predictions = extract_top_k(&item_logits, job.top_k, engine.labels());
                    let post_us = t_post.elapsed().as_micros() as u64;

                    let telemetry = LatencyBreakdown {
                        queue_wait_us,
                        decode_us,
                        resize_and_transpose_us: resize_us,
                        inference_tier1_us: batch_inf_us / (batch_len as u64).max(1),
                        inference_tier2_us: None,
                        postprocess_us: post_us,
                        total_us: batch_start.elapsed().as_micros() as u64 + queue_wait_us,
                    };

                    let _ = job.responder.send(Ok((
                        ClassificationResult {
                            predictions,
                            entropy,
                            resolved_by_tier2: false,
                        },
                        telemetry,
                    )));
                }
            }
        }
    }
}
