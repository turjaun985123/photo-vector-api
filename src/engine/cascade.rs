use crate::config::EngineConfig;
use crate::config::LatencyBreakdown;
use crate::engine::ort_backend::OrtBackend;
use crate::error::VisionResult;
use crate::memory::TensorPool;
use crate::pipeline::postprocess::{
    compute_softmax_and_entropy, diou_nms, extract_top_k, l2_normalize_in_place, BoundingBox,
    ClassificationResult, DetectionResult,
};
use crate::pipeline::preprocess::{create_horizontal_flip_nchw, preprocess_single_into_slot};
use std::sync::Arc;
use std::time::Instant;

/// Unified Speculative Cascade, Object Detection, and Embedding Engine.
pub struct SpeculativeCascadeEngine {
    config: EngineConfig,
    tier1_backend: OrtBackend,
    tier2_backend: OrtBackend,
    tier1_pool: Arc<TensorPool>,
    tier2_pool: Arc<TensorPool>,
    labels: Vec<String>,
}

impl SpeculativeCascadeEngine {
    pub fn new(config: EngineConfig) -> VisionResult<Self> {
        if config.tier1_width == 0
            || config.tier1_height == 0
            || config.tier2_width == 0
            || config.tier2_height == 0
        {
            return Err(crate::error::VisionError::Config(
                "model input dimensions must be non-zero".into(),
            ));
        }
        if config.max_batch_size == 0
            || config.tensor_pool_capacity == 0
            || config.max_image_pixels == 0
            || config.intra_op_threads == 0
            || config.inter_op_threads == 0
        {
            return Err(crate::error::VisionError::Config(
                "batch size, tensor pool capacity, image pixel limit, and inference thread counts must be non-zero".into(),
            ));
        }
        if config.num_classes == 0 || config.embedding_dim == 0 {
            return Err(crate::error::VisionError::Config(
                "class count and embedding dimension must be non-zero".into(),
            ));
        }
        if !config.cascade.min_confidence.is_finite()
            || !config.cascade.max_entropy.is_finite()
            || !config.cascade.temperature.is_finite()
            || config.cascade.temperature <= 0.0
        {
            return Err(crate::error::VisionError::Config("cascade thresholds and temperature must be finite, and temperature must be positive".into()));
        }
        if !config.detection.confidence_threshold.is_finite()
            || !config.detection.diou_nms_threshold.is_finite()
            || config.detection.max_detections == 0
        {
            return Err(crate::error::VisionError::Config(
                "detection thresholds must be finite and max_detections must be non-zero".into(),
            ));
        }
        if config
            .normalization
            .mean
            .iter()
            .chain(config.normalization.std.iter())
            .chain(config.normalization.fma_scale.iter())
            .chain(config.normalization.fma_bias.iter())
            .any(|v| !v.is_finite())
            || config.normalization.std.iter().any(|v| v.abs() < 1e-9)
        {
            return Err(crate::error::VisionError::Config(
                "normalization parameters must be finite and standard deviations non-zero".into(),
            ));
        }
        for model_path in [&config.tier1_model_path, &config.tier2_model_path]
            .into_iter()
            .flatten()
        {
            if !model_path.is_file() {
                return Err(crate::error::VisionError::Config(format!(
                    "model file does not exist or is not a file: {}",
                    model_path.display()
                )));
            }
        }
        if let Some(labels_path) = &config.labels_path {
            if !labels_path.is_file() {
                return Err(crate::error::VisionError::Config(format!(
                    "label file does not exist or is not a file: {}",
                    labels_path.display()
                )));
            }
        }
        let output_dim = match config.mode {
            crate::config::OperationalMode::SpeculativeCascade => config.num_classes,
            crate::config::OperationalMode::ObjectDetection => 84 * 8400,
            crate::config::OperationalMode::EmbeddingSearch => config.embedding_dim,
        };

        let tier1_backend = OrtBackend::new(
            config.tier1_model_path.as_deref(),
            &config,
            config.tier1_width,
            config.tier1_height,
            output_dim,
        )?;

        let tier2_backend = OrtBackend::new(
            config.tier2_model_path.as_deref(),
            &config,
            config.tier2_width,
            config.tier2_height,
            output_dim,
        )?;

        let tier1_pool = Arc::new(TensorPool::new(
            config.tensor_pool_capacity,
            config.max_batch_size,
            3,
            config.tier1_height as usize,
            config.tier1_width as usize,
        )?);

        let tier2_pool = Arc::new(TensorPool::new(
            config.tensor_pool_capacity,
            config.max_batch_size.max(2),
            3,
            config.tier2_height as usize,
            config.tier2_width as usize,
        )?);

        let labels = if let Some(path) = &config.labels_path {
            let contents =
                std::fs::read_to_string(path).map_err(|source| crate::error::VisionError::Io {
                    path: Some(path.clone()),
                    source,
                })?;
            let labels: Vec<String> = contents
                .lines()
                .map(|line| line.trim().to_owned())
                .collect();
            if labels.len() != config.num_classes || labels.iter().any(String::is_empty) {
                return Err(crate::error::VisionError::Config(format!(
                    "label file must contain exactly {} non-empty lines; found {}",
                    config.num_classes,
                    labels.len()
                )));
            }
            labels
        } else {
            (0..config.num_classes)
                .map(|i| format!("class_{i}"))
                .collect()
        };

        Ok(Self {
            config,
            tier1_backend,
            tier2_backend,
            tier1_pool,
            tier2_pool,
            labels,
        })
    }

    #[must_use]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    #[must_use]
    pub fn tier1_pool(&self) -> &Arc<TensorPool> {
        &self.tier1_pool
    }

    #[must_use]
    pub fn tier1_backend(&self) -> &OrtBackend {
        &self.tier1_backend
    }

    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Executes Speculative Cascade Classification on a single image with optional 2x TTA on Tier-2.
    pub fn classify_single(
        &self,
        raw_bytes: &[u8],
        top_k: usize,
        queue_wait_us: u64,
    ) -> VisionResult<(ClassificationResult, LatencyBreakdown)> {
        self.ensure_mode(crate::config::OperationalMode::SpeculativeCascade)?;
        let total_start = Instant::now();
        let mut guard1 = self.tier1_pool.try_acquire()?;

        let prep = preprocess_single_into_slot(
            raw_bytes,
            self.config.max_image_pixels,
            self.config.tier1_width,
            self.config.tier1_height,
            false,
            &self.config.normalization,
            &mut guard1,
            0,
        )?;

        let t_inf1 = Instant::now();
        let mut tier1_logits = self
            .tier1_backend
            .run_batch(guard1.active_batch_slice(1)?, 1)?;
        if tier1_logits.len() != self.config.num_classes {
            return Err(crate::error::VisionError::Inference(format!(
                "Tier-1 model returned {} values; expected {} classes",
                tier1_logits.len(),
                self.config.num_classes
            )));
        }
        let inference_tier1_us = t_inf1.elapsed().as_micros() as u64;

        let t_post = Instant::now();
        let entropy1 =
            compute_softmax_and_entropy(&mut tier1_logits, self.config.cascade.temperature);
        let max_conf1 = tier1_logits.iter().copied().fold(0.0_f32, f32::max);

        // Speculative Gate Check
        if (max_conf1 >= self.config.cascade.min_confidence
            && entropy1 <= self.config.cascade.max_entropy)
            || self.config.tier2_model_path.is_none()
        {
            let predictions = extract_top_k(&tier1_logits, top_k, &self.labels);
            let postprocess_us = t_post.elapsed().as_micros() as u64;
            let telemetry = LatencyBreakdown {
                queue_wait_us,
                decode_us: prep.decode_us,
                resize_and_transpose_us: prep.resize_and_transpose_us,
                inference_tier1_us,
                inference_tier2_us: None,
                postprocess_us,
                total_us: total_start.elapsed().as_micros() as u64 + queue_wait_us,
            };
            return Ok((
                ClassificationResult {
                    predictions,
                    entropy: entropy1,
                    resolved_by_tier2: false,
                },
                telemetry,
            ));
        }

        // Ambiguous sample -> Escalate to Tier-2 Verifier with optional 2x Test-Time Augmentation
        let t_inf2 = Instant::now();
        let mut guard2 = self.tier2_pool.try_acquire()?;
        let _ = preprocess_single_into_slot(
            raw_bytes,
            self.config.max_image_pixels,
            self.config.tier2_width,
            self.config.tier2_height,
            false,
            &self.config.normalization,
            &mut guard2,
            0,
        )?;

        let final_probs = if self.config.cascade.enable_tier2_tta && guard2.max_batch() >= 2 {
            let stride = guard2.single_image_stride();
            let mut slot0_copy = vec![0.0_f32; stride];
            slot0_copy.copy_from_slice(guard2.slot_mut(0)?);
            let slot1 = guard2.slot_mut(1)?;
            create_horizontal_flip_nchw(
                &slot0_copy,
                slot1,
                self.config.tier2_width as usize,
                self.config.tier2_height as usize,
            );

            let raw_tta = self
                .tier2_backend
                .run_batch(guard2.active_batch_slice(2)?, 2)?;
            let c = self.config.num_classes;
            if raw_tta.len() != 2 * c {
                return Err(crate::error::VisionError::Inference(format!(
                    "Tier-2 model returned {} values for 2x TTA; expected {}",
                    raw_tta.len(),
                    2 * c
                )));
            }
            let mut merged = vec![0.0_f32; c];
            for i in 0..c {
                merged[i] = 0.5 * (raw_tta[i] + raw_tta[c + i]);
            }
            merged
        } else {
            let raw = self
                .tier2_backend
                .run_batch(guard2.active_batch_slice(1)?, 1)?;
            if raw.len() != self.config.num_classes {
                return Err(crate::error::VisionError::Inference(format!(
                    "Tier-2 model returned {} values; expected {} classes",
                    raw.len(),
                    self.config.num_classes
                )));
            }
            raw
        };
        let inference_tier2_us = t_inf2.elapsed().as_micros() as u64;

        let t_post2 = Instant::now();
        let mut calibrated = final_probs;
        let entropy2 = compute_softmax_and_entropy(&mut calibrated, 1.0);
        let predictions = extract_top_k(&calibrated, top_k, &self.labels);
        let postprocess_us = t_post2.elapsed().as_micros() as u64;

        let telemetry = LatencyBreakdown {
            queue_wait_us,
            decode_us: prep.decode_us,
            resize_and_transpose_us: prep.resize_and_transpose_us,
            inference_tier1_us,
            inference_tier2_us: Some(inference_tier2_us),
            postprocess_us,
            total_us: total_start.elapsed().as_micros() as u64 + queue_wait_us,
        };

        Ok((
            ClassificationResult {
                predictions,
                entropy: entropy2,
                resolved_by_tier2: true,
            },
            telemetry,
        ))
    }

    /// Runs Object Detection with Letterbox padding, inverse affine mapping, and DIoU-NMS.
    pub fn detect_single(
        &self,
        raw_bytes: &[u8],
        queue_wait_us: u64,
    ) -> VisionResult<(DetectionResult, LatencyBreakdown)> {
        self.ensure_mode(crate::config::OperationalMode::ObjectDetection)?;
        let total_start = Instant::now();
        let mut guard = self.tier1_pool.try_acquire()?;

        let prep = preprocess_single_into_slot(
            raw_bytes,
            self.config.max_image_pixels,
            self.config.tier1_width,
            self.config.tier1_height,
            true,
            &self.config.normalization,
            &mut guard,
            0,
        )?;

        let t_inf = Instant::now();
        let raw_out = self
            .tier1_backend
            .run_batch(guard.active_batch_slice(1)?, 1)?;
        let inference_tier1_us = t_inf.elapsed().as_micros() as u64;

        let t_post = Instant::now();
        let mut candidates = Vec::new();

        // Decode the configured YOLOv8 COCO output layout [1, 84, 8400]:
        // four box channels followed by 80 class score channels.
        const DETECTION_CLASSES: usize = 80;
        const DETECTION_ANCHORS: usize = 8400;
        const DETECTION_CHANNELS: usize = 4 + DETECTION_CLASSES;
        if raw_out.len() < DETECTION_CHANNELS * DETECTION_ANCHORS {
            return Err(crate::error::VisionError::Inference(format!(
                "detector returned {} values; expected at least {} for [1, 84, 8400] output",
                raw_out.len(),
                DETECTION_CHANNELS * DETECTION_ANCHORS
            )));
        }
        let channel = |idx: usize| &raw_out[idx * DETECTION_ANCHORS..(idx + 1) * DETECTION_ANCHORS];
        let center_x = channel(0);
        let center_y = channel(1);
        let widths = channel(2);
        let heights = channel(3);
        for anchor in 0..DETECTION_ANCHORS {
            let mut class_id = 0;
            let mut score = f32::NEG_INFINITY;
            for class in 0..DETECTION_CLASSES {
                let value = raw_out[(4 + class) * DETECTION_ANCHORS + anchor];
                if value.is_finite() && value > score {
                    score = value;
                    class_id = class;
                }
            }
            if !score.is_finite() {
                continue;
            }
            // ONNX exports may expose either probabilities or logits.
            if !(0.0..=1.0).contains(&score) {
                score = 1.0 / (1.0 + (-score).exp());
            }
            if score >= self.config.detection.confidence_threshold {
                let net_w = self.config.tier1_width as f32;
                let net_h = self.config.tier1_height as f32;
                let cx = center_x[anchor];
                let cy = center_y[anchor];
                let width = widths[anchor];
                let height = heights[anchor];
                if !cx.is_finite() || !cy.is_finite() || !width.is_finite() || !height.is_finite() {
                    continue;
                }
                let x1 = (cx - width * 0.5).clamp(0.0, net_w);
                let y1 = (cy - height * 0.5).clamp(0.0, net_h);
                let x2 = (cx + width * 0.5).clamp(0.0, net_w);
                let y2 = (cy + height * 0.5).clamp(0.0, net_h);

                let mapped = prep.letterbox.invert_box([x1, y1, x2, y2]);
                candidates.push(BoundingBox {
                    x1: mapped[0],
                    y1: mapped[1],
                    x2: mapped[2],
                    y2: mapped[3],
                    confidence: score,
                    class_id,
                });
            }
        }

        let boxes = if self.config.detection.is_end_to_end_nms_free {
            candidates.sort_unstable_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(self.config.detection.max_detections);
            candidates
        } else {
            diou_nms(
                candidates,
                self.config.detection.diou_nms_threshold,
                self.config.detection.max_detections,
            )
        };

        let postprocess_us = t_post.elapsed().as_micros() as u64;
        let telemetry = LatencyBreakdown {
            queue_wait_us,
            decode_us: prep.decode_us,
            resize_and_transpose_us: prep.resize_and_transpose_us,
            inference_tier1_us,
            inference_tier2_us: None,
            postprocess_us,
            total_us: total_start.elapsed().as_micros() as u64 + queue_wait_us,
        };

        Ok((DetectionResult { boxes }, telemetry))
    }

    /// Extracts an L2-normalized embedding vector $\hat{v} \in \mathbb{R}^D$ for visual similarity search.
    pub fn embed_single(&self, raw_bytes: &[u8]) -> VisionResult<(Vec<f32>, LatencyBreakdown)> {
        self.ensure_mode(crate::config::OperationalMode::EmbeddingSearch)?;
        let total_start = Instant::now();
        let mut guard = self.tier1_pool.try_acquire()?;

        let prep = preprocess_single_into_slot(
            raw_bytes,
            self.config.max_image_pixels,
            self.config.tier1_width,
            self.config.tier1_height,
            false,
            &self.config.normalization,
            &mut guard,
            0,
        )?;

        let t_inf = Instant::now();
        let vec = self
            .tier1_backend
            .run_batch(guard.active_batch_slice(1)?, 1)?;
        if vec.len() != self.config.embedding_dim {
            return Err(crate::error::VisionError::Inference(format!(
                "embedding model returned {} values; expected {}",
                vec.len(),
                self.config.embedding_dim
            )));
        }
        let mut vec = vec;
        let inference_tier1_us = t_inf.elapsed().as_micros() as u64;

        let t_post = Instant::now();
        l2_normalize_in_place(&mut vec);
        let postprocess_us = t_post.elapsed().as_micros() as u64;

        let telemetry = LatencyBreakdown {
            queue_wait_us: 0,
            decode_us: prep.decode_us,
            resize_and_transpose_us: prep.resize_and_transpose_us,
            inference_tier1_us,
            inference_tier2_us: None,
            postprocess_us,
            total_us: total_start.elapsed().as_micros() as u64,
        };

        Ok((vec, telemetry))
    }

    fn ensure_mode(&self, required: crate::config::OperationalMode) -> VisionResult<()> {
        if self.config.mode != required {
            return Err(crate::error::VisionError::Config(format!(
                "operation requires mode {required:?}, configured mode is {:?}",
                self.config.mode
            )));
        }
        Ok(())
    }
}
