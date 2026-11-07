use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Operational modes supported by the unified vision engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum OperationalMode {
    /// Two-tier speculative cascade classification (Fast Tier-1 -> Entropy Gate -> Heavy Tier-2).
    SpeculativeCascade,
    /// End-to-end object detection with Letterbox scaling and DIoU-NMS.
    ObjectDetection,
    /// L2-normalized embedding extraction and SIMD similarity search.
    EmbeddingSearch,
}

/// Hardware execution provider priority list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderPreference {
    TensorRt,
    Cuda,
    CoreMl,
    DirectMl,
    OpenVino,
    Cpu,
}

/// Per-channel normalization constants alongside precomputed FMA scale and bias.
///
/// Transforms `(pixel / 255.0 - mean) / std` into `pixel.mul_add(scale, bias)`
/// where `scale = 1.0 / (255.0 * std)` and `bias = -mean / std`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub fma_scale: [f32; 3],
    pub fma_bias: [f32; 3],
}

impl NormalizationParams {
    #[must_use]
    pub fn imagenet() -> Self {
        Self::from_mean_std([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }

    #[must_use]
    pub fn clip() -> Self {
        Self::from_mean_std(
            [0.481_454_66, 0.457_827_5, 0.408_210_73],
            [0.268_629_54, 0.261_302_58, 0.275_777_11],
        )
    }

    #[must_use]
    pub fn zero_to_one() -> Self {
        Self::from_mean_std([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    #[must_use]
    pub fn from_mean_std(mean: [f32; 3], std: [f32; 3]) -> Self {
        let mut fma_scale = [0.0_f32; 3];
        let mut fma_bias = [0.0_f32; 3];
        for c in 0..3 {
            let safe_std = if std[c].abs() < 1e-9 { 1.0 } else { std[c] };
            fma_scale[c] = 1.0 / (255.0 * safe_std);
            fma_bias[c] = -mean[c] / safe_std;
        }
        Self {
            mean,
            std,
            fma_scale,
            fma_bias,
        }
    }
}

/// Speculative cascade gating thresholds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CascadeGateConfig {
    /// Minimum top-1 probability required to accept Tier-1 without invoking Tier-2.
    pub min_confidence: f32,
    /// Maximum Shannon entropy allowed to accept Tier-1 without invoking Tier-2.
    pub max_entropy: f32,
    /// Learned temperature scaling factor for calibrating Tier-1 logits.
    pub temperature: f32,
    /// Enable 2x Test-Time Augmentation (original + horizontal flip) on Tier-2.
    pub enable_tier2_tta: bool,
}

impl Default for CascadeGateConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.90,
            max_entropy: 0.35,
            temperature: 1.0,
            enable_tier2_tta: true,
        }
    }
}

/// Detection post-processing thresholds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub diou_nms_threshold: f32,
    pub max_detections: usize,
    pub is_end_to_end_nms_free: bool,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            diou_nms_threshold: 0.45,
            max_detections: 100,
            is_end_to_end_nms_free: false,
        }
    }
}

/// Master configuration for the `rust-vision-engine`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub mode: OperationalMode,
    pub tier1_model_path: Option<PathBuf>,
    pub tier2_model_path: Option<PathBuf>,
    pub labels_path: Option<PathBuf>,
    pub tier1_width: u32,
    pub tier1_height: u32,
    pub tier2_width: u32,
    pub tier2_height: u32,
    pub num_classes: usize,
    pub embedding_dim: usize,
    pub max_batch_size: usize,
    pub max_batch_window_us: u64,
    pub tensor_pool_capacity: usize,
    pub max_image_pixels: u64,
    pub intra_op_threads: usize,
    pub inter_op_threads: usize,
    pub normalization: NormalizationParams,
    pub cascade: CascadeGateConfig,
    pub detection: DetectionConfig,
    pub execution_providers: Vec<ExecutionProviderPreference>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            mode: OperationalMode::SpeculativeCascade,
            tier1_model_path: None,
            tier2_model_path: None,
            labels_path: None,
            tier1_width: 224,
            tier1_height: 224,
            tier2_width: 384,
            tier2_height: 384,
            num_classes: 1000,
            embedding_dim: 384,
            max_batch_size: 32,
            max_batch_window_us: 3_000, // 3ms adaptive window
            tensor_pool_capacity: 16,
            max_image_pixels: 64_000_000, // 64 Megapixels bomb limit
            intra_op_threads: 2,
            inter_op_threads: 1,
            normalization: NormalizationParams::imagenet(),
            cascade: CascadeGateConfig::default(),
            detection: DetectionConfig::default(),
            execution_providers: vec![
                ExecutionProviderPreference::TensorRt,
                ExecutionProviderPreference::Cuda,
                ExecutionProviderPreference::CoreMl,
                ExecutionProviderPreference::DirectMl,
                ExecutionProviderPreference::Cpu,
            ],
        }
    }
}

/// Microsecond-resolution per-stage latency telemetry returned with every inference.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub queue_wait_us: u64,
    pub decode_us: u64,
    pub resize_and_transpose_us: u64,
    pub inference_tier1_us: u64,
    pub inference_tier2_us: Option<u64>,
    pub postprocess_us: u64,
    pub total_us: u64,
}
