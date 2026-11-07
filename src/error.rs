use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Unified, zero-unwrap error taxonomy for the entire vision engine.
#[derive(Debug, Error)]
pub enum VisionError {
    #[error("I/O error on path '{path:?}': {source}")]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: io::Error,
    },

    #[error("Image format unrecognized or corrupted header ({0})")]
    InvalidHeader(String),

    #[error(
        "Image dimensions {width}x{height} ({pixels} px) exceed safety limit of {max_pixels} px"
    )]
    ImageTooLarge {
        width: u32,
        height: u32,
        pixels: u64,
        max_pixels: u64,
    },

    #[error("Image dimension is zero: {width}x{height}")]
    ZeroDimension { width: u32, height: u32 },

    #[error("JPEG SIMD decode failure: {0}")]
    JpegDecode(String),

    #[error("Fallback image decode failure: {0}")]
    FallbackDecode(#[from] image::ImageError),

    #[error("SIMD resize failure: {0}")]
    ResizeError(String),

    #[error("Aligned memory layout or allocation failure: {0}")]
    MemoryLayout(String),

    #[error("Tensor pool exhausted (capacity: {capacity})")]
    PoolExhausted { capacity: usize },

    #[error("Batch index {slot} out of bounds for batch capacity {max_batch}")]
    BatchSlotOutOfBounds { slot: usize, max_batch: usize },

    #[error("Inference engine error: {0}")]
    Inference(String),

    #[error("Batcher channel closed or overloaded: {0}")]
    BatcherClosed(String),

    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Serialization or index persistence error: {0}")]
    Serialization(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type VisionResult<T> = Result<T, VisionError>;
