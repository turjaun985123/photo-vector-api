use crate::config::NormalizationParams;
use crate::error::{VisionError, VisionResult};
use crate::memory::AlignedBatchBuffer;
use crate::pipeline::decode::decode_into_scratchpad;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use std::cell::RefCell;
use std::time::Instant;

/// Affine letterbox metadata for preserving aspect ratio during object detection.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_left: u32,
    pub pad_top: u32,
    pub orig_width: u32,
    pub orig_height: u32,
}

impl LetterboxTransform {
    /// Computes letterbox scaling and border offsets for mapping `(orig_w, orig_h)` into `(net_w, net_h)`.
    #[must_use]
    pub fn compute(orig_width: u32, orig_height: u32, net_width: u32, net_height: u32) -> Self {
        let scale_w = net_width as f32 / orig_width.max(1) as f32;
        let scale_h = net_height as f32 / orig_height.max(1) as f32;
        let scale = scale_w.min(scale_h);

        let scaled_w = ((orig_width as f32 * scale).round() as u32).min(net_width);
        let scaled_h = ((orig_height as f32 * scale).round() as u32).min(net_height);

        let pad_left = (net_width - scaled_w) / 2;
        let pad_top = (net_height - scaled_h) / 2;

        Self {
            scale,
            pad_left,
            pad_top,
            orig_width,
            orig_height,
        }
    }

    /// Maps a network-space bounding box `[x1, y1, x2, y2]` back to original image pixel coordinates.
    #[inline]
    #[must_use]
    pub fn invert_box(&self, net_box: [f32; 4]) -> [f32; 4] {
        let inv_scale = 1.0 / self.scale.max(1e-6);
        let x1 =
            ((net_box[0] - self.pad_left as f32) * inv_scale).clamp(0.0, self.orig_width as f32);
        let y1 =
            ((net_box[1] - self.pad_top as f32) * inv_scale).clamp(0.0, self.orig_height as f32);
        let x2 =
            ((net_box[2] - self.pad_left as f32) * inv_scale).clamp(0.0, self.orig_width as f32);
        let y2 =
            ((net_box[3] - self.pad_top as f32) * inv_scale).clamp(0.0, self.orig_height as f32);
        [x1, y1, x2, y2]
    }
}

/// Per-worker thread-local scratchpad that retains heap capacity across frames.
struct ThreadScratchpad {
    decode_rgb: Vec<u8>,
    orient_rgb: Vec<u8>,
    resized_rgb: Vec<u8>,
    letterbox_rgb: Vec<u8>,
    resizer: Resizer,
}

impl ThreadScratchpad {
    fn new() -> Self {
        Self {
            decode_rgb: Vec::with_capacity(4096 * 4096 * 3),
            orient_rgb: Vec::with_capacity(1920 * 1080 * 3),
            resized_rgb: Vec::with_capacity(640 * 640 * 3),
            letterbox_rgb: Vec::with_capacity(640 * 640 * 3),
            resizer: Resizer::new(),
        }
    }
}

thread_local! {
    static WORKER_SCRATCH: RefCell<ThreadScratchpad> = RefCell::new(ThreadScratchpad::new());
}

/// Return metadata from preprocessing a single frame.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessOutput {
    pub decode_us: u64,
    pub resize_and_transpose_us: u64,
    pub letterbox: LetterboxTransform,
}

/// Decodes, SIMD-resizes, and performs Fused Multiply-Add (`HWC u8 -> NCHW f32`) directly
/// into `slot_idx` of `batch_buffer` with zero heap allocations after warm-up.
pub fn preprocess_single_into_slot(
    raw_bytes: &[u8],
    max_pixels: u64,
    target_width: u32,
    target_height: u32,
    preserve_aspect_letterbox: bool,
    norm: &NormalizationParams,
    batch_buffer: &mut AlignedBatchBuffer,
    slot_idx: usize,
) -> VisionResult<PreprocessOutput> {
    let (plane_r, plane_g, plane_b) = batch_buffer.slot_planes_mut(slot_idx)?;
    preprocess_Batch_Slot(
        raw_bytes,
        max_pixels,
        target_width,
        target_height,
        preserve_aspect_letterbox,
        norm,
        plane_r,
        plane_g,
        plane_b,
    )
}

/// Core thread-local implementation writing directly into disjoint `(plane_r, plane_g, plane_b)` slices
/// so Rayon threads can populate different slots of the same batch buffer in parallel.
#[allow(non_snake_case)]
pub fn preprocess_batch_Slot(
    raw_bytes: &[u8],
    max_pixels: u64,
    target_width: u32,
    target_height: u32,
    preserve_aspect_letterbox: bool,
    norm: &NormalizationParams,
    plane_r: &mut [f32],
    plane_g: &mut [f32],
    plane_b: &mut [f32],
) -> VisionResult<PreprocessOutput> {
    preprocess_Batch_Slot(
        raw_bytes,
        max_pixels,
        target_width,
        target_height,
        preserve_aspect_letterbox,
        norm,
        plane_r,
        plane_g,
        plane_b,
    )
}

#[allow(non_snake_case)]
fn preprocess_Batch_Slot(
    raw_bytes: &[u8],
    max_pixels: u64,
    target_width: u32,
    target_height: u32,
    preserve_aspect_letterbox: bool,
    norm: &NormalizationParams,
    plane_r: &mut [f32],
    plane_g: &mut [f32],
    plane_b: &mut [f32],
) -> VisionResult<PreprocessOutput> {
    let expected_pixels = (target_width as usize) * (target_height as usize);
    if plane_r.len() != expected_pixels
        || plane_g.len() != expected_pixels
        || plane_b.len() != expected_pixels
    {
        return Err(VisionError::MemoryLayout(format!(
            "Target plane length mismatch: expected {expected_pixels}"
        )));
    }

    WORKER_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let ThreadScratchpad {
            decode_rgb,
            orient_rgb,
            resized_rgb,
            letterbox_rgb,
            resizer,
        } = &mut *scratch;

        let t0 = Instant::now();
        let (orig_w, orig_h) =
            decode_into_scratchpad(raw_bytes, max_pixels, decode_rgb, orient_rgb)?;
        let decode_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        let src_image = ImageRef::new(orig_w, orig_h, decode_rgb, PixelType::U8x3)
            .map_err(|e| VisionError::ResizeError(e.to_string()))?;

        let resize_opts =
            ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));

        let letterbox = if preserve_aspect_letterbox {
            let lb = LetterboxTransform::compute(orig_w, orig_h, target_width, target_height);
            let scaled_w = ((orig_w as f32 * lb.scale).round() as u32).clamp(1, target_width);
            let scaled_h = ((orig_h as f32 * lb.scale).round() as u32).clamp(1, target_height);

            let scaled_bytes = (scaled_w as usize) * (scaled_h as usize) * 3;
            resized_rgb.clear();
            resized_rgb.resize(scaled_bytes, 0);

            let mut dst_scaled =
                Image::from_slice_u8(scaled_w, scaled_h, resized_rgb, PixelType::U8x3)
                    .map_err(|e| VisionError::ResizeError(e.to_string()))?;

            resizer
                .resize(&src_image, &mut dst_scaled, &resize_opts)
                .map_err(|e| VisionError::ResizeError(e.to_string()))?;

            let target_bytes = expected_pixels * 3;
            letterbox_rgb.clear();
            // Fill letterbox border with neutral constant 114
            letterbox_rgb.resize(target_bytes, 114);

            let dst_row_stride = (target_width as usize) * 3;
            let src_row_stride = (scaled_w as usize) * 3;
            let pad_left_bytes = (lb.pad_left as usize) * 3;

            for row in 0..(scaled_h as usize) {
                let dst_offset = (lb.pad_top as usize + row) * dst_row_stride + pad_left_bytes;
                let src_offset = row * src_row_stride;
                letterbox_rgb[dst_offset..dst_offset + src_row_stride]
                    .copy_from_slice(&resized_rgb[src_offset..src_offset + src_row_stride]);
            }

            fused_hwc_to_nchw_fma(letterbox_rgb, plane_r, plane_g, plane_b, norm);
            lb
        } else {
            let target_bytes = expected_pixels * 3;
            resized_rgb.clear();
            resized_rgb.resize(target_bytes, 0);

            let mut dst_image =
                Image::from_slice_u8(target_width, target_height, resized_rgb, PixelType::U8x3)
                    .map_err(|e| VisionError::ResizeError(e.to_string()))?;

            resizer
                .resize(&src_image, &mut dst_image, &resize_opts)
                .map_err(|e| VisionError::ResizeError(e.to_string()))?;

            fused_hwc_to_nchw_fma(resized_rgb, plane_r, plane_g, plane_b, norm);
            LetterboxTransform {
                scale: 1.0,
                pad_left: 0,
                pad_top: 0,
                orig_width: orig_w,
                orig_height: orig_h,
            }
        };

        let resize_and_transpose_us = t1.elapsed().as_micros() as u64;

        Ok(PreprocessOutput {
            decode_us,
            resize_and_transpose_us,
            letterbox,
        })
    })
}

/// Single-pass cache-friendly Fused Multiply-Add (`f32::mul_add`) transpose from interleaved
/// `HWC u8` to planar `NCHW f32` with horizontal flip support for Test-Time Augmentation.
#[inline(always)]
pub fn fused_hwc_to_nchw_fma(
    hwc_rgb: &[u8],
    plane_r: &mut [f32],
    plane_g: &mut [f32],
    plane_b: &mut [f32],
    norm: &NormalizationParams,
) {
    let [sr, sg, sb] = norm.fma_scale;
    let [br, bg, bb] = norm.fma_bias;

    for (((rgb, out_r), out_g), out_b) in hwc_rgb
        .chunks_exact(3)
        .zip(plane_r.iter_mut())
        .zip(plane_g.iter_mut())
        .zip(plane_b.iter_mut())
    {
        *out_r = f32::from(rgb[0]).mul_add(sr, br);
        *out_g = f32::from(rgb[1]).mul_add(sg, bg);
        *out_b = f32::from(rgb[2]).mul_add(sb, bb);
    }
}

/// Generates a horizontally flipped copy of an NCHW slot into another NCHW slot for 2x TTA.
pub fn create_horizontal_flip_nchw(
    src_slot: &[f32],
    dst_slot: &mut [f32],
    width: usize,
    height: usize,
) {
    let plane_size = width * height;
    for c in 0..3 {
        let c_offset = c * plane_size;
        for y in 0..height {
            let row_offset = c_offset + y * width;
            for x in 0..width {
                dst_slot[row_offset + (width - 1 - x)] = src_slot[row_offset + x];
            }
        }
    }
}
