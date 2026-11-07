use crate::error::{VisionError, VisionResult};
use image::ImageReader;
use std::io::Cursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Supported wire image formats detected via magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    Jpeg,
    Png,
    Webp,
}

/// EXIF orientation tag values (1..=8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExifOrientation {
    Normal = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90Cw = 6,
    Transverse = 7,
    Rotate270Cw = 8,
}

impl ExifOrientation {
    #[must_use]
    pub fn from_u16(val: u16) -> Self {
        match val {
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90Cw,
            7 => Self::Transverse,
            8 => Self::Rotate270Cw,
            _ => Self::Normal,
        }
    }
}

/// Validated metadata extracted from image headers prior to full pixel allocation.
#[derive(Debug, Clone, Copy)]
pub struct ImageHeader {
    pub width: u32,
    pub height: u32,
    pub kind: ImageKind,
    pub orientation: ExifOrientation,
}

/// Parses magic bytes, dimensions, and EXIF orientation without allocating full frame buffers.
pub fn inspect_header(raw_bytes: &[u8], max_pixels: u64) -> VisionResult<ImageHeader> {
    if raw_bytes.len() < 12 {
        return Err(VisionError::InvalidHeader(
            "Input buffer smaller than 12 bytes".into(),
        ));
    }

    let kind = detect_magic_bytes(raw_bytes)?;
    let orientation = if kind == ImageKind::Jpeg {
        parse_jpeg_exif_orientation(raw_bytes)
    } else {
        ExifOrientation::Normal
    };

    let (width, height) = match kind {
        ImageKind::Jpeg => {
            let mut decoder = JpegDecoder::new(raw_bytes);
            decoder
                .decode_headers()
                .map_err(|e| VisionError::JpegDecode(e.to_string()))?;
            let info = decoder.info().ok_or_else(|| {
                VisionError::InvalidHeader("Missing JPEG info after header decode".into())
            })?;
            (u32::from(info.width), u32::from(info.height))
        }
        ImageKind::Png | ImageKind::Webp => {
            let reader = ImageReader::new(Cursor::new(raw_bytes))
                .with_guessed_format()
                .map_err(|e| VisionError::InvalidHeader(e.to_string()))?;
            reader.into_dimensions()?
        }
    };

    if width == 0 || height == 0 {
        return Err(VisionError::ZeroDimension { width, height });
    }

    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        return Err(VisionError::ImageTooLarge {
            width,
            height,
            pixels,
            max_pixels,
        });
    }

    Ok(ImageHeader {
        width,
        height,
        kind,
        orientation,
    })
}

fn detect_magic_bytes(bytes: &[u8]) -> VisionResult<ImageKind> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Ok(ImageKind::Jpeg)
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Ok(ImageKind::Png)
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Ok(ImageKind::Webp)
    } else {
        Err(VisionError::InvalidHeader(
            "Unsupported magic bytes (expected JPEG, PNG, or WebP)".into(),
        ))
    }
}

/// Lightweight, zero-allocation scanner for the JPEG APP1 EXIF Orientation tag (`0x0112`).
fn parse_jpeg_exif_orientation(bytes: &[u8]) -> ExifOrientation {
    let mut idx = 2;
    while idx + 4 < bytes.len() {
        if bytes[idx] != 0xFF {
            break;
        }
        let marker = bytes[idx + 1];
        if marker == 0xDA || marker == 0xD9 {
            // Reached Start of Scan (SOS) or End of Image (EOI)
            break;
        }
        let seg_len = u16::from_be_bytes([bytes[idx + 2], bytes[idx + 3]]) as usize;
        if seg_len < 2 || idx + 2 + seg_len > bytes.len() {
            break;
        }
        // APP1 marker (0xFFE1) with "Exif\0\0" header
        if marker == 0xE1 && seg_len >= 14 {
            let payload = &bytes[idx + 4..idx + 2 + seg_len];
            if payload.starts_with(b"Exif\0\0") {
                if let Some(orient) = parse_tiff_orientation(&payload[6..]) {
                    return orient;
                }
            }
        }
        idx += 2 + seg_len;
    }
    ExifOrientation::Normal
}

fn parse_tiff_orientation(tiff: &[u8]) -> Option<ExifOrientation> {
    if tiff.len() < 8 {
        return None;
    }
    let little_endian = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let read_u16 = |offset: usize| -> Option<u16> {
        let slice = tiff.get(offset..offset + 2)?;
        Some(if little_endian {
            u16::from_le_bytes([slice[0], slice[1]])
        } else {
            u16::from_be_bytes([slice[0], slice[1]])
        })
    };
    let read_u32 = |offset: usize| -> Option<u32> {
        let slice = tiff.get(offset..offset + 4)?;
        Some(if little_endian {
            u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
        } else {
            u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]])
        })
    };

    let ifd_offset = read_u32(4)? as usize;
    let num_entries = read_u16(ifd_offset)? as usize;
    let entries_start = ifd_offset + 2;

    for i in 0..num_entries.min(64) {
        let entry_offset = entries_start + i * 12;
        let tag = read_u16(entry_offset)?;
        if tag == 0x0112 {
            let val = read_u16(entry_offset + 8)?;
            return Some(ExifOrientation::from_u16(val));
        }
    }
    None
}

/// Decodes an image into a caller-provided reusable RGB8 `Vec<u8>` scratchpad.
/// Returns the final `(width, height)` after applying EXIF orientation transforms.
pub fn decode_into_scratchpad(
    raw_bytes: &[u8],
    max_pixels: u64,
    rgb_scratch: &mut Vec<u8>,
    orient_scratch: &mut Vec<u8>,
) -> VisionResult<(u32, u32)> {
    let header = inspect_header(raw_bytes, max_pixels)?;
    let required_bytes = (header.width as usize)
        .saturating_mul(header.height as usize)
        .saturating_mul(3);

    rgb_scratch.clear();
    if rgb_scratch.capacity() < required_bytes {
        rgb_scratch.reserve(required_bytes - rgb_scratch.capacity());
    }
    rgb_scratch.resize(required_bytes, 0);

    match header.kind {
        ImageKind::Jpeg => {
            let options = DecoderOptions::default()
                .jpeg_set_out_colorspace(ColorSpace::RGB)
                .set_max_width(header.width as usize)
                .set_max_height(header.height as usize);
            let mut decoder = JpegDecoder::new_with_options(raw_bytes, options);
            decoder
                .decode_into(rgb_scratch)
                .map_err(|e| VisionError::JpegDecode(e.to_string()))?;
        }
        ImageKind::Png | ImageKind::Webp => {
            let dynamic = image::load_from_memory(raw_bytes)?;
            // Keep transparent product-photo backgrounds consistent. Simply
            // dropping alpha leaves arbitrary hidden RGB values that can
            // change the embedding even though those pixels are invisible.
            let rgba = dynamic.to_rgba8();
            let raw = rgba.as_raw();
            if raw.len() != (header.width as usize) * (header.height as usize) * 4 {
                return Err(VisionError::InvalidHeader(
                    "Decoded RGBA buffer size mismatch".into(),
                ));
            }
            for (rgba_pixel, rgb_pixel) in raw.chunks_exact(4).zip(rgb_scratch.chunks_exact_mut(3))
            {
                let alpha = u16::from(rgba_pixel[3]);
                for channel in 0..3 {
                    let foreground = u16::from(rgba_pixel[channel]);
                    let background = 255_u16;
                    rgb_pixel[channel] =
                        ((foreground * alpha + background * (255 - alpha) + 127) / 255) as u8;
                }
            }
        }
    }

    if header.orientation == ExifOrientation::Normal {
        return Ok((header.width, header.height));
    }

    apply_exif_orientation(
        rgb_scratch,
        orient_scratch,
        header.width,
        header.height,
        header.orientation,
    )
}

fn apply_exif_orientation(
    rgb_buf: &mut Vec<u8>,
    scratch: &mut Vec<u8>,
    width: u32,
    height: u32,
    orientation: ExifOrientation,
) -> VisionResult<(u32, u32)> {
    let w = width as usize;
    let h = height as usize;
    let total_bytes = w * h * 3;
    scratch.clear();
    if scratch.capacity() < total_bytes {
        scratch.reserve(total_bytes - scratch.capacity());
    }
    scratch.resize(total_bytes, 0);

    let (out_w, out_h) = match orientation {
        ExifOrientation::Normal => return Ok((width, height)),
        ExifOrientation::FlipHorizontal => {
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 3;
                    let dst = (y * w + (w - 1 - x)) * 3;
                    scratch[dst..dst + 3].copy_from_slice(&rgb_buf[src..src + 3]);
                }
            }
            (width, height)
        }
        ExifOrientation::Rotate180 => {
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 3;
                    let dst = ((h - 1 - y) * w + (w - 1 - x)) * 3;
                    scratch[dst..dst + 3].copy_from_slice(&rgb_buf[src..src + 3]);
                }
            }
            (width, height)
        }
        ExifOrientation::FlipVertical => {
            for y in 0..h {
                let src_row = y * w * 3;
                let dst_row = (h - 1 - y) * w * 3;
                scratch[dst_row..dst_row + w * 3]
                    .copy_from_slice(&rgb_buf[src_row..src_row + w * 3]);
            }
            (width, height)
        }
        ExifOrientation::Rotate90Cw => {
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 3;
                    let dst = (x * h + (h - 1 - y)) * 3;
                    scratch[dst..dst + 3].copy_from_slice(&rgb_buf[src..src + 3]);
                }
            }
            (height, width)
        }
        ExifOrientation::Rotate270Cw => {
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 3;
                    let dst = ((w - 1 - x) * h + y) * 3;
                    scratch[dst..dst + 3].copy_from_slice(&rgb_buf[src..src + 3]);
                }
            }
            (height, width)
        }
        ExifOrientation::Transpose => {
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 3;
                    let dst = (x * h + y) * 3;
                    scratch[dst..dst + 3].copy_from_slice(&rgb_buf[src..src + 3]);
                }
            }
            (height, width)
        }
        ExifOrientation::Transverse => {
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 3;
                    let dst = ((w - 1 - x) * h + (h - 1 - y)) * 3;
                    scratch[dst..dst + 3].copy_from_slice(&rgb_buf[src..src + 3]);
                }
            }
            (height, width)
        }
    };

    std::mem::swap(rgb_buf, scratch);
    Ok((out_w, out_h))
}
