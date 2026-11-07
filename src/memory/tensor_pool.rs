use crate::error::{VisionError, VisionResult};
use crossbeam_queue::ArrayQueue;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

/// 64-byte alignment guarantees clean AVX2/AVX-512/NEON vector loads and prevents cache-line splits.
pub const CACHE_LINE_ALIGN: usize = 64;

/// A contiguous `[max_batch, 3, height, width]` `f32` buffer allocated on a 64-byte boundary.
pub struct AlignedBatchBuffer {
    ptr: NonNull<f32>,
    layout: Layout,
    len_f32: usize,
    max_batch: usize,
    channels: usize,
    height: usize,
    width: usize,
}

// SAFETY: `AlignedBatchBuffer` exclusively owns its heap allocation (`ptr`) and does not alias
// with any other pointer. Sending or sharing across threads follows standard owned slice rules.
unsafe impl Send for AlignedBatchBuffer {}
// SAFETY: Shared references `&AlignedBatchBuffer` only allow immutable slice access.
unsafe impl Sync for AlignedBatchBuffer {}

impl AlignedBatchBuffer {
    /// Allocates a zeroed, 64-byte cache-line-aligned tensor buffer for `[max_batch, 3, height, width]`.
    pub fn new(
        max_batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> VisionResult<Self> {
        let len_f32 = max_batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(height))
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| {
                VisionError::MemoryLayout("Tensor buffer element count overflow".into())
            })?;

        if len_f32 == 0 {
            return Err(VisionError::MemoryLayout(
                "Cannot allocate zero-length tensor buffer".into(),
            ));
        }

        let byte_size = len_f32
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| VisionError::MemoryLayout("Tensor buffer byte size overflow".into()))?;

        let layout = Layout::from_size_align(byte_size, CACHE_LINE_ALIGN)
            .map_err(|e| VisionError::MemoryLayout(e.to_string()))?;

        // SAFETY: `layout.size()` is verified non-zero above and `CACHE_LINE_ALIGN` is a power of two (64).
        let raw_ptr = unsafe { alloc_zeroed(layout) };
        let non_null_u8 = NonNull::new(raw_ptr).ok_or_else(|| {
            VisionError::MemoryLayout(format!("Failed to allocate {byte_size} aligned bytes"))
        })?;

        Ok(Self {
            ptr: non_null_u8.cast::<f32>(),
            layout,
            len_f32,
            max_batch,
            channels,
            height,
            width,
        })
    }

    #[inline]
    #[must_use]
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    #[must_use]
    pub fn single_image_stride(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Returns the entire contiguous buffer as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `self.ptr` points to `self.len_f32` initialized `f32` elements for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len_f32) }
    }

    /// Returns a slice covering `[0..active_batch, 3, H, W]`.
    #[inline]
    pub fn active_batch_slice(&self, active_batch: usize) -> VisionResult<&[f32]> {
        if active_batch > self.max_batch {
            return Err(VisionError::BatchSlotOutOfBounds {
                slot: active_batch,
                max_batch: self.max_batch,
            });
        }
        let count = active_batch * self.single_image_stride();
        Ok(&self.as_slice()[..count])
    }

    /// Returns the mutable slice for a single batch slot `slot_idx` of length `3 * H * W`.
    #[inline]
    pub fn slot_mut(&mut self, slot_idx: usize) -> VisionResult<&mut [f32]> {
        if slot_idx >= self.max_batch {
            return Err(VisionError::BatchSlotOutOfBounds {
                slot: slot_idx,
                max_batch: self.max_batch,
            });
        }
        let stride = self.single_image_stride();
        let start = slot_idx * stride;
        // SAFETY: `self.ptr` is uniquely owned and `start + stride <= self.len_f32`.
        let full_slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len_f32) };
        Ok(&mut full_slice[start..start + stride])
    }

    /// Splits a batch slot into three mutable planar slices `(plane_r, plane_g, plane_b)`.
    #[inline]
    pub fn slot_planes_mut(
        &mut self,
        slot_idx: usize,
    ) -> VisionResult<(&mut [f32], &mut [f32], &mut [f32])> {
        let plane_len = self.height * self.width;
        let slot = self.slot_mut(slot_idx)?;
        let (r, rest) = slot.split_at_mut(plane_len);
        let (g, b) = rest.split_at_mut(plane_len);
        Ok((r, g, b))
    }
}

impl Drop for AlignedBatchBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated via `alloc_zeroed` with `self.layout`.
        unsafe {
            dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout);
        }
    }
}

/// Lock-free object pool managing pre-allocated 64-byte aligned tensor batch buffers.
pub struct TensorPool {
    queue: Arc<ArrayQueue<AlignedBatchBuffer>>,
    capacity: usize,
}

impl TensorPool {
    /// Pre-allocates `capacity` aligned batch buffers at engine startup.
    pub fn new(
        capacity: usize,
        max_batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> VisionResult<Self> {
        if capacity == 0 {
            return Err(VisionError::MemoryLayout(
                "TensorPool capacity must be at least 1".into(),
            ));
        }
        let queue = Arc::new(ArrayQueue::new(capacity));
        for _ in 0..capacity {
            let buf = AlignedBatchBuffer::new(max_batch, channels, height, width)?;
            let _ = queue.push(buf);
        }
        Ok(Self { queue, capacity })
    }

    /// Leases an `AlignedBatchBuffer` from the lock-free pool, wrapped in an RAII guard.
    pub fn try_acquire(&self) -> VisionResult<PooledBatchGuard> {
        let buffer = self.queue.pop().ok_or(VisionError::PoolExhausted {
            capacity: self.capacity,
        })?;
        Ok(PooledBatchGuard {
            buffer: Some(buffer),
            pool: Arc::clone(&self.queue),
        })
    }

    #[inline]
    #[must_use]
    pub fn available(&self) -> usize {
        self.queue.len()
    }
}

/// RAII guard that automatically returns the `AlignedBatchBuffer` to the lock-free `ArrayQueue` on drop.
pub struct PooledBatchGuard {
    buffer: Option<AlignedBatchBuffer>,
    pool: Arc<ArrayQueue<AlignedBatchBuffer>>,
}

impl Deref for PooledBatchGuard {
    type Target = AlignedBatchBuffer;

    fn deref(&self) -> &Self::Target {
        match &self.buffer {
            Some(buf) => buf,
            None => unreachable!("PooledBatchGuard buffer is always Some until Drop"),
        }
    }
}

impl DerefMut for PooledBatchGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.buffer {
            Some(buf) => buf,
            None => unreachable!("PooledBatchGuard buffer is always Some until Drop"),
        }
    }
}

impl Drop for PooledBatchGuard {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            let _ = self.pool.push(buf);
        }
    }
}
