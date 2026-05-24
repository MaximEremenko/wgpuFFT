use crate::error::{FftError, Result};
use crate::runtime::large_policy::LargePolicyLimits;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LargeChunkPlan {
    bytes_per_batch: u64,
    batch_count: u64,
    chunk_batch_count: u64,
    staging_size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LargeChunkRange {
    pub batch_start: u64,
    pub batch_count: u64,
    pub byte_offset: u64,
    pub byte_size: u64,
}

impl LargeChunkPlan {
    #[cfg(test)]
    pub(crate) fn new(
        bytes_per_batch: u64,
        batch_count: u64,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        Self::new_with_max_batches(bytes_per_batch, batch_count, limits, None)
    }

    pub(crate) fn new_with_max_batches(
        bytes_per_batch: u64,
        batch_count: u64,
        limits: LargePolicyLimits,
        max_batches: Option<usize>,
    ) -> Result<Self> {
        if batch_count <= 1 {
            return Err(FftError::LargeChunkUnsupported {
                reason: "batch chunking requires at least two independent batches",
                bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }
        if bytes_per_batch == 0 || bytes_per_batch % 4 != 0 {
            return Err(FftError::LargeChunkUnsupported {
                reason: "bytes per batch must be non-zero and 4-byte aligned",
                bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }
        if bytes_per_batch > limits.max_storage_buffer_binding_size {
            return Err(FftError::LargeChunkUnsupported {
                reason: "one batch exceeds the maximum storage-buffer binding size",
                bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }
        if bytes_per_batch > limits.max_buffer_size {
            return Err(FftError::LargeChunkUnsupported {
                reason: "one batch exceeds the maximum buffer size",
                bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }

        let max_batches = match max_batches {
            Some(0) => {
                return Err(FftError::LargeChunkUnsupported {
                    reason: "large chunk max batches must be at least one",
                    bytes_per_batch,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
            }
            Some(value) => u64::try_from(value).unwrap_or(u64::MAX),
            None => u64::MAX,
        };

        let bind_capacity = limits.max_storage_buffer_binding_size / bytes_per_batch;
        let buffer_capacity = limits.max_buffer_size / bytes_per_batch;
        let chunk_batch_count = bind_capacity
            .min(buffer_capacity)
            .min(batch_count)
            .min(max_batches);
        if chunk_batch_count == 0 {
            return Err(FftError::LargeChunkUnsupported {
                reason: "no batch chunk can fit within the active device limits",
                bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }

        let staging_size_bytes = checked_mul(
            bytes_per_batch,
            chunk_batch_count,
            limits.max_storage_buffer_binding_size,
        )?;
        Ok(Self {
            bytes_per_batch,
            batch_count,
            chunk_batch_count,
            staging_size_bytes,
        })
    }

    pub(crate) fn bytes_per_batch(self) -> u64 {
        self.bytes_per_batch
    }

    pub(crate) fn batch_count(self) -> u64 {
        self.batch_count
    }

    pub(crate) fn chunk_batch_count(self) -> u64 {
        self.chunk_batch_count
    }

    pub(crate) fn staging_size_bytes(self) -> u64 {
        self.staging_size_bytes
    }

    pub(crate) fn ranges(self) -> LargeChunkRanges {
        LargeChunkRanges {
            plan: self,
            next_batch: 0,
        }
    }
}

pub(crate) struct LargeChunkRanges {
    plan: LargeChunkPlan,
    next_batch: u64,
}

impl Iterator for LargeChunkRanges {
    type Item = Result<LargeChunkRange>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_batch >= self.plan.batch_count {
            return None;
        }
        let batch_start = self.next_batch;
        let remaining = self.plan.batch_count - batch_start;
        let batch_count = remaining.min(self.plan.chunk_batch_count);
        let byte_offset = match checked_range_mul(
            batch_start,
            self.plan.bytes_per_batch,
            self.plan.staging_size_bytes,
        ) {
            Ok(byte_offset) => byte_offset,
            Err(error) => return Some(Err(error)),
        };
        let byte_size = match checked_range_mul(
            batch_count,
            self.plan.bytes_per_batch,
            self.plan.staging_size_bytes,
        ) {
            Ok(byte_size) => byte_size,
            Err(error) => return Some(Err(error)),
        };
        self.next_batch += batch_count;

        Some(Ok(LargeChunkRange {
            batch_start,
            batch_count,
            byte_offset,
            byte_size,
        }))
    }
}

fn checked_mul(a: u64, b: u64, max_bind_bytes: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(FftError::LargeChunkUnsupported {
        reason: "chunk size overflowed u64",
        bytes_per_batch: a,
        max_bind_bytes,
    })
}

fn checked_range_mul(value: u64, bytes_per_batch: u64, max_bind_bytes: u64) -> Result<u64> {
    value
        .checked_mul(bytes_per_batch)
        .ok_or(FftError::LargeChunkUnsupported {
            reason: "chunk range overflowed u64",
            bytes_per_batch,
            max_bind_bytes,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_bind: u64) -> LargePolicyLimits {
        LargePolicyLimits {
            max_storage_buffer_binding_size: max_bind,
            max_buffer_size: 1 << 30,
        }
    }

    #[test]
    fn plans_exact_and_tail_chunks() {
        let plan = LargeChunkPlan::new(128, 5, limits(256)).unwrap();
        assert_eq!(plan.bytes_per_batch(), 128);
        assert_eq!(plan.batch_count(), 5);
        assert_eq!(plan.chunk_batch_count(), 2);
        assert_eq!(plan.staging_size_bytes(), 256);

        let ranges = plan.ranges().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(
            ranges,
            [
                LargeChunkRange {
                    batch_start: 0,
                    batch_count: 2,
                    byte_offset: 0,
                    byte_size: 256,
                },
                LargeChunkRange {
                    batch_start: 2,
                    batch_count: 2,
                    byte_offset: 256,
                    byte_size: 256,
                },
                LargeChunkRange {
                    batch_start: 4,
                    batch_count: 1,
                    byte_offset: 512,
                    byte_size: 128,
                },
            ]
        );
    }

    #[test]
    fn optional_batch_cap_limits_chunk_size_without_changing_default() {
        let uncapped = LargeChunkPlan::new(128, 7, limits(1024)).unwrap();
        let capped = LargeChunkPlan::new_with_max_batches(128, 7, limits(1024), Some(3)).unwrap();
        assert_eq!(uncapped.chunk_batch_count(), 7);
        assert_eq!(uncapped.staging_size_bytes(), 896);
        assert_eq!(capped.chunk_batch_count(), 3);
        assert_eq!(capped.staging_size_bytes(), 384);
        assert_eq!(
            capped
                .ranges()
                .map(|range| range.unwrap().batch_count)
                .collect::<Vec<_>>(),
            [3, 3, 1]
        );
    }

    #[test]
    fn zero_batch_cap_is_rejected_structurally() {
        assert_eq!(
            LargeChunkPlan::new_with_max_batches(128, 2, limits(1024), Some(0)).unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "large chunk max batches must be at least one",
                bytes_per_batch: 128,
                max_bind_bytes: 1024,
            }
        );
    }

    #[test]
    fn chunk_range_overflow_returns_route_error() {
        let plan = LargeChunkPlan {
            bytes_per_batch: u64::MAX,
            batch_count: 3,
            chunk_batch_count: 1,
            staging_size_bytes: u64::MAX,
        };
        let mut ranges = plan.ranges();

        assert!(ranges.next().unwrap().is_ok());
        assert!(ranges.next().unwrap().is_ok());
        assert_eq!(
            ranges.next().unwrap(),
            Err(FftError::LargeChunkUnsupported {
                reason: "chunk range overflowed u64",
                bytes_per_batch: u64::MAX,
                max_bind_bytes: u64::MAX,
            })
        );
    }

    #[test]
    fn rejects_unsupported_batch_chunk_shapes() {
        assert_eq!(
            LargeChunkPlan::new(128, 1, limits(256)).unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "batch chunking requires at least two independent batches",
                bytes_per_batch: 128,
                max_bind_bytes: 256,
            }
        );
        assert_eq!(
            LargeChunkPlan::new(132, 2, limits(128)).unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "one batch exceeds the maximum storage-buffer binding size",
                bytes_per_batch: 132,
                max_bind_bytes: 128,
            }
        );
        assert_eq!(
            LargeChunkPlan::new(10, 2, limits(128)).unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "bytes per batch must be non-zero and 4-byte aligned",
                bytes_per_batch: 10,
                max_bind_bytes: 128,
            }
        );
    }
}
