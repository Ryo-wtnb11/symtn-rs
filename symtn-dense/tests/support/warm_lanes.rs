//! The warm multi-lane grouped GEMM fixture: measured process-wide by
//! `tests/warm_lane_allocations.rs`, and its lane placement checked by the
//! crate's block GEMM tests, which can read the placement. One definition,
//! so both measure the same workload.

use super::DenseGemmBatchJob;

/// Worker threads of the fixture's pool.
pub const THREADS: usize = 4;

/// Elements of each operand and of the destination.
pub const LEN: usize = 64;

/// Four heterogeneous jobs, so the batch is grouped, not one strided run,
/// each in its own 16-element span of every operand. The blocks are small
/// enough that the provider's GEMM needs no packing buffer of its own.
pub fn jobs() -> Vec<DenseGemmBatchJob> {
    [(2, 4, 2), (3, 3, 2), (2, 3, 3), (3, 2, 3)]
        .into_iter()
        .enumerate()
        .map(|(index, (m, k, n))| DenseGemmBatchJob {
            rows: m,
            contracted: k,
            cols: n,
            dst_offset: 16 * index,
            lhs_offset: 16 * index,
            rhs_offset: 16 * index,
        })
        .collect()
}
