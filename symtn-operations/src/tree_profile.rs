use std::time::Duration;

#[derive(Clone, Copy, Debug, Default)]
pub struct TreeTransformReplayProfile {
    pub total: Duration,
    pub cache_lookup: Duration,
    pub validate: Duration,
    pub single_total: Duration,
    pub multi_workspace_prepare: Duration,
    pub multi_pack: Duration,
    pub multi_coefficient_prepare: Duration,
    pub multi_matmul_total: Duration,
    /// Zero for the built-in replays: no current route times this phase
    /// separately.
    pub multi_dense_view_setup: Duration,
    pub multi_dense_matmul_call: Duration,
    /// Zero for the built-in replays: no current route times this phase
    /// separately.
    pub multi_scalar_recoupling: Duration,
    pub multi_scatter: Duration,
    /// Zero for the built-in replays: no current route times this phase
    /// separately.
    pub strided_view_setup: Duration,
    pub strided_kernel: Duration,
    pub single_blocks: usize,
    pub multi_blocks: usize,
    pub packed_columns: usize,
    pub scattered_columns: usize,
}

impl TreeTransformReplayProfile {
    #[inline]
    pub fn accumulate(&mut self, other: &Self) {
        self.total += other.total;
        self.cache_lookup += other.cache_lookup;
        self.validate += other.validate;
        self.single_total += other.single_total;
        self.multi_workspace_prepare += other.multi_workspace_prepare;
        self.multi_pack += other.multi_pack;
        self.multi_coefficient_prepare += other.multi_coefficient_prepare;
        self.multi_matmul_total += other.multi_matmul_total;
        self.multi_dense_view_setup += other.multi_dense_view_setup;
        self.multi_dense_matmul_call += other.multi_dense_matmul_call;
        self.multi_scalar_recoupling += other.multi_scalar_recoupling;
        self.multi_scatter += other.multi_scatter;
        self.strided_view_setup += other.strided_view_setup;
        self.strided_kernel += other.strided_kernel;
        self.single_blocks += other.single_blocks;
        self.multi_blocks += other.multi_blocks;
        self.packed_columns += other.packed_columns;
        self.scattered_columns += other.scattered_columns;
    }
}
