#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseDotConfig {
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
    // Elementwise conjugation of the operands, folded into the contraction
    // kernel (BLAS Aᴴ forms / conjugating accumulator) rather than
    // materialized. Flags are in dot-operand order, i.e. `lhs_conj` applies to
    // the first operand passed to `dot_general_into`, which may be the caller's
    // rhs after a route-order swap — the caller is responsible for that mapping.
    lhs_conj: bool,
    rhs_conj: bool,
}

impl DenseDotConfig {
    pub fn new(
        lhs_contracting_dims: Vec<usize>,
        rhs_contracting_dims: Vec<usize>,
        lhs_batch_dims: Vec<usize>,
        rhs_batch_dims: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
            lhs_conj: false,
            rhs_conj: false,
        }
    }

    /// Set elementwise conjugation of the operands (dot-operand order).
    pub fn with_conjugation(mut self, lhs_conj: bool, rhs_conj: bool) -> Self {
        self.lhs_conj = lhs_conj;
        self.rhs_conj = rhs_conj;
        self
    }

    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    #[inline]
    pub fn lhs_conj(&self) -> bool {
        self.lhs_conj
    }

    #[inline]
    pub fn rhs_conj(&self) -> bool {
        self.rhs_conj
    }

    #[inline]
    pub(crate) fn lhs_contracting_dims(&self) -> &[usize] {
        &self.lhs_contracting_dims
    }

    #[inline]
    pub(crate) fn rhs_contracting_dims(&self) -> &[usize] {
        &self.rhs_contracting_dims
    }

    #[inline]
    pub(crate) fn lhs_batch_dims(&self) -> &[usize] {
        &self.lhs_batch_dims
    }

    #[inline]
    pub(crate) fn rhs_batch_dims(&self) -> &[usize] {
        &self.rhs_batch_dims
    }
}
