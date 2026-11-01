//! Structure is reported before values. The finite-input refusals
//! themselves are `non_finite_matrix.rs` (multiplicity-free, U(1)/SU(2)) and
//! `checked_generic_facade/nonfinite.rs` (checked Generic).

use super::*;
use symtn::typed::HermitianTol;

in_both_modes! {
    /// A nonfinite non-endomorphism is refused by `eig`/`eigh` as a
    /// non-endomorphism and a nonfinite wide block by `left_polar` as a shape
    /// error; a family with no structural admission (`svd_vals`) refuses
    /// the values.
    fn structural_admission_precedes_the_finite_input_stage<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let row = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 1)]).unwrap();
        let col = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 2)]).unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&row], [&col], |_, _| f64::NAN)
                .unwrap();
        let space_mismatch = |message| OperationError::SpaceMismatch { message };
        let eigh = space_mismatch("eigh requires an endomorphism (codomain == domain)");
        let eig = space_mismatch("eig requires an endomorphism (codomain == domain)");
        assert_eq!(operation_err(wide.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)), eigh);
        assert_eq!(operation_err(wide.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)), eigh);
        assert_eq!(operation_err(wide.eig_vals(engine, &[0], &[1])), eig);
        assert_eq!(operation_err(wide.eig_full(engine, &[0], &[1])), eig);
        assert_eq!(
            operation_err(wide.left_polar(engine, &[0], &[1])),
            OperationError::InvalidArgument {
                message: "left_polar requires rows >= columns in every coupled-sector matrix"
            }
        );
        assert_eq!(
            operation_err(wide.svd_vals(engine, &[0], &[1])),
            OperationError::InvalidArgument { message: "svd input components must be finite" }
        );
    }
}
