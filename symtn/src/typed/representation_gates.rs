use super::*;
use crate::typed::probe::{self, Event};

include!("../../tests/common/predicate_chains.rs");
include!("../../tests/support/report.rs");

#[cfg(feature = "cgc-gen")]
mod compact_admission;
mod compact_transforms;
mod contract;
#[cfg(feature = "cuda")]
mod cuda;
mod cuda_factorize;
mod dense_lease;
#[cfg(feature = "cgc-gen")]
mod factorize_checked;
#[cfg(feature = "cgc-gen")]
mod factorize_checked_eig;
mod factorize_mf_eig;
mod factorize_mf_qr_lq_polar;
mod factorize_mf_svd_null;
mod lazy_adjoint;
mod lazy_adjoint_factorize;
mod lazy_adjoint_spectral;
mod network_restriction;
mod overwrite;
mod preflight_order;

/// The innermost message of an error chain: the cause both fusion modes must
/// agree on, below each mode's own wrapper context (#1862).
fn root_cause(error: &(dyn std::error::Error + 'static)) -> String {
    std::iter::successors(Some(error), |&error| error.source())
        .last()
        .map_or_else(String::new, ToString::to_string)
}

/// Whether `error` is `family`'s documented finite-input refusal, the
/// operation-layer `InvalidArgument` both fusion modes carry.
fn is_finite_input_refusal(error: &(dyn std::error::Error + 'static), family: &str) -> bool {
    matches!(
        cause::<symtn_tensors::OperationError>(error),
        Some(symtn_tensors::OperationError::InvalidArgument { message })
            if *message == format!("{family} input components must be finite")
    )
}

impl<T: ScalarOps> ChainCoefficient for T {
    fn real(value: f64) -> Self {
        T::from_real(value)
    }
}

/// The pre-#1541 tuple order of a two-factor result, for the helpers
/// below that treat QR/LQ and left/right polar factors uniformly.
trait FactorPair<T> {
    fn pair(self) -> (T, T);
}
impl<T> FactorPair<T> for Qr<T> {
    fn pair(self) -> (T, T) {
        (self.q, self.r)
    }
}
impl<T> FactorPair<T> for Lq<T> {
    fn pair(self) -> (T, T) {
        (self.l, self.q)
    }
}
impl<T> FactorPair<T> for LeftPolar<T> {
    fn pair(self) -> (T, T) {
        (self.w, self.p)
    }
}
impl<T> FactorPair<T> for RightPolar<T> {
    fn pair(self) -> (T, T) {
        (self.p, self.wh)
    }
}
use symtn_core::{product_sector, ProductFusionRuleExt};
use symtn_core::{
    BlockKey, BlockSpec, BlockStructure, CU1FusionRule, CU1Irrep, FermionParityFusionRule,
    FusionTreeKey, FusionTreePairKey, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2FusionRule,
    Z2Irrep, ZNFusionRule,
};
use symtn_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};

include!("../../tests/common/spy_executor.rs");

// The kernel sets the factorization gates allow; any other dense entry panics.
const POLAR_SVD: &[Kernel] = &[Kernel::Svd, Kernel::SvdInto, Kernel::SvdVals];
const POLAR: &[Kernel] = &[
    Kernel::Svd,
    Kernel::SvdInto,
    Kernel::SvdVals,
    Kernel::DotGeneral,
    Kernel::Matmul,
    Kernel::MatmulAxpby,
    Kernel::MatmulBatch,
    Kernel::MatmulBatchOps,
];
const EIGH_FULL: &[Kernel] = &[Kernel::Eigh, Kernel::EighInto, Kernel::EighVals];
const EIG_FULL: &[Kernel] = &[Kernel::Eig, Kernel::EigVals];

/// SVD (`POLAR_SVD`) and GEMM (`Kernel::GEMM`) only.
fn polar_spy(counts: &Arc<SpyCounts>) -> SpyExecutor {
    SpyExecutor::counting(counts).only(POLAR, "test only exercises polar")
}

fn eigh_vals_spy(counts: &Arc<SpyCounts>) -> SpyExecutor {
    SpyExecutor::counting(counts).only(&[Kernel::EighVals], "test only exercises eigh_vals")
}

fn eigh_full_spy(counts: &Arc<SpyCounts>) -> SpyExecutor {
    SpyExecutor::counting(counts).only(EIGH_FULL, "test only exercises eigh_full")
}

/// EIG (`EIG_FULL`) only: any other kernel, such as an eigenvector SVD,
/// panics.
fn eig_full_spy(counts: &Arc<SpyCounts>) -> SpyExecutor {
    SpyExecutor::counting(counts).only(EIG_FULL, "test only exercises eig_full")
}

fn eig_vals_spy(counts: &Arc<SpyCounts>) -> SpyExecutor {
    SpyExecutor::counting(counts).only(&[Kernel::EigVals], "test only exercises eig_vals")
}

/// SVD only; the second SVD call fails.
fn fail_second_svd(counts: &Arc<SpyCounts>) -> SpyExecutor {
    SpyExecutor::counting(counts)
        .only(POLAR_SVD, "test only exercises SVD")
        .failing(POLAR_SVD, Some(2), "injected second-sector failure")
}

struct NonCloneHost(Vec<f64>);

impl TensorStorage<f64> for NonCloneHost {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> symtn_core::Placement {
        symtn_core::Placement::Host
    }
}

impl HostReadableStorage<f64> for NonCloneHost {
    fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

fn owned<R, D, S>(tensor: &TensorMap<R, D, S>) -> &Arc<TypedTensorBody<R, D, S>> {
    tensor.owned_body().expect("test fixture must be owned")
}

#[test]
fn physical_projection_publishes_only_after_success_on_receiver_authority() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let half = SU2Irrep::from_twice_spin(1);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(half, 1)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg, &leg], [&leg], |trees, _| {
        if trees.codomain_innerlines()[0].twice_spin() == 0 {
            1.25
        } else {
            -0.75
        }
    })
    .unwrap();
    let target =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg, &leg], [&leg], |_, _| f64::NAN).unwrap();
    let target_body = Arc::clone(owned(&target));
    let target_data = Arc::clone(&target_body.data);
    let physical = source.to_physical_dense().unwrap();
    let projected = target.project_physical_dense(&physical).unwrap();

    assert!(projected.runtime.same_runtime(&target.runtime));
    assert!(Arc::ptr_eq(
        projected.logical_space().provider_arc(),
        target.logical_space().provider_arc()
    ));
    assert_eq!(
        projected.logical_space().space(),
        target.logical_space().space()
    );
    assert!(projected
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(&actual, &expected)| (actual - expected).abs() < 2.0e-12));
    assert!(Arc::ptr_eq(owned(&target), &target_body));
    assert!(Arc::ptr_eq(&owned(&target).data, &target_data));
    assert!(target
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_nan()));

    let failure = target.project_physical_dense(&PhysicalDense {
        shape: vec![2, 2],
        data: vec![0.0; 4],
    });
    assert!(failure.is_err());
    assert!(Arc::ptr_eq(owned(&target), &target_body));
    assert!(Arc::ptr_eq(&owned(&target).data, &target_data));
}

fn u1_lazy_fixture() -> TensorMap<U1FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    let right = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let domain = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(&runtime, [&left, &right], [&domain], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap()
}

#[test]
fn storage_parameter_clone_shares_non_clone_payload() {
    let source = u1_lazy_fixture();
    let tensor: TensorMap<_, _, NonCloneHost> = TensorMap {
        runtime: source.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            NonCloneHost(source.dense_data().unwrap().to_vec()),
        )),
    };

    let twin = tensor.clone();

    assert!(Arc::ptr_eq(owned(&tensor), owned(&twin)));
    assert!(std::ptr::eq(tensor.provider(), twin.provider()));
    assert_eq!(tensor.dense_data().unwrap(), twin.dense_data().unwrap());
}

#[test]
fn typed_placement_is_diagnostic_for_dense_compact_and_lazy_host_storage() {
    let densified = probe::record();
    let source = u1_lazy_fixture();
    let diagonal = source.svd_compact(&[0, 1], &[2]).unwrap().s;
    let lazy = source.adjoint().unwrap();

    assert_eq!(source.placement(), Placement::Host);
    assert_eq!(diagonal.placement(), Placement::Host);
    assert_eq!(lazy.placement(), Placement::Host);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn typed_zeros_like_is_exact_and_representation_preserving() {
    let densified = probe::record();
    let source = u1_lazy_fixture();
    let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let index = std::cell::Cell::new(0usize);
    let dense = TensorMap::from_subblock_fn(
        source.runtime(),
        &source.codomain(),
        &source.domain(),
        |_, _| {
            let i = index.get();
            index.set(i + 1);
            values[i % values.len()]
        },
    )
    .unwrap();
    let source_bits: Vec<_> = dense
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let provider = dense.provider() as *const _;
    let zero = dense.zeros_like().unwrap();
    assert!(zero
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == 0));
    assert_eq!(
        dense
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
    assert!(std::ptr::eq(zero.provider(), provider));
    assert!(zero.runtime().same_runtime(dense.runtime()));
    assert_eq!(zero.logical_space().space(), dense.logical_space().space());

    let complex = dense.convert::<Complex64>();
    let complex = complex.with_data(
        complex
            .dense_data()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, _)| {
                num_complex::Complex64::new(
                    values[i % values.len()],
                    values[(i + 1) % values.len()],
                )
            })
            .collect(),
    );
    let complex_zero = complex.zeros_like().unwrap();
    assert!(complex_zero
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

    let compact = source.svd_compact(&[0, 1], &[2]).unwrap().s;
    let compact = compact.with_spectrum(
        compact
            .spectrum()
            .unwrap()
            .iter()
            .map(|entry| symtn_matrixalgebra::SectorSpectrum {
                sector: entry.sector,
                values: (0..entry.values.len())
                    .map(|i| values[i % values.len()])
                    .collect(),
            })
            .collect(),
    );
    let compact_zero = compact.zeros_like().unwrap();
    assert!(matches!(
        owned(&compact_zero).data.as_ref(),
        TypedData::Diagonal(_)
    ));
    assert!(compact_zero
        .spectrum()
        .unwrap()
        .iter()
        .flat_map(|entry| &entry.values)
        .all(|value| value.to_bits() == 0));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let lazy = dense.adjoint().unwrap();
    let lazy_zero = lazy.zeros_like().unwrap();
    assert!(matches!(lazy_zero.repr, TypedTensorRepr::Adjoint(_)));
    assert!(std::ptr::eq(lazy_zero.provider(), provider));

    let empty_leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 0)]).unwrap();
    let empty =
        TensorMap::from_subblock_fn(source.runtime(), [&empty_leg], [&empty_leg], |_, _| {
            f64::NAN
        })
        .unwrap();
    assert!(empty.dense_data().unwrap().is_empty());
    assert!(empty.zeros_like().unwrap().dense_data().unwrap().is_empty());
}

fn su2_lazy_fixture() -> TensorMap<SU2FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap()
}

fn u1_matrix_fixture(
    codomain: impl IntoIterator<Item = (i32, usize)>,
    domain: impl IntoIterator<Item = (i32, usize)>,
) -> TensorMap<U1FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain = GradedSpace::try_new(
        Arc::clone(&provider),
        codomain
            .into_iter()
            .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    let domain = GradedSpace::try_new(
        provider,
        domain
            .into_iter()
            .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::from_subblock_fn(&runtime, [&codomain], [&domain], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap()
}

fn genuinely_complex<R>(source: &TensorMap<R, f64>) -> TensorMap<R, num_complex::Complex64> {
    TensorMap {
        runtime: source.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            source
                .dense_data()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(index, &value)| num_complex::Complex64::new(value, (index + 1) as f64 / 7.0))
                .collect(),
        )),
    }
}

fn assert_typed_map_close<R, D>(
    actual: &TensorMap<R, D>,
    expected: &TensorMap<R, D>,
    tolerance: f64,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(actual
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.materialize().unwrap().dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < tolerance
        }));
}

fn assert_polar_factors<R, D>(
    source: &TensorMap<R, D>,
    target: &TensorMap<R, D>,
    actual: &(TensorMap<R, D>, TensorMap<R, D>),
    expected: &(TensorMap<R, D>, TensorMap<R, D>),
    left: bool,
) where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let reconstructed = actual.0.compose(&actual.1).unwrap();
    assert_typed_map_close(&reconstructed, target, 1e-10);
    let (positive, isometry) = if left {
        (&actual.1, &actual.0)
    } else {
        (&actual.0, &actual.1)
    };
    assert!(if left {
        is_isometric!(isometry, 1e-11)
    } else {
        is_isometric!(isometry.adjoint().unwrap(), 1e-11)
    });
    assert!(is_hermitian!(positive, 1e-11));
    assert!(positive
        .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .iter()
        .all(|entry| entry.values.iter().all(|&value| value >= -1e-11)));
    for factor in [&actual.0, &actual.1] {
        assert!(factor.owned_body().is_some());
        assert!(Arc::ptr_eq(
            factor.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        let _ = factor.materialize().unwrap().dense_data().unwrap();
    }
    assert_eq!(
        actual.0.logical_space().space(),
        expected.0.logical_space().space()
    );
    assert_eq!(
        actual.1.logical_space().space(),
        expected.1.logical_space().space()
    );
}

fn eager_adjoint_oracle<R, D>(source: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let (space, data) =
        symtn_tensors::adjoint_bound_dyn(source.logical_space(), source.dense_data().unwrap())
            .unwrap();
    TensorMap {
        runtime: source.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    }
}

use crate::test_numerics::numerics;

/// Floating terms per compared scalar of these gates: an inner product or
/// trace over the fixtures' stored entries, at most this many.
const GATE_TERMS: usize = 16;

/// A rank-(1, 1) domain leg against a rank-(1, 1) codomain leg, default split.
const RANK_TWO_COMPOSE: ContractSpec<'static> = ContractSpec {
    lhs: &[1],
    rhs: &[0],
    codomain: &[0],
    domain: &[1],
};

fn fixture() -> TensorMap<Z2FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, 8)]).unwrap();
    let mut state = 0x5eed_0580u64;
    TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], move |_, _| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
    })
    .unwrap()
}

#[test]
fn clone_copies_no_payload_bytes() {
    // What: `clone` is O(1) in the payload. Measured structurally rather
    // than by an allocator, which cannot distinguish "no copy" from "a
    // copy the size of a warm cache line".
    let tensor = fixture();
    let twin = tensor.clone();
    assert!(Arc::ptr_eq(owned(&tensor), owned(&twin)));
    assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&twin).data));
    assert_eq!(
        tensor.dense_data().unwrap().as_ptr(),
        twin.dense_data().unwrap().as_ptr()
    );
    // One payload, however many handles reach it.
    assert_eq!(Arc::strong_count(&owned(&tensor).data), 1);
    assert_eq!(Arc::strong_count(owned(&tensor)), 2);
}

/// A small fermionic fixture whose codomain leg 0 carries only the even
/// sector (θ = 1 everywhere on it) while leg 1 and the domain leg carry
/// the odd sector too — so one tensor exposes both twist short-circuit
/// answers.
fn fz2_fixture() -> TensorMap<symtn_core::FermionParityFusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(symtn_core::FermionParityFusionRule);
    let even_only = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap();
    let mixed = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    let mut state = 0x5eed_0613u64;
    TensorMap::from_subblock_fn(&runtime, [&even_only, &mixed], [&mixed], move |_, _| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
    })
    .unwrap()
}

#[test]
fn unit_insert_and_remove_share_the_dense_payload_arc() {
    let densified = probe::record();
    // What (#580): payload reuse at pointer cost under a rewritten space,
    // through the real operations — a dense payload's `Arc` is shared
    // unchanged through an insert→remove round trip, and the fresh
    // bodies start with cold caches.
    let tensor = fixture();
    let inserted = tensor.insert_unit(1, Side::Domain, Duality::Plain).unwrap();
    assert!(!Arc::ptr_eq(owned(&tensor), owned(&inserted)));
    assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&inserted).data));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let removed = inserted.remove_unit(1).unwrap();
    assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&removed).data));
    // One payload allocation, three bodies holding it.
    assert_eq!(Arc::strong_count(&owned(&tensor).data), 3);
    assert_eq!(
        tensor.dense_data().unwrap().as_ptr(),
        removed.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn a_compact_payload_materializes_exactly_once_for_the_unit_ops() {
    // What (#580, #613): the compact half of the unit-leg payload
    // contract — a `Diagonal` payload is materialized into a *fresh*
    // dense payload (one copy), and the subsequent remove shares that
    // dense `Arc` rather than copying again.
    let s = fixture().svd_compact(&[0], &[1]).unwrap().s;
    let inserted = s.insert_unit(0, Side::Domain, Duality::Plain).unwrap();
    assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&inserted).data));
    assert!(matches!(&*owned(&inserted).data, TypedData::Dense(_)));
    assert!(matches!(&*owned(&s).data, TypedData::Diagonal(_)));
    let removed = inserted.remove_unit(0).unwrap();
    assert!(Arc::ptr_eq(&owned(&inserted).data, &owned(&removed).data));
}

#[test]
fn twist_identity_short_circuit_shares_the_whole_body() {
    // What (#580): both identity answers allocate nothing —
    // the bosonic O(1) arm (Z2) and the fermionic per-block scan when no
    // requested leg touches a twisted sector (fZ2, even-only leg 0) both
    // return a body-sharing clone; a leg that does touch the odd sector
    // publishes a new body.
    let tensor = fixture();
    let twisted = tensor.twist(&[0, 1], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(owned(&tensor), owned(&twisted)));

    let fermionic = fz2_fixture();
    let untouched = fermionic.twist(&[0], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(owned(&fermionic), owned(&untouched)));
    let touched = fermionic.twist(&[1], Direction::Forward).unwrap();
    assert!(!Arc::ptr_eq(owned(&fermionic), owned(&touched)));
}

#[test]
fn a_written_payload_leaves_the_shared_one_untouched() {
    // What: clone-then-modify. Sharing is only sound if a write on one
    // handle cannot be seen through the other — every write route in this
    // module publishes a new payload rather than reaching through the `Arc`.
    let tensor = fixture();
    let twin = tensor.clone();
    let before: Vec<f64> = tensor.dense_data().unwrap().to_vec();

    let scaled = twin.scale(2.0).unwrap();

    assert_eq!(tensor.dense_data().unwrap(), before.as_slice());
    assert_eq!(twin.dense_data().unwrap(), before.as_slice());
    assert_ne!(
        scaled.dense_data().unwrap().as_ptr(),
        tensor.dense_data().unwrap().as_ptr()
    );
    assert!(!Arc::ptr_eq(&owned(&scaled).data, &owned(&tensor).data));
}
