//! #1361: the multiplicity-free compact-QR matricization fallback and the
//! checked-Generic compact QR enter one CPU linear-algebra
//! session per call, however many coupled sectors they factorize. The
//! direct-region routes are gated through the facade in
//! `symtn/tests/factorization_session_scope.rs`.
//!
//! This is a separate test binary because `cpu_session_stats` is a
//! process-wide counter; the mutex serializes its readers here.
//!
//! Route: an adjoint view keeps the source blocks' offsets and strides but
//! swaps their codomain and domain axes, so its structure has no coupled-sector
//! matrix regions (asserted below). Only the multiplicity-free compact QR has
//! a direct route, taken when `coupled_sector_regions(nout)` is `Some`
//! (`build_compact_factor_plan`), so its fixture takes the per-sector
//! matricization fallback. The checked-Generic entry has no direct route: it
//! always factorizes the per-sector matrices of `generic_input_matricizations`.

#![cfg(feature = "cpu-faer")]

use std::sync::{Arc, Mutex};

use symtn_core::{
    BraidingStyleKind, CheckedGenericFusion, FusionProductSpace, FusionRule, FusionStyleKind,
    FusionTreeHomSpace, RuleIdentity, SectorId, SectorLeg, SectorVec, U1FusionRule, U1Irrep,
};
use symtn_dense::{cpu_session_stats, DefaultDenseExecutor};
use symtn_matrixalgebra::seam::{
    lq_compact_from_source, qr_compact_from_source, BoundDynamicTensorRef, ExecutorLease,
    FactorMode, FactorSource,
};
use symtn_matrixalgebra::{Lq, Qr};
use symtn_tensors::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace};

/// A U(1) tensor bound to its provider, built the way the facade binds one
/// (`bind_multiplicity_free` over the tensor's own fusion space).
struct BoundTensorMap {
    space: BoundDynamicFusionMapSpace<U1FusionRule>,
    data: Vec<f64>,
}

impl BoundTensorMap {
    fn try_new(
        provider: Arc<U1FusionRule>,
        space: DynamicFusionMapSpace,
        data: Vec<f64>,
    ) -> Option<Self> {
        let space = BoundDynamicFusionMapSpace::bind_multiplicity_free(space, provider).ok()?;
        (space.space().required_len().ok()? == data.len()).then_some(Self { space, data })
    }

    fn space(&self) -> &BoundDynamicFusionMapSpace<U1FusionRule> {
        &self.space
    }

    fn data(&self) -> &[f64] {
        &self.data
    }
}

static COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn sessions_during<T>(call: impl FnOnce() -> T) -> u64 {
    let before = cpu_session_stats().sessions_opened;
    let _ = call();
    cpu_session_stats().sessions_opened - before
}

/// A U(1) 2 <- 2 tensor with legs over `charges`, degeneracy 2 each.
fn u1_tensor(charges: &[i32]) -> BoundTensorMap {
    u1_tensor_with(charges, 2, 2)
}

/// [`u1_tensor`] with codomain legs of degeneracy `rows` and domain legs of
/// degeneracy `cols` per charge.
fn u1_tensor_with(charges: &[i32], rows: usize, cols: usize) -> BoundTensorMap {
    let rule = U1FusionRule;
    let leg = |degeneracy| {
        SectorLeg::new(
            charges
                .iter()
                .map(|&q| (U1Irrep::new(q).sector_id(), degeneracy)),
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(rows), leg(rows)]),
        FusionProductSpace::new([leg(cols), leg(cols)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let len = space.required_len().unwrap();
    let data = (0..len)
        .map(|i| ((i * 11 + 5) % 29) as f64 * 0.25 - 3.0)
        .collect();
    BoundTensorMap::try_new(Arc::new(rule), space, data).unwrap()
}

/// A Generic toy rule with one self-dual sector `x` whose `x ⊗ x` channel has
/// multiplicity 2: the smallest rule that takes the Generic entry.
#[path = "../../tests/support/generic_toy_fusion.rs"]
mod generic_toy_fusion;
use generic_toy_fusion::ToyGenericRule;

/// [`ToyGenericRule`] behind the fallible checked-Generic provider surface.
struct CheckedToyRule;

// The plain view too, so its spaces take the plain layout operations
// (`adjoint_view`); every answer is `ToyGenericRule`'s.
impl FusionRule for CheckedToyRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        ToyGenericRule::<0>.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        ToyGenericRule::<0>.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        ToyGenericRule::<0>.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        ToyGenericRule::<0>.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        ToyGenericRule::<0>.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        ToyGenericRule::<0>.nsymbol(left, right, coupled)
    }
}

impl CheckedGenericFusion for CheckedToyRule {
    type Error = std::convert::Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        ToyGenericRule::<0>.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        ToyGenericRule::<0>.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        ToyGenericRule::<0>.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(ToyGenericRule::<0>.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(ToyGenericRule::<0>.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(ToyGenericRule::<0>.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(ToyGenericRule::<0>.nsymbol(left, right, coupled))
    }
}

fn assert_not_matrix_layout<R: FusionRule>(space: &BoundDynamicFusionMapSpace<R>) {
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap();
    assert!(
        regions.is_none(),
        "fixture must not be a coupled-sector matrix layout"
    );
}

fn assert_fallback_and_checked_generic_routes_open_one_session(mut dense: DefaultDenseExecutor) {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    // U(1) charges -1, 0, 1 on each leg: five coupled sectors on a 2 <- 2 map.
    let bound = u1_tensor(&[-1, 0, 1]);
    let adjoint = bound.space().adjoint_view().unwrap();
    assert_not_matrix_layout(&adjoint);
    let input = BoundDynamicTensorRef::try_new(&adjoint, bound.data()).unwrap();
    assert_eq!(
        sessions_during(|| qr_compact_dyn(&mut dense, &input).unwrap()),
        1,
        "multiplicity-free matricization qr_compact"
    );

    // Generic: sectors {0, x} on every leg, so both coupled sectors 0 and x
    // occur, x with fusion multiplicity.
    let leg = |d0, d1| SectorLeg::new([(SectorId::new(0), d0), (SectorId::new(1), d1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2, 2), leg(1, 2)]),
        FusionProductSpace::new([leg(2, 1)]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedToyRule),
        homspace,
    )
    .unwrap();
    let data = (0..space.space().required_len().unwrap())
        .map(|i| 1.0 + ((i * 7 + 3) % 13) as f64 / 8.0)
        .collect::<Vec<f64>>();
    let adjoint = space.adjoint_view().unwrap();
    assert_not_matrix_layout(&adjoint);
    let input = BoundDynamicTensorRef::try_new(&adjoint, &data).unwrap();
    assert_eq!(
        sessions_during(|| qr_compact_checked_generic(&mut dense, dense_source(&input)).unwrap()),
        1,
        "Generic matricization qr_compact"
    );

    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedToyRule),
        space.space().homspace().clone(),
    )
    .unwrap();
    let data = (0..checked.space().required_len().unwrap())
        .map(|i| 1.0 + ((i * 5 + 2) % 11) as f64 / 8.0)
        .collect::<Vec<f64>>();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    assert_eq!(
        sessions_during(|| qr_compact_checked_generic(&mut dense, dense_source(&input)).unwrap()),
        1,
        "checked-Generic qr_compact"
    );
}

#[test]
fn qr_fallback_and_checked_generic_routes_open_one_session_at_one_thread() {
    assert_fallback_and_checked_generic_routes_open_one_session(
        DefaultDenseExecutor::with_threads(1).unwrap(),
    );
}

#[test]
fn qr_fallback_and_checked_generic_routes_open_one_session_at_default_threads() {
    assert_fallback_and_checked_generic_routes_open_one_session(DefaultDenseExecutor::new());
}

// #1389: the streaming per-block loops enter one Tenferro execution scope per
// call. `sessions_opened` still counts one session per dense factorization,
// while `admissions` counts the permit acquisitions (under default threads,
// Rayon pool hops) those sessions cost: one per call.

fn admissions_during<T>(call: impl FnOnce() -> T) -> (u64, u64) {
    let before = cpu_session_stats();
    let _ = call();
    let after = cpu_session_stats();
    (
        after.sessions_opened - before.sessions_opened,
        after.admissions - before.admissions,
    )
}

fn hermitian_u1_tensor(charges: &[i32]) -> BoundTensorMap {
    let tensor = u1_tensor(charges);
    let mut data = tensor.data().to_vec();
    symmetrize(tensor.space().space().structure(), 2, &mut data);
    BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor.space().space().clone(), data).unwrap()
}

fn symmetrize(structure: &symtn_core::BlockStructure, nout: usize, data: &mut [f64]) {
    let regions = structure.coupled_sector_regions(nout).unwrap().unwrap();
    for region in regions.iter() {
        let n = region.rows();
        let block = &mut data[region.range()];
        for c in 0..n {
            for r in 0..c {
                let value = 0.5 * (block[r + n * c] + block[c + n * r]);
                block[r + n * c] = value;
                block[c + n * r] = value;
            }
        }
    }
}

fn toy_leg(d0: usize, d1: usize) -> SectorLeg {
    SectorLeg::new([(SectorId::new(0), d0), (SectorId::new(1), d1)], false)
}

fn toy_data(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| 1.0 + ((i * 7 + 3) % 13) as f64 / 8.0)
        .collect()
}

fn assert_one_admission(label: &str, (sessions, admissions): (u64, u64)) {
    assert!(
        sessions > 1,
        "{label}: fixture must factorize several blocks"
    );
    assert_eq!(admissions, 1, "{label}: {sessions} sessions");
}

fn assert_streaming_sites_admit_once(mut dense: DefaultDenseExecutor) {
    use symtn_matrixalgebra::seam::{
        eigh_full_dyn, left_polar_from_source, pinv_direct_into_dyn,
        svd_compact_dyn_checked_generic,
    };
    use symtn_matrixalgebra::HermitianTol;
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let dense = &mut dense;

    let bound = u1_tensor(&[-1, 0, 1]);
    let direct = BoundDynamicTensorRef::try_new(bound.space(), bound.data()).unwrap();
    let adjoint = bound.space().adjoint_view().unwrap();
    let matricized = BoundDynamicTensorRef::try_new(&adjoint, bound.data()).unwrap();
    assert_one_admission(
        "lq direct",
        admissions_during(|| lq_compact_dyn(dense, &direct).unwrap()),
    );
    assert_one_admission(
        "lq matricized",
        admissions_during(|| lq_compact_dyn(dense, &matricized).unwrap()),
    );
    assert_one_admission(
        "svd matricized",
        admissions_during(|| svd_compact_factors_dyn(dense, &matricized).unwrap()),
    );
    // The shape-based null space factorizes only blocks longer on the null
    // side: a tall map and its (wide) matricized adjoint.
    let tall = u1_tensor_with(&[-1, 0, 1], 3, 2);
    let tall_direct = BoundDynamicTensorRef::try_new(tall.space(), tall.data()).unwrap();
    let tall_adjoint = tall.space().adjoint_view().unwrap();
    let wide = BoundDynamicTensorRef::try_new(&tall_adjoint, tall.data()).unwrap();
    assert_one_admission(
        "left_null",
        admissions_during(|| left_null_dyn(dense, &tall_direct).unwrap()),
    );
    assert_one_admission(
        "right_null",
        admissions_during(|| right_null_dyn(dense, &wide).unwrap()),
    );
    let hermitian = hermitian_u1_tensor(&[-1, 0, 1]);
    let h_direct = BoundDynamicTensorRef::try_new(hermitian.space(), hermitian.data()).unwrap();
    let h_adjoint = hermitian.space().adjoint_view().unwrap();
    let h_matricized = BoundDynamicTensorRef::try_new(&h_adjoint, hermitian.data()).unwrap();
    assert_one_admission(
        "eigh direct",
        admissions_during(|| eigh_full_dyn(dense, &h_direct, HermitianTol::DEFAULT).unwrap()),
    );
    assert_one_admission(
        "eigh matricized",
        admissions_during(|| eigh_full_dyn(dense, &h_matricized, HermitianTol::DEFAULT).unwrap()),
    );

    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([toy_leg(2, 2), toy_leg(1, 2)]),
        FusionProductSpace::new([toy_leg(2, 1)]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedToyRule),
        homspace.clone(),
    )
    .unwrap();
    let data = toy_data(space.space().required_len().unwrap());
    let generic = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let generic_adjoint = space.adjoint_view().unwrap();
    let generic_matricized = BoundDynamicTensorRef::try_new(&generic_adjoint, &data).unwrap();
    assert_one_admission(
        "generic lq direct",
        admissions_during(|| {
            lq_compact_checked_generic(&mut *dense, dense_source(&generic)).unwrap()
        }),
    );
    assert_one_admission(
        "generic lq matricized",
        admissions_during(|| {
            lq_compact_checked_generic(&mut *dense, dense_source(&generic_matricized)).unwrap()
        }),
    );
    assert_one_admission(
        "generic svd matricized",
        admissions_during(|| svd_compact_dyn_checked_generic(dense, &generic_matricized).unwrap()),
    );

    let provider = Arc::new(CheckedToyRule);
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();
    let data = toy_data(checked.space().required_len().unwrap());
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    assert_one_admission(
        "checked lq",
        admissions_during(|| {
            lq_compact_checked_generic(&mut *dense, dense_source(&input)).unwrap()
        }),
    );
    assert_one_admission(
        "checked svd",
        admissions_during(|| svd_compact_dyn_checked_generic(dense, &input).unwrap()),
    );
    assert_one_admission(
        "checked left_null",
        admissions_during(|| left_null_checked_generic(&mut *dense, dense_source(&input)).unwrap()),
    );
    // `homspace` is tall, so the right null space runs on its adjoint.
    let checked_adjoint = checked.adjoint_view().unwrap();
    let wide_input = BoundDynamicTensorRef::try_new(&checked_adjoint, &data).unwrap();
    assert_one_admission(
        "checked right_null",
        admissions_during(|| {
            right_null_checked_generic(&mut *dense, dense_source(&wide_input)).unwrap()
        }),
    );

    // Polar and pinv read canonical coupled-sector storage: a 1 <- 1 map.
    let matrix = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([toy_leg(3, 2)]),
            FusionProductSpace::new([toy_leg(2, 1)]),
        ),
    )
    .unwrap();
    let data = toy_data(matrix.space().required_len().unwrap());
    let input = BoundDynamicTensorRef::try_new(&matrix, &data).unwrap();
    assert_one_admission(
        "checked left_polar",
        admissions_during(|| {
            left_polar_from_source::<symtn_core::CheckedGenericAdmissionMode, _, _, _, _>(
                &mut *dense,
                dense_source(&input),
            )
            .unwrap()
        }),
    );
    let swapped = FusionTreeHomSpace::new(
        matrix.space().homspace().domain().clone(),
        matrix.space().homspace().codomain().clone(),
    );
    let output = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        swapped,
    )
    .unwrap();
    assert_one_admission(
        "checked pinv",
        admissions_during(|| pinv_direct_into_dyn(dense, &input, output, 1e-12).unwrap()),
    );

    let endomorphism = FusionTreeHomSpace::new(
        FusionProductSpace::new([toy_leg(1, 2), toy_leg(2, 1)]),
        FusionProductSpace::new([toy_leg(1, 2), toy_leg(2, 1)]),
    );
    let endo =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, endomorphism)
            .unwrap();
    let mut data = toy_data(endo.space().required_len().unwrap());
    symmetrize(endo.space().structure(), endo.space().nout(), &mut data);
    let input = BoundDynamicTensorRef::try_new(&endo, &data).unwrap();
    assert_one_admission(
        "checked eigh",
        admissions_during(|| eigh_full_checked_generic(&mut *dense, dense_source(&input)).unwrap()),
    );
}

#[test]
fn streaming_factorization_sites_admit_once_per_call_at_one_thread() {
    assert_streaming_sites_admit_once(DefaultDenseExecutor::with_threads(1).unwrap());
}

#[test]
fn streaming_factorization_sites_admit_once_per_call_at_default_threads() {
    assert_streaming_sites_admit_once(DefaultDenseExecutor::new());
}

#[path = "../src/tests/scripted_executor.rs"]
mod scripted_executor;

/// Negative control for the admission gate: without the executor's scope the
/// same streaming site pays one admission per session.
#[test]
fn streaming_site_without_the_executor_scope_admits_per_session() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let bound = u1_tensor(&[-1, 0, 1]);
    let direct = BoundDynamicTensorRef::try_new(bound.space(), bound.data()).unwrap();
    for inner in [
        DefaultDenseExecutor::new(),
        DefaultDenseExecutor::with_threads(1).unwrap(),
    ] {
        // Forwards the dense calls to `inner` but keeps the trait's default,
        // inline `with_linalg_scope`.
        let mut dense = scripted_executor::ScriptedExecutor::with_inner(inner, ());
        let (sessions, admissions) =
            admissions_during(|| lq_compact_dyn(&mut dense, &direct).unwrap());
        assert!(sessions > 1);
        assert_eq!(admissions, sessions);
    }
}

/// `input` as the dense source of a checked entry.
fn dense_source<'a, R, D>(input: &'a BoundDynamicTensorRef<'_, R, D>) -> FactorSource<'a, R, D> {
    FactorSource::Dense(BoundDynamicTensorRef::try_new(input.space(), input.data()).unwrap())
}

// The QR/LQ entries under their pre-#1862 names, as the single entry runs
// them in each mode.
type QrOut<R, D> = Qr<symtn_matrixalgebra::seam::FactorOutput<R, D>>;
type LqOut<R, D> = Lq<symtn_matrixalgebra::seam::FactorOutput<R, D>>;

fn qr_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<QrOut<R, D>, symtn_tensors::OperationError>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::MultiplicityFreeAdmissionMode: FactorMode<R, Error = symtn_tensors::OperationError>,
    D: symtn_matrixalgebra::FactorScalar,
{
    qr_compact_from_source::<symtn_core::MultiplicityFreeAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
}

fn lq_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LqOut<R, D>, symtn_tensors::OperationError>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::MultiplicityFreeAdmissionMode: FactorMode<R, Error = symtn_tensors::OperationError>,
    D: symtn_matrixalgebra::FactorScalar,
{
    lq_compact_from_source::<symtn_core::MultiplicityFreeAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
}

fn qr_compact_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<QrOut<R, D>, <symtn_core::CheckedGenericAdmissionMode as FactorMode<R>>::Error>
where
    L: ExecutorLease<Executor = E>,
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::CheckedGenericAdmissionMode: FactorMode<R>,
    D: symtn_matrixalgebra::FactorScalar,
{
    qr_compact_from_source::<symtn_core::CheckedGenericAdmissionMode, _, _, _, _>(lease, source)
}

fn lq_compact_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<LqOut<R, D>, <symtn_core::CheckedGenericAdmissionMode as FactorMode<R>>::Error>
where
    L: ExecutorLease<Executor = E>,
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::CheckedGenericAdmissionMode: FactorMode<R>,
    D: symtn_matrixalgebra::FactorScalar,
{
    lq_compact_from_source::<symtn_core::CheckedGenericAdmissionMode, _, _, _, _>(lease, source)
}

fn svd_compact_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<symtn_matrixalgebra::seam::SvdFactorsDyn<R, D>, symtn_tensors::OperationError>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::MultiplicityFreeAdmissionMode: FactorMode<R, Error = symtn_tensors::OperationError>,
    D: symtn_matrixalgebra::FactorScalar,
{
    symtn_matrixalgebra::seam::svd_compact_from_source::<
        symtn_core::MultiplicityFreeAdmissionMode,
        _,
        _,
        _,
        _,
    >(dense, dense_source(input))
}

fn left_null_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<symtn_matrixalgebra::BoundDynFactor<R, D>, symtn_tensors::OperationError>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::MultiplicityFreeAdmissionMode: FactorMode<R, Error = symtn_tensors::OperationError>,
    D: symtn_matrixalgebra::FactorScalar,
{
    symtn_matrixalgebra::seam::left_null_from_source::<
        symtn_core::MultiplicityFreeAdmissionMode,
        _,
        _,
        _,
        _,
    >(dense, dense_source(input))
}

fn left_null_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<
    symtn_matrixalgebra::BoundDynFactor<R, D>,
    <symtn_core::CheckedGenericAdmissionMode as FactorMode<R>>::Error,
>
where
    L: ExecutorLease<Executor = E>,
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::CheckedGenericAdmissionMode: FactorMode<R>,
    D: symtn_matrixalgebra::FactorScalar,
{
    symtn_matrixalgebra::seam::left_null_from_source::<
        symtn_core::CheckedGenericAdmissionMode,
        _,
        _,
        _,
        _,
    >(lease, source)
}

fn right_null_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<symtn_matrixalgebra::BoundDynFactor<R, D>, symtn_tensors::OperationError>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::MultiplicityFreeAdmissionMode: FactorMode<R, Error = symtn_tensors::OperationError>,
    D: symtn_matrixalgebra::FactorScalar,
{
    symtn_matrixalgebra::seam::right_null_from_source::<
        symtn_core::MultiplicityFreeAdmissionMode,
        _,
        _,
        _,
        _,
    >(dense, dense_source(input))
}

fn right_null_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<
    symtn_matrixalgebra::BoundDynFactor<R, D>,
    <symtn_core::CheckedGenericAdmissionMode as FactorMode<R>>::Error,
>
where
    L: ExecutorLease<Executor = E>,
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::CheckedGenericAdmissionMode: FactorMode<R>,
    D: symtn_matrixalgebra::FactorScalar,
{
    symtn_matrixalgebra::seam::right_null_from_source::<
        symtn_core::CheckedGenericAdmissionMode,
        _,
        _,
        _,
        _,
    >(lease, source)
}

fn eigh_full_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<
    symtn_matrixalgebra::seam::EighFullDyn<R, D>,
    <symtn_core::CheckedGenericAdmissionMode as FactorMode<R>>::Error,
>
where
    L: ExecutorLease<Executor = E>,
    E: symtn_dense::DenseExecutor + ?Sized,
    symtn_core::CheckedGenericAdmissionMode: FactorMode<R>,
    D: symtn_matrixalgebra::FactorScalar,
{
    symtn_matrixalgebra::seam::eigh_full_from_source::<
        symtn_core::CheckedGenericAdmissionMode,
        _,
        _,
        _,
        _,
    >(lease, source, symtn_matrixalgebra::HermitianTol::DEFAULT)
}
