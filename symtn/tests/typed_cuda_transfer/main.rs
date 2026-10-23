//! Real-device gates for provider-neutral typed ownership transfer.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! typed_cuda_transfer -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

include!("../common/predicate_chains.rs");
include!("../common/predicate_chain_coefficients.rs");

#[path = "../support/report.rs"]
mod report;
use report::Report;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use symtn::typed::HermitianTol;

use num_complex::{Complex32, Complex64};

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    SectorId, U1FusionRule, U1Irrep, Z2Irrep, ZNFusionRule,
};
use symtn::sector::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorCodec, SectorVec,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::TensorScalar;
use symtn::typed::{
    BlockFusionTrees, ContractSpec, CudaStorage, Eigh, GradedSpace, Qr, Runtime, Svd, TensorMap,
    Truncation,
};

#[path = "../../../tests/support/numerics.rs"]
mod numerics;

/// Floating terms per compared entry, per family of device gates: an
/// elementwise combination, a contraction over the fixtures' bond, and a
/// factorization or its reconstruction over the fixtures' blocks.
const ELEMENTWISE_TERMS: usize = 4;
const CONTRACT_TERMS: usize = 16;
const FACTOR_TERMS: usize = 64;

/// The Host truncated SVD the device composition is compared with: the same
/// composition on Host factors, with the kept values widened to `f64` and
/// ordered by provider label.
struct HostSvdTrunc<R: SectorCodec, D> {
    u: TensorMap<R, D>,
    s: TensorMap<R, D>,
    vh: TensorMap<R, D>,
    singular_values: Vec<symtn::typed::SectorSpectrum<R::Sector, f64>>,
    error: f64,
}

fn labelled_f64<R, D>(
    factor: &TensorMap<R, D>,
    to_f64: impl Fn(D) -> f64,
) -> Vec<symtn::typed::SectorSpectrum<<R as SectorCodec>::Sector, f64>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    let mut spectra: Vec<_> = factor
        .diagview()
        .unwrap()
        .into_iter()
        .map(|entry| symtn::typed::SectorSpectrum {
            sector: entry.sector,
            values: entry.values.into_iter().map(&to_f64).collect(),
        })
        .collect();
    spectra.sort_by(|left, right| left.sector.cmp(&right.sector));
    spectra
}

#[derive(Debug, Eq, PartialEq)]
struct LegSnapshot<S> {
    sectors: Vec<S>,
    degeneracies: Vec<usize>,
    is_dual: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct BlockSnapshot<S> {
    key: symtn::typed::BlockKey,
    fusion_trees: BlockFusionTrees<S>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Debug, Eq, PartialEq)]
struct StructuralSnapshot<S> {
    codomain: Vec<LegSnapshot<S>>,
    domain: Vec<LegSnapshot<S>>,
    blocks: Vec<BlockSnapshot<S>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProbeSector;

/// Tiny device operation for the canary below: `to_cuda` takes the Runtime's
/// device lease, on a provider that never calls back into the Runtime.
fn device_lease_probe(runtime: &Runtime) {
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    let probe: TensorMap<_, f64> = TensorMap::zeros(runtime, [&leg], [&leg]).unwrap();
    probe.to_cuda().unwrap();
}

/// One-sector provider whose dimension callback re-enters the Runtime's device
/// lease. A reduction deadlocks here if it calls provider code under that
/// lease. (`cuda_device_ordinal` takes no lock, #1281.)
struct ReentrantDimensionRule {
    runtime: Runtime,
    calls: Arc<AtomicUsize>,
}

impl symtn_core::testing::Provider for ReentrantDimensionRule {}

impl symtn::sector::TypedSectorAdmission for ReentrantDimensionRule {
    type Sector = <Self as symtn::sector::SectorCodec>::Sector;
    type Error = symtn::typed::FusionAlgebraError;
    type Mode = symtn::sector::MultiplicityFreeAdmissionMode;
    fn typed_rule_identity(&self) -> symtn::sector::RuleIdentity {
        symtn::sector::FusionRule::rule_identity(self)
    }
    fn try_encode_label(
        &self,
        sector: &Self::Sector,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::SectorCodec::encode_sector(self, sector)
    }
    fn try_decode_label(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<Self::Sector, Self::Error> {
        symtn::sector::SectorCodec::decode_sector(self, sector)
    }
    fn try_dual_id(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::CheckedFusionAlgebra::try_dual_sector(self, sector)
    }
}

impl FusionRule for ReentrantDimensionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x7520_0000_0000_0001, Arc::<[u8]>::from([]))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl MultiplicityFreeFusionRule for ReentrantDimensionRule {}

impl MultiplicityFreeFusionSymbols for ReentrantDimensionRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }

    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for ReentrantDimensionRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        assert_eq!(
            symtn::typed::__network::cuda_device_ordinal(&self.runtime),
            Some(0)
        );
        device_lease_probe(&self.runtime);
        self.calls.fetch_add(1, Ordering::SeqCst);
        1.0
    }

    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }

    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }

    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

impl CheckedFusionAlgebra for ReentrantDimensionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl SectorCodec for ReentrantDimensionRule {
    type Sector = ProbeSector;

    fn encode_sector(&self, _: &ProbeSector) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(0))
    }

    fn decode_sector(&self, sector: SectorId) -> Result<ProbeSector, FusionAlgebraError> {
        if sector == SectorId::new(0) {
            Ok(ProbeSector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
}

fn structural_snapshot<R, D>(
    tensor: &TensorMap<R, D>,
) -> StructuralSnapshot<<R as SectorCodec>::Sector>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    let leg_snapshot = |leg: GradedSpace<R>| LegSnapshot {
        sectors: leg.sectors().unwrap(),
        degeneracies: leg.degeneracies().to_vec(),
        is_dual: leg.is_dual(),
    };
    StructuralSnapshot {
        codomain: tensor.codomain().into_iter().map(&leg_snapshot).collect(),
        domain: tensor.domain().into_iter().map(leg_snapshot).collect(),
        blocks: (0..tensor.subblock_count())
            .map(|index| {
                let block = tensor.subblock(index).unwrap();
                BlockSnapshot {
                    key: block.key().clone(),
                    fusion_trees: tensor.subblock_fusion_trees(index).unwrap(),
                    offset: block.offset(),
                    shape: block.shape().to_vec(),
                    strides: block.strides().to_vec(),
                }
            })
            .collect(),
    }
}

fn assert_typed_cuda_svd_matches_host<R>(source: &TensorMap<R, f64>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let source_data = source.dense_data().unwrap().to_vec();
    let source_device = source.to_cuda().unwrap();
    let expected = source
        .svd_compact(&codomain_axes(source), &domain_axes(source))
        .unwrap();
    assert_cuda_svd_result(
        source,
        &expected,
        source_device
            .svd_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
            .unwrap(),
    );
    assert_eq!(
        source_device.to_host().unwrap().dense_data().unwrap(),
        source_data
    );
}

/// Host and device factors, as `svd_compact` returns them.
type HostSvdFactors<R, D> = Svd<TensorMap<R, D>>;
type DeviceSvdFactors<R, D> = Svd<TensorMap<R, D, CudaStorage<D>>>;

fn assert_cuda_svd_result<R>(
    source: &TensorMap<R, f64>,
    expected: &HostSvdFactors<R, f64>,
    factors: DeviceSvdFactors<R, f64>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = symtn::typed::__network::runtime_identity(source.runtime());
    for factor in [&factors.u, &factors.s, &factors.vh] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), symtn::expert::Placement::Cuda(0));
    }
    let actual = (
        factors.u.to_host().unwrap(),
        factors.s.to_host().unwrap(),
        factors.vh.to_host().unwrap(),
    );
    numerics::assert_slices_close(
        "actual.1",
        actual.1.materialize().unwrap().dense_data().unwrap(),
        expected.s.materialize().unwrap().dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_eq!(
        structural_snapshot(&actual.0),
        structural_snapshot(&expected.u)
    );
    assert_eq!(
        structural_snapshot(&actual.1),
        structural_snapshot(&expected.s)
    );
    assert_eq!(
        structural_snapshot(&actual.2),
        structural_snapshot(&expected.vh)
    );
    assert!(is_isometric!(actual.0, 1e-10));
    assert!(is_isometric!(actual.2.adjoint().unwrap(), 1e-10));
    let rebuilt = actual
        .0
        .compose(&actual.1)
        .unwrap()
        .compose(&actual.2)
        .unwrap();
    numerics::assert_slices_close(
        "rebuilt",
        rebuilt.materialize().unwrap().dense_data().unwrap(),
        source.materialize().unwrap().dense_data().unwrap(),
        FACTOR_TERMS,
    );
}

// ---------------------------------------------------------------------------
// Complex64 device payload (#1268). Every fixture below has nonzero
// imaginary parts in every stored entry: a real value cast to `Complex64`
// cannot distinguish a conjugation defect from a transpose.
// ---------------------------------------------------------------------------

/// Genuinely complex fill: `im` is never zero for any admitted index tuple.
fn complex_entry(indices: &[usize], seed: f64) -> Complex64 {
    let ramp = indices.iter().map(|&index| index as f64).sum::<f64>();
    Complex64::new(ramp + seed, -(ramp + seed + 0.75))
}

// ---------------------------------------------------------------------------
// Complex64 device factorizations
// ---------------------------------------------------------------------------

/// Deterministic, genuinely complex, generically full-rank fill.
///
/// A ramp fill such as [`complex_entry`] makes every coupled-sector block
/// rank two at most, which neither exercises the positive-diagonal gauge on a
/// full-rank sector nor lets Host factors be compared at all. Every element
/// gets its own hashed value instead.
fn pseudo_random_c64(ordinal: usize) -> Complex64 {
    fn mix(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        state ^= state >> 29;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 32;
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
    let ordinal = ordinal as u64 + 1;
    Complex64::new(mix(ordinal), mix(ordinal.wrapping_add(0x0005_DEEC_E66D)))
}

/// Per-element counter fill; see [`pseudo_random_c64`].
fn distinct_c64_fill<S>() -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> Complex64 {
    let mut ordinal = 0usize;
    move |_, _| {
        ordinal += 1;
        pseudo_random_c64(ordinal)
    }
}

fn assert_device_factor_handles<R, const N: usize>(
    source: &TensorMap<R, Complex64>,
    factors: [&TensorMap<R, Complex64, CudaStorage<Complex64>>; N],
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = symtn::typed::__network::runtime_identity(source.runtime());
    for factor in factors {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), symtn::expert::Placement::Cuda(0));
    }
}

fn assert_c64_svd_matches_host<R>(source: &TensorMap<R, Complex64>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let source_data = source.materialize().unwrap().dense_data().unwrap().to_vec();
    let expected = source
        .svd_compact(&codomain_axes(source), &domain_axes(source))
        .unwrap();
    let device = source.to_cuda().unwrap();
    let Svd {
        u: u_device,
        s: s_device,
        vh: vh_device,
    } = device
        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    assert_device_factor_handles(source, [&u_device, &s_device, &vh_device]);

    let u = u_device.to_host().unwrap();
    let s = s_device.to_host().unwrap();
    let vh = vh_device.to_host().unwrap();
    for (actual, expected) in [(&u, &expected.u), (&s, &expected.s), (&vh, &expected.vh)] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
    // `u`/`vh` keep the raw device gauge, so only gauge-invariant quantities
    // are compared: the spectrum, both orthonormality relations, and `u s vh`.
    numerics::assert_slices_close(
        "s",
        s.materialize().unwrap().dense_data().unwrap(),
        expected.s.materialize().unwrap().dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert!(is_isometric!(u, 1e-10), "U^H U = I");
    assert!(is_isometric!(vh.adjoint().unwrap(), 1e-10), "V^H V = I");
    numerics::assert_slices_close(
        "u.compose(s)  .compose(vh)",
        u.compose(&s)
            .unwrap()
            .compose(&vh)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        &source_data,
        FACTOR_TERMS,
    );
    assert_eq!(device.to_host().unwrap().dense_data().unwrap(), source_data);
}

mod arithmetic;
mod contract;
mod factorization;
mod factorization_svd;
mod transfer;
