//! Gate suite for the provider-typed facade (`symtn::typed`, issue #557).
//!
//! Every external provider here is a `fusion_symbols` category with its own
//! codec in `CategoryRule`, built from the public vocabulary alone — no
//! provider-trait implementation, no crate-internal machinery — so the suite
//! doubles as proof that a downstream application can drive the typed facade
//! with its own category.

use crate::{numerics, ulp};

/// TensorKit's argument-free `transpose(t)`: the full planar rotation, which
/// carries every codomain leg across the boundary and every domain leg back.
macro_rules! full_transpose {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        let codomain_rank = tensor.codomain_rank();
        let codomain_axes: Vec<usize> = (codomain_rank..tensor.rank()).rev().collect();
        let domain_axes: Vec<usize> = (0..codomain_rank).rev().collect();
        tensor.transpose(&codomain_axes, &domain_axes)
    }};
}

include!("../common/predicate_chains.rs");
include!("../common/predicate_chain_coefficients.rs");

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use symtn::typed::HermitianTol;
use symtn::typed::{ContractSpec, Direction, Duality, Side};

use symtn::cache::{StructureCacheInfo, StructureCacheKind};

fn structure_cache_info(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}
use fusion_symbols::category::{BraidingStyle, FusionCategory, FusionStyle};
use fusion_symbols::lie::{GroupId, LieError, LieGroup, LieIrrep};
use symtn::sector::{CU1FusionRule, CU1Irrep, SU2FusionRule, SU2Irrep, SectorId};
use symtn::sector::{
    CanonicalUnitAdmission, CategoryAdapterError, CategoryCodec, CategoryFailure,
    CategoryOperation, CategoryRule, CheckedFusionAlgebra, FusionRule,
    MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorCodec,
};
use symtn::sector::{TableRule, TableSector};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{
    Eig, Eigh, GradedSpace, LeftPolar, Lq, Qr, RightPolar, Svd, TensorMap, Truncation,
};

use crate::report;

use report::Report;

/// Orders the local cache probes in addition to the target-wide serial guard.
static CACHE_LOCK: Mutex<()> = Mutex::new(());

fn cache_lock() -> MutexGuard<'static, ()> {
    CACHE_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Re-executes exactly one cache probe with explicit fresh-process state.
///
/// Call at the top of the `#[test]` fn with a name-unique env var and the
/// test's libtest path; when it returns `true`, the child already ran the
/// real body and the caller must return immediately.
fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
    if std::env::var_os(isolated_env).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("typed_facade::{test_path}"),
            "--include-ignored",
        ])
        .env(isolated_env, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
        "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    true
}

// ---------------------------------------------------------------------------
// External Unique-fusion provider: Z3 charges, addition mod 3.
//
// Z3 rather than Z2/XOR because its non-vacuum charges are not self-dual, so
// every dual-leg assertion in this suite is about a sector that actually
// changes under the dual.
// ---------------------------------------------------------------------------

/// One deliberately broken behaviour, injected into an otherwise valid
/// provider. The rule identity is unaffected, matching how the checked
/// admission tests in `symtn-tensors` inject failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quirk {
    /// Encodes every label to the vacuum id, violating codec injectivity.
    AliasLabels,
    /// Fails the checked dual, i.e. mid-staging inside the checked path.
    FailDual,
    /// Refuses to decode charge 2, which the engine reaches by fusing two
    /// charge-1 legs — a violation of the codec's decode-totality law.
    NarrowDecode,
    /// Duals every sector to the vacuum, i.e. a non-injective dual: a broken
    /// rigidity structure rather than an unrepresentable value.
    CollapsingDual,
    /// Labels the charges in the reverse of the engine's `SectorId` order
    /// (`c <-> 2 - c`). Not broken at all — a valid codec whose label order
    /// simply is not the id order, which is the only way to observe that a
    /// facade sorts by label rather than by id.
    ReversedLabels,
}

/// Z3 as a downstream `fusion_symbols` category: objects are the charges, so
/// `CategoryRule` reaches the typed facade without any symtn provider impl.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExternalZ3;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Z3Charge(u8);

impl FusionCategory for ExternalZ3 {
    type Object = Z3Charge;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = core::iter::Once<Z3Charge>;
    type Error = core::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> Z3Charge {
        Z3Charge(0)
    }
    fn contains(&self, a: &Z3Charge) -> bool {
        a.0 < 3
    }
    fn dual(&self, a: &Z3Charge) -> Z3Charge {
        Z3Charge((3 - a.0) % 3)
    }
    fn dim(&self, _: &Z3Charge) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _: &Z3Charge) -> f64 {
        1.0
    }
    fn frobenius_schur_indicator(&self, a: &Z3Charge) -> f64 {
        if a.0 == 0 {
            1.0
        } else {
            0.0
        }
    }
    fn twist(&self, _: &Z3Charge) -> f64 {
        1.0
    }
    fn fusion<'a>(&'a self, a: &Z3Charge, b: &Z3Charge) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(core::iter::once(Z3Charge((a.0 + b.0) % 3)))
    }
    fn nsymbol(&self, a: &Z3Charge, b: &Z3Charge, c: &Z3Charge) -> u32 {
        u32::from((a.0 + b.0) % 3 == c.0)
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(
        &self,
        a: &Z3Charge,
        b: &Z3Charge,
        c: &Z3Charge,
        d: &Z3Charge,
        e: &Z3Charge,
        f: &Z3Charge,
    ) -> Result<f64, Self::Error> {
        let admissible = self.nsymbol(a, b, e)
            * self.nsymbol(e, c, d)
            * self.nsymbol(b, c, f)
            * self.nsymbol(a, f, d);
        Ok(f64::from(admissible))
    }
    fn r_symbol(&self, a: &Z3Charge, b: &Z3Charge, c: &Z3Charge) -> Result<f64, Self::Error> {
        Ok(f64::from(self.nsymbol(a, b, c)))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"symtn-typed-facade-external-z3"
    }
}

/// The Z3 labels, numbering, identity, and injected faults. A quirk is a
/// broken codec, not a different fusion algebra, so it keeps the identity of
/// its tag: the failure-injection tests rely on that.
#[derive(Clone, Copy, Debug)]
struct Z3Codec {
    quirk: Option<Quirk>,
    /// Distinguishes two otherwise identical provider values, so the facade's
    /// `RuleIdentity` handling can be tested in both directions with one type.
    identity_tag: u8,
}

type ExternalZ3Rule = CategoryRule<ExternalZ3, Z3Codec>;

fn external_z3_rule(quirk: Option<Quirk>, identity_tag: u8) -> ExternalZ3Rule {
    CategoryRule::try_new(
        ExternalZ3,
        Z3Codec {
            quirk,
            identity_tag,
        },
    )
    .expect("Z3 is admitted")
}

fn external_z3() -> ExternalZ3Rule {
    external_z3_rule(None, 0)
}

fn external_z3_with(quirk: Quirk) -> ExternalZ3Rule {
    external_z3_rule(Some(quirk), 0)
}

fn external_z3_tagged(identity_tag: u8) -> ExternalZ3Rule {
    external_z3_rule(None, identity_tag)
}

impl CategoryCodec<ExternalZ3> for Z3Codec {
    type Sector = Z3Charge;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a ExternalZ3 + Copy,
        rule: &RuleIdentity,
        label: &Z3Charge,
    ) -> Result<SectorId, FusionAlgebraError> {
        if self.quirk == Some(Quirk::AliasLabels) {
            return Ok(SectorId::new(0));
        }
        if label.0 < 3 {
            let id = if self.quirk == Some(Quirk::ReversedLabels) {
                2 - label.0
            } else {
                label.0
            };
            Ok(SectorId::new(usize::from(id)))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: format!("Z3 charge {}", label.0),
            })
        }
    }

    fn decode_label<'a>(
        &self,
        _: impl Fn() -> &'a ExternalZ3 + Copy,
        sector: SectorId,
    ) -> Result<Z3Charge, FusionAlgebraError> {
        let limit = if self.quirk == Some(Quirk::NarrowDecode) {
            2
        } else {
            3
        };
        u8::try_from(sector.id())
            .ok()
            .filter(|&charge| charge < limit)
            .map(|charge| {
                Z3Charge(if self.quirk == Some(Quirk::ReversedLabels) {
                    2 - charge
                } else {
                    charge
                })
            })
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }

    fn object<'a>(
        &self,
        _: impl Fn() -> &'a ExternalZ3 + Copy,
        sector: SectorId,
    ) -> Result<Z3Charge, FusionAlgebraError> {
        u8::try_from(sector.id())
            .ok()
            .filter(|&charge| charge < 3)
            .map(Z3Charge)
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }

    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a ExternalZ3 + Copy,
        _: &RuleIdentity,
        object: &Z3Charge,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(usize::from(object.0)))
    }

    fn rule_identity(&self, _: &ExternalZ3) -> Option<RuleIdentity> {
        Some(RuleIdentity::from_canonical_bytes::<ExternalZ3>(
            0x5a33_0000_0000_0000,
            Arc::<[u8]>::from(vec![self.identity_tag]),
        ))
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a ExternalZ3 + Copy,
        _: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[Z3Charge; N], FusionAlgebraError> {
        if operation == CategoryOperation::Dual {
            match self.quirk {
                Some(Quirk::FailDual) => {
                    return Err(FusionAlgebraError::InvalidSector { sector: ids[0] });
                }
                // The dual of every sector is then the vacuum's dual.
                Some(Quirk::CollapsingDual) => return Ok([Z3Charge(0); N]),
                _ => {}
            }
        }
        let mut objects = [Z3Charge(0); N];
        for (object, id) in objects.iter_mut().zip(ids) {
            *object = self.object(category, id)?;
        }
        Ok(objects)
    }

    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<core::convert::Infallible>,
    ) -> FusionAlgebraError {
        match error {
            CategoryAdapterError::ForeignObject {
                sector: Some(sector),
            } => FusionAlgebraError::InvalidSector { sector },
            other => FusionAlgebraError::CategoryFailure(CategoryFailure::new(other)),
        }
    }
}

// ---------------------------------------------------------------------------
// External Simple-fusion provider: a downstream category that delegates its
// SU(2) mathematics to fusion-symbols, with its own compact codec.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExternalSu2(LieGroup);

impl FusionCategory for ExternalSu2 {
    type Object = LieIrrep;
    type Scalar = f64;
    type FSymbol = <LieGroup as FusionCategory>::FSymbol;
    type RSymbol = <LieGroup as FusionCategory>::RSymbol;
    type Fusion<'a> = <LieGroup as FusionCategory>::Fusion<'a>;
    type Error = LieError;

    fn fusion_style(&self) -> FusionStyle {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyle {
        self.0.braiding_style()
    }
    fn unit(&self) -> LieIrrep {
        self.0.unit()
    }
    fn contains(&self, a: &LieIrrep) -> bool {
        self.0.contains(a)
    }
    fn dual(&self, a: &LieIrrep) -> LieIrrep {
        self.0.dual(a)
    }
    fn dim(&self, a: &LieIrrep) -> f64 {
        self.0.dim(a)
    }
    fn frobenius_schur_phase(&self, a: &LieIrrep) -> f64 {
        self.0.frobenius_schur_phase(a)
    }
    fn frobenius_schur_indicator(&self, a: &LieIrrep) -> f64 {
        self.0.frobenius_schur_indicator(a)
    }
    fn twist(&self, a: &LieIrrep) -> f64 {
        self.0.twist(a)
    }
    fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
        self.0.fusion(a, b)
    }
    fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
        self.0.nsymbol(a, b, c)
    }
    fn has_canonical_unit(&self) -> bool {
        self.0.has_canonical_unit()
    }
    fn f_symbol(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        d: &LieIrrep,
        e: &LieIrrep,
        f: &LieIrrep,
    ) -> Result<Self::FSymbol, LieError> {
        self.0.f_symbol(a, b, c, d, e, f)
    }
    fn r_symbol(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
    ) -> Result<Self::RSymbol, LieError> {
        self.0.r_symbol(a, b, c)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        self.0.authority_fingerprint()
    }
}

/// Doubled-spin ids under the built-in `SU2Irrep` labels.
#[derive(Clone, Copy, Debug)]
struct Su2Codec;

type ExternalSu2Rule = CategoryRule<ExternalSu2, Su2Codec>;

fn external_su2() -> ExternalSu2Rule {
    let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    CategoryRule::try_new(ExternalSu2(group), Su2Codec).expect("SU(2) is admitted")
}

impl CategoryCodec<ExternalSu2> for Su2Codec {
    type Sector = SU2Irrep;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a ExternalSu2 + Copy,
        _: &RuleIdentity,
        label: &SU2Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        SU2FusionRule.encode_sector(label)
    }

    fn decode_label<'a>(
        &self,
        _: impl Fn() -> &'a ExternalSu2 + Copy,
        sector: SectorId,
    ) -> Result<SU2Irrep, FusionAlgebraError> {
        SectorCodec::decode_sector(&SU2FusionRule, sector)
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a ExternalSu2 + Copy,
        sector: SectorId,
    ) -> Result<LieIrrep, FusionAlgebraError> {
        i64::try_from(sector.id())
            .ok()
            .and_then(|twice_spin| category().0.irrep(&[twice_spin], &[]).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }

    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a ExternalSu2 + Copy,
        rule: &RuleIdentity,
        object: &LieIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        object
            .dynkin_labels()
            .next()
            .and_then(|twice_spin| usize::try_from(twice_spin).ok())
            .map(SectorId::new)
            .ok_or_else(|| FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: format!("{object:?}"),
            })
    }

    fn rule_identity(&self, _: &ExternalSu2) -> Option<RuleIdentity> {
        Some(RuleIdentity::of_type::<ExternalSu2>())
    }

    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<LieError>,
    ) -> FusionAlgebraError {
        match error {
            CategoryAdapterError::ForeignObject {
                sector: Some(sector),
            } => FusionAlgebraError::InvalidSector { sector },
            other => FusionAlgebraError::CategoryFailure(CategoryFailure::new(other)),
        }
    }
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn z3_leg(provider: &Arc<ExternalZ3Rule>, is_dual: bool) -> GradedSpace<ExternalZ3Rule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [(Z3Charge(0), 2), (Z3Charge(1), 3), (Z3Charge(2), 1)],
    )
    .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
    .expect("Z3 leg is well formed")
}

fn su2_leg(provider: &Arc<ExternalSu2Rule>, is_dual: bool) -> GradedSpace<ExternalSu2Rule> {
    GradedSpace::try_new(Arc::clone(provider), [(SU2Irrep::from_twice_spin(1), 2)])
        .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
        .expect("SU(2) leg is well formed")
}

fn runtime() -> Runtime {
    Runtime::builder().build().expect("runtime builds")
}

// ---------------------------------------------------------------------------
// `GradedSpace<R>`.
// ---------------------------------------------------------------------------

/// The same value computed from the typed labels the facade hands the closure.
fn typed_fill_value(
    sectors: &symtn::typed::BlockFusionTrees<symtn::sector::Z2Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().parity()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 10.0 * (position + 1) as f64;
    }
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
}

// ---------------------------------------------------------------------------
// `TensorMap::permute`.
// ---------------------------------------------------------------------------

/// The second Z3 leg shape, deliberately unlike [`z3_leg`]'s: with four legs of
/// two distinct degeneracy patterns, a permuted leg can be identified by its
/// degeneracies, which a uniform fixture could not distinguish.
fn z3_other_leg(provider: &Arc<ExternalZ3Rule>, is_dual: bool) -> GradedSpace<ExternalZ3Rule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [(Z3Charge(0), 1), (Z3Charge(1), 2), (Z3Charge(2), 4)],
    )
    .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
    .expect("Z3 leg is well formed")
}

/// A rank-4 Z3 tensor map whose elements are all distinct, so any leg, block or
/// element reordering is visible in the buffer. The layout is
/// `[wide, narrow] <- [wide', narrow']`, so no two axes share a shape.
fn z3_rank_four(
    runtime: &Runtime,
    provider: &Arc<ExternalZ3Rule>,
) -> TensorMap<ExternalZ3Rule, f64> {
    let wide = z3_leg(provider, false);
    let narrow = z3_other_leg(provider, false);
    let wide_dual = wide.try_dual().unwrap();
    let narrow_dual = narrow.try_dual().unwrap();
    let mut counter = 0.0;
    TensorMap::from_subblock_fn(
        runtime,
        [&wide, &narrow],
        [&wide_dual, &narrow_dual],
        |_, _| {
            counter += 1.0;
            counter
        },
    )
    .expect("Z3 rank-4 layout is admissible")
}

/// One built-in Z2 layout filled from typed fusion-tree labels.
fn z2_tensor(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    z2_tensor_split(runtime, 2)
}

/// The same tensor over three identical legs, split into
/// `num_codomain <- rest`. `2` is tall in every coupled sector and `1` is wide,
/// which is what separates the compact factorizations from the full ones:
/// on a tall input LQ-compact and LQ-full coincide.
fn z2_tensor_split(
    runtime: &Runtime,
    num_codomain: usize,
) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let legs = [&leg, &leg, &leg];
    let (codomain, domain) = legs.split_at(num_codomain);
    TensorMap::from_subblock_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        typed_fill_value,
    )
    .unwrap()
}

/// The c64 sibling of [`z2_tensor`]. The imaginary part is deliberately
/// not proportional to the real one, so a stray conjugation or a real-only
/// path is visible in every comparison this tensor feeds.
fn z2_complex_tensor(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, Complex64> {
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg], |sectors, indices| {
        complex(typed_fill_value(sectors, indices))
    })
    .unwrap()
}

// ---------------------------------------------------------------------------
// `TensorMap::contract`.
// ---------------------------------------------------------------------------

/// A single-sector Z3 leg: the whole tensor map is then one dense block, so a
/// contraction result can be checked against a hand-computed matrix product.
fn z3_dense_leg(provider: &Arc<ExternalZ3Rule>, degeneracy: usize) -> GradedSpace<ExternalZ3Rule> {
    GradedSpace::try_new(Arc::clone(provider), [(Z3Charge(0), degeneracy)])
        .expect("single-sector Z3 leg is well formed")
}

/// Fills a tensor map's storage with `start, start + 1, ...` in storage order.
fn counting_z3(
    runtime: &Runtime,
    codomain: &GradedSpace<ExternalZ3Rule>,
    domain: &GradedSpace<ExternalZ3Rule>,
    start: f64,
) -> TensorMap<ExternalZ3Rule, f64> {
    let mut next = start - 1.0;
    TensorMap::from_subblock_fn(runtime, [codomain], [domain], |_, _| {
        next += 1.0;
        next
    })
    .expect("single-block Z3 layout is admissible")
}

// TensorKitSectors `anyons.jl` PlanarTrivial: one simple object, unique
// fusion, no braiding, N = F = 1, and the object is the canonical unit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PlanarTrivial;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PlanarTrivialSector;

/// The failure of a braiding query on a planar category.
#[derive(Debug)]
struct NoBraiding;

impl core::fmt::Display for NoBraiding {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("planar category has no braiding")
    }
}

impl std::error::Error for NoBraiding {}

impl FusionCategory for PlanarTrivial {
    type Object = PlanarTrivialSector;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = core::iter::Once<PlanarTrivialSector>;
    type Error = NoBraiding;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::None
    }
    fn unit(&self) -> PlanarTrivialSector {
        PlanarTrivialSector
    }
    fn dual(&self, _: &PlanarTrivialSector) -> PlanarTrivialSector {
        PlanarTrivialSector
    }
    fn dim(&self, _: &PlanarTrivialSector) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _: &PlanarTrivialSector) -> f64 {
        1.0
    }
    fn frobenius_schur_indicator(&self, _: &PlanarTrivialSector) -> f64 {
        1.0
    }
    fn twist(&self, _: &PlanarTrivialSector) -> f64 {
        panic!("PlanarTrivial has no twist")
    }
    fn fusion<'a>(
        &'a self,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
    ) -> Result<Self::Fusion<'a>, NoBraiding> {
        Ok(core::iter::once(PlanarTrivialSector))
    }
    fn nsymbol(
        &self,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
    ) -> u32 {
        1
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(
        &self,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
    ) -> Result<f64, NoBraiding> {
        Ok(1.0)
    }
    fn r_symbol(
        &self,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
        _: &PlanarTrivialSector,
    ) -> Result<f64, NoBraiding> {
        Err(NoBraiding)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"symtn-typed-facade-planar-trivial"
    }
}

#[derive(Clone, Copy, Debug)]
struct PlanarTrivialCodec;

type PlanarTrivialRule = CategoryRule<PlanarTrivial, PlanarTrivialCodec>;

fn planar_trivial() -> PlanarTrivialRule {
    CategoryRule::try_new(PlanarTrivial, PlanarTrivialCodec).expect("PlanarTrivial is admitted")
}

impl CategoryCodec<PlanarTrivial> for PlanarTrivialCodec {
    type Sector = PlanarTrivialSector;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a PlanarTrivial + Copy,
        _: &RuleIdentity,
        _: &PlanarTrivialSector,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(0))
    }

    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a PlanarTrivial + Copy,
        sector: SectorId,
    ) -> Result<PlanarTrivialSector, FusionAlgebraError> {
        self.object(category, sector)
    }

    fn object<'a>(
        &self,
        _: impl Fn() -> &'a PlanarTrivial + Copy,
        sector: SectorId,
    ) -> Result<PlanarTrivialSector, FusionAlgebraError> {
        if sector == SectorId::new(0) {
            Ok(PlanarTrivialSector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }

    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a PlanarTrivial + Copy,
        _: &RuleIdentity,
        _: &PlanarTrivialSector,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(0))
    }

    fn rule_identity(&self, _: &PlanarTrivial) -> Option<RuleIdentity> {
        Some(RuleIdentity::of_type::<PlanarTrivial>())
    }

    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<NoBraiding>,
    ) -> FusionAlgebraError {
        match error {
            CategoryAdapterError::ForeignObject {
                sector: Some(sector),
            } => FusionAlgebraError::InvalidSector { sector },
            other => FusionAlgebraError::CategoryFailure(CategoryFailure::new(other)),
        }
    }
}

// ---------------------------------------------------------------------------
// Non-regression gates.
// ---------------------------------------------------------------------------

#[test]
fn a_failing_typed_operation_publishes_no_cache_state() {
    let _serial = crate::counting_alloc::serial();
    // Keep the absolute process-global snapshot in an explicitly fresh child
    // process, as in the original #1606 regression gate.
    if run_isolated_or_return(
        "SYMTN_TYPED_FACADE_FAILING_OPERATION_PUBLISHES_NO_CACHE_STATE_ISOLATED",
        "a_failing_typed_operation_publishes_no_cache_state",
    ) {
        return;
    }
    // What: an operation rejected by the expert layer leaves both process-global
    // layout caches and the runtime's tree-transform cache exactly as they were,
    // the same transactional guarantee construction gives.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let tensor = z3_rank_four(&runtime, &provider);
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );
    let runtime_before = completed_transformers();

    assert!(tensor.permute(&[0, 0], &[2, 3]).is_err());
    assert!(tensor
        .contract(
            &tensor,
            &ContractSpec {
                lhs: &[3],
                rhs: &[9],
                codomain: &[0, 1, 2],
                domain: &[3, 4, 5]
            }
        )
        .is_err());

    assert_eq!(
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        ),
        before
    );
    assert_eq!(completed_transformers(), runtime_before);
}

// ---------------------------------------------------------------------------
// `TensorMap::transpose`.
// ---------------------------------------------------------------------------

/// The `(is_dual, degeneracies)` shape of a typed tensor map's legs, codomain
/// first — enough to pin where each source leg landed and how it was bent.
fn typed_leg_shapes<R, D>(tensor: &TensorMap<R, D>) -> Vec<(bool, Vec<usize>)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: symtn::typed::TensorScalar,
{
    tensor
        .codomain()
        .iter()
        .chain(tensor.domain().iter())
        .map(|leg| (leg.is_dual(), leg.degeneracies().to_vec()))
        .collect()
}

// ---------------------------------------------------------------------------
// `symtn::typed` self-sufficiency (issue #557).
// ---------------------------------------------------------------------------

/// Deliberately imports nothing but `symtn::typed` names and the provider
/// this suite defines: if `Error` or `Runtime` were missing from the module,
/// this module would not compile. The provider itself must come from
/// somewhere — a typed facade is parameterised by one — which is exactly the
/// "self-sufficient apart from the provider" claim.
mod typed_is_self_sufficient {
    use std::sync::Arc;
    use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

    use super::{external_z3, ExternalZ3Rule, Z3Charge};

    #[test]
    fn typed_imports_run_an_end_to_end_typed_operation() {
        let _serial = crate::counting_alloc::serial();
        let runtime: Runtime = Runtime::builder().build().expect("runtime builds");
        let provider = Arc::new(external_z3());
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 2), (Z3Charge(1), 3)])
            .expect("leg is well formed");

        let build = || -> Result<TensorMap<ExternalZ3Rule, f64>, Error> {
            let mut next = 0.0;
            let tensor = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
                next += 1.0;
                next
            })?;
            full_transpose!(tensor)
        };

        let transposed = build().expect("the typed pipeline runs");
        assert_eq!(transposed.codomain().len(), 1);
        assert_eq!(transposed.domain().len(), 2);

        // The re-exported `Error` is the same type the facade returns, not a
        // lookalike: this only type-checks if the two are one.
        let failure: Error = transposed.repartition(9).unwrap_err();
        assert!(matches!(failure, Error::InvalidArgument(_)));
    }
}

// ---------------------------------------------------------------------------
// Planar is not permute.
//
// The built-in `FermionParityFusionRule` is the only rule reachable from this
// facade whose braiding is not symmetric (`BraidingStyleKind::Fermionic`), so
// it is the only one that can tell a planar bend apart from a braid. It meets
// every typed-facade bound — it is a genuine external-shaped provider here,
// not crate-internal machinery.
// ---------------------------------------------------------------------------

/// A rank-2 fermionic tensor map, `[odd, odd] <- []`. Both legs odd, so every
/// bend crosses a fermion past a fermion and the sign is observable; the empty
/// domain keeps the layout to a single one-element block, so a sign flip is the
/// only thing a comparison can be reporting.
/// A fermionic leg carrying both parities, degeneracy one each.
fn fermionic_leg() -> GradedSpace<symtn::sector::FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .expect("fermionic leg is well formed")
}

/// `[leg, leg] <- [leg]`, counting fill: four elements across two blocks, all
/// distinct, so both the motion and the sign of every element are visible.
fn fermionic_rank_three(
    runtime: &Runtime,
) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let leg = fermionic_leg();
    let mut next = 0.0;
    TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .expect("fermionic layout is admissible")
}

/// The truncated SVD, composed: `svd_compact` -> `diagview` ->
/// `find_truncated` -> `restrict_leg`.
/// Evaluates to `(u, s, vh, error)`.
macro_rules! truncated_svd {
    ($tensor:expr, $truncation:expr) => {{
        let Svd { u, s, vh } = $tensor
            .svd_compact(&codomain_axes(&$tensor), &domain_axes(&$tensor))
            .unwrap();
        let found = s.domain()[0]
            .find_truncated(&s.diagview().unwrap(), &$truncation)
            .unwrap();
        (
            u.restrict_leg(&[(u.codomain_rank(), &found.selection)])
                .unwrap(),
            s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])
                .unwrap(),
            vh.restrict_leg(&[(0, &found.selection)]).unwrap(),
            found.error,
        )
    }};
}

// ---------------------------------------------------------------------------
// `norm`, `norm(Inf)`, `normalize`.
// ---------------------------------------------------------------------------

/// An SU(2) tensor over two legs, split into `num_codomain <- rest`.
///
/// Why this fixture exists at all, next to the Z2 one: SU(2) is the only
/// non-abelian rule here, so it is the only one whose coupled sectors have
/// `dim(c) != 1`. Z2 is abelian and takes `weighted_inner`'s `Unique` fast
/// path, where the quantum-dimension weights are all one and therefore
/// invisible — every dimension-weighted operation needs this fixture as well.
///
fn su2_tensor_split(
    runtime: &Runtime,
    num_codomain: usize,
) -> TensorMap<symtn::sector::SU2FusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let legs = [&leg, &leg];
    let (codomain, domain) = legs.split_at(num_codomain);
    let mut next = 0.0;
    TensorMap::from_subblock_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        |_, _| {
            next += 1.0;
            next
        },
    )
    .unwrap()
}

/// The endomorphism split of [`su2_tensor_split`]: `[v] <- [v]`, which is
/// what `tr` needs and what every other SU(2) assertion here happens to use.
fn su2_tensor(runtime: &Runtime) -> TensorMap<symtn::sector::SU2FusionRule, f64> {
    su2_tensor_split(runtime, 1)
}

// ---------------------------------------------------------------------------
// `inner`, `dot`, `tr`.
// ---------------------------------------------------------------------------

/// A Z2 endomorphism, `[v] <- [v]`: the abelian half of the `tr`
/// comparison, where every quantum dimension is one.
fn z2_endomorphism(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], typed_fill_value).unwrap()
}

// ---------------------------------------------------------------------------
// `TensorMap::trace_pairs`.
// ---------------------------------------------------------------------------

/// A fermionic endomorphism `[v] <- [v]` with distinct sector values.
fn fermionic_endo(runtime: &Runtime) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let leg = fermionic_leg();
    let mut next = 0.0;
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap()
}

// ---------------------------------------------------------------------------
// `map_diagonal`: elementwise maps of a compact diagonal (issue #1558).
// ---------------------------------------------------------------------------

fn spectra<S: Clone, D: Clone>(entries: &[(S, &[D])]) -> Vec<symtn::typed::SectorSpectrum<S, D>> {
    entries
        .iter()
        .map(|(sector, values)| symtn::typed::SectorSpectrum {
            sector: sector.clone(),
            values: values.to_vec(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Issue #585: compact diagonal parity, round 2.
//
// The value oracles below are what pin the compact arms that follow. They are
// written against the typed dense and pointwise routes those arms replace, so
// they are independent of whatever shared helper the compact arms end up
// calling.
// ---------------------------------------------------------------------------

/// A dense twin of a compact bond factor. Adding an exact dense zero keeps the
/// values and space while forcing the mixed compact/dense arm.
fn forced_dense<R, D>(compact: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: symtn::typed::TensorScalar + std::fmt::Debug,
{
    let codomain = compact.codomain();
    let domain = compact.domain();
    let zeros = TensorMap::zeros(compact.runtime(), &codomain, &domain)
        .expect("zero tensor on an admitted bond space is total");
    let dense = compact
        .axpby(D::from_real(1.0), &zeros, D::from_real(1.0))
        .expect("mixed add on one bond space is total");
    assert_eq!(
        dense.dense_data().unwrap(),
        compact.materialize().unwrap().dense_data().unwrap(),
        "the dense twin lost values"
    );
    assert!(
        symtn::expert::diagonal_spectrum(&dense).unwrap().is_none(),
        "the dense twin stayed compact"
    );
    dense
}

fn z2_bond(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed = TensorMap::from_subblock_fn(runtime, [&leg], [&leg], typed_fill_value).unwrap();
    typed.svd_compact(&[0], &[1]).unwrap().s
}

// ---------------------------------------------------------------------------
// Issue #589: semantic laws for U(1), U(1) x fZ2, and
// fZ2 x U(1) x SU(2). The two product rules exercise the packed product codec.
// ---------------------------------------------------------------------------

type U1Fz2Codec = symtn::sector::PackedProductCodec<
    symtn::sector::U1SectorLayout,
    symtn::sector::Fz2SectorLayout,
>;
type Fz2U1Codec = symtn::sector::PackedProductCodec<
    symtn::sector::Fz2SectorLayout,
    symtn::sector::U1SectorLayout,
>;
type Fz2U1Layout = symtn::sector::ProductSectorLayout<
    symtn::sector::Fz2SectorLayout,
    symtn::sector::U1SectorLayout,
>;
type Fz2U1Su2Codec = symtn::sector::PackedProductCodec<Fz2U1Layout, symtn::sector::Su2SectorLayout>;

type U1Fz2Rule = symtn::sector::ProductFusionRule<
    symtn::sector::U1FusionRule,
    symtn::sector::FermionParityFusionRule,
    U1Fz2Codec,
>;
type Fz2U1Rule = symtn::sector::ProductFusionRule<
    symtn::sector::FermionParityFusionRule,
    symtn::sector::U1FusionRule,
    Fz2U1Codec,
>;
type Fz2U1Su2Rule =
    symtn::sector::ProductFusionRule<Fz2U1Rule, symtn::sector::SU2FusionRule, Fz2U1Su2Codec>;

/// `[p, q] <- [p, q]`, filled with a counter starting at `first_value`.
///
/// Why this one geometry for all three families: it is simultaneously a
/// composition (`self`'s domain *is* `other`'s codomain, so `compose` and a
/// two-axis `contract` are both legal on the same pair, which is what pins the
/// fermionic twist), square enough for every factorization, and rank 4, so a
/// nonidentity output order has somewhere to move a leg.
///
/// Why a plain counter rather than a label-derived fill: every element is then
/// distinct, so any reordering of blocks or of elements within a block moves
/// the buffer. `first_value` makes the second operand carry different values;
/// `contract` against a copy of `self` could otherwise hide a swap.
///
/// Why `p != q` in every family: with two distinct legs, permuting them is
/// visible in the leg shapes, so a nonidentity output order cannot be undone
/// by coincidence. One of the two is dual, because a dual leg is the only
/// place a fermionic twist can act.
fn counter_oracle<R>(
    runtime: &Runtime,
    typed_legs: (&GradedSpace<R>, &GradedSpace<R>),
    first_value: f64,
) -> TensorMap<R, f64>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let mut next = first_value - 1.0;
    let typed = TensorMap::from_subblock_fn(
        runtime,
        [typed_legs.0, typed_legs.1],
        [typed_legs.0, typed_legs.1],
        |_, _| {
            next += 1.0;
            next
        },
    )
    .unwrap();
    assert!(!typed.dense_data().unwrap().is_empty());
    typed
}

/// The fZ2 x U(1) x SU(2) oracle: the only route that exercises the
/// fermionic twist and the quantum-dimension weights at once.
///
/// Every one of the three factors is deliberately nonconstant across the two
/// legs: both parities appear (so the twist has somewhere to act), the charges
/// are `-1, 0, 1, 2` (so the U(1) balance is not automatic), and the spins are
/// `0, 1/2, 1` (so `dim(c)` takes the values 1, 2 and 3 and a weight-free
/// `inner` cannot pass). A fixture with spin 0 everywhere would not exercise
/// the weighted laws below.
fn fz2_u1_su2_oracle(runtime: &Runtime, first_value: f64) -> TensorMap<Fz2U1Su2Rule, f64> {
    let (typed_p, typed_q) = fz2_u1_su2_typed_legs();
    counter_oracle(runtime, (&typed_p, &typed_q), first_value)
}

/// The legs shared with the twist identity test below.
fn fz2_u1_su2_typed_legs() -> (GradedSpace<Fz2U1Su2Rule>, GradedSpace<Fz2U1Su2Rule>) {
    let rule = Arc::new(
        Fz2U1Su2Rule::try_new(
            Fz2U1Rule::try_new(
                symtn::sector::FermionParityFusionRule,
                symtn::sector::U1FusionRule,
            )
            .unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let label = |parity: u8, charge: i32, twice_spin: usize| {
        symtn::sector::ProductSector::new(
            symtn::sector::ProductSector::new(
                if parity == 0 {
                    symtn::sector::Z2Irrep::EVEN
                } else {
                    symtn::sector::Z2Irrep::ODD
                },
                symtn::sector::U1Irrep::new(charge),
            ),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    let typed_p = GradedSpace::try_new(
        Arc::clone(&rule),
        [(label(0, 0, 0), 1), (label(1, 1, 1), 2)],
    )
    .unwrap();
    let typed_q = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (label(1, -1, 1), 1),
            (label(0, 0, 2), 1),
            (label(0, 2, 0), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    (typed_p, typed_q)
}

// ---------------------------------------------------------------------------
// Typed constructors — `rand_with_seed` and the structural
// family (`isomorphism`/`isometry`).
// ---------------------------------------------------------------------------

fn u1_typed_leg() -> GradedSpace<symtn::sector::U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(symtn::sector::U1FusionRule),
        [
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
            (symtn::sector::U1Irrep::new(-1), 3),
        ],
    )
    .unwrap()
}

fn fz2_typed_leg() -> GradedSpace<symtn::sector::FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Typed `left_polar` / `right_polar`.
// ---------------------------------------------------------------------------

/// Elementwise closeness at factorization tolerance, relative to the wanted
/// entry's magnitude (floored at 1).
fn assert_data_close_f64(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "{g} vs {w}");
    }
}

/// The c64 sibling of [`assert_data_close_f64`].
fn assert_data_close_c64(got: &[Complex64], want: &[Complex64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).norm() <= 1e-12 * w.norm().max(1.0), "{g} vs {w}");
    }
}

/// Content-wise leg equality: `GradedSpace` has no `PartialEq` (deliberate,
/// see its rustdoc), so space equality is asserted through the public
/// accessors — labels, degeneracies and duality all have to agree.
fn assert_same_legs<R>(got: &[GradedSpace<R>], want: &[GradedSpace<R>])
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + SectorCodec
        + CheckedFusionAlgebra,
    <R as SectorCodec>::Sector: std::fmt::Debug + PartialEq,
{
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.sectors().unwrap(), w.sectors().unwrap());
        assert_eq!(g.degeneracies(), w.degeneracies());
        assert_eq!(g.is_dual(), w.is_dual());
    }
}

fn u1_leg(
    provider: &Arc<symtn::sector::U1FusionRule>,
    pairs: &[(i32, usize)],
) -> GradedSpace<symtn::sector::U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (symtn::sector::U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// NoBraiding preflight for twist and flip.
//
// External NoBraiding provider: planar Z2 (the symtn-core test fixture
// `PlanarZ2Rule`, rebuilt from the public vocabulary with a codec). The
// These gates verify that planar providers need no braiding capability;
// twist/flip still route through the shared structural preflight.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PlanarZ2;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PlanarParity(u8);

impl FusionCategory for PlanarZ2 {
    type Object = PlanarParity;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = core::iter::Once<PlanarParity>;
    type Error = NoBraiding;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::None
    }
    fn unit(&self) -> PlanarParity {
        PlanarParity(0)
    }
    fn contains(&self, a: &PlanarParity) -> bool {
        a.0 < 2
    }
    fn dual(&self, a: &PlanarParity) -> PlanarParity {
        *a
    }
    fn dim(&self, _: &PlanarParity) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _: &PlanarParity) -> f64 {
        1.0
    }
    fn frobenius_schur_indicator(&self, _: &PlanarParity) -> f64 {
        1.0
    }
    fn twist(&self, _: &PlanarParity) -> f64 {
        panic!("PlanarZ2 has no twist")
    }
    fn fusion<'a>(
        &'a self,
        a: &PlanarParity,
        b: &PlanarParity,
    ) -> Result<Self::Fusion<'a>, NoBraiding> {
        Ok(core::iter::once(PlanarParity(a.0 ^ b.0)))
    }
    fn nsymbol(&self, a: &PlanarParity, b: &PlanarParity, c: &PlanarParity) -> u32 {
        u32::from(a.0 ^ b.0 == c.0)
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(
        &self,
        a: &PlanarParity,
        b: &PlanarParity,
        c: &PlanarParity,
        d: &PlanarParity,
        e: &PlanarParity,
        f: &PlanarParity,
    ) -> Result<f64, NoBraiding> {
        let admissible = self.nsymbol(a, b, e)
            * self.nsymbol(e, c, d)
            * self.nsymbol(b, c, f)
            * self.nsymbol(a, f, d);
        Ok(f64::from(admissible))
    }
    fn r_symbol(
        &self,
        _: &PlanarParity,
        _: &PlanarParity,
        _: &PlanarParity,
    ) -> Result<f64, NoBraiding> {
        Err(NoBraiding)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"symtn-typed-facade-planar-z2"
    }
}

#[derive(Clone, Copy, Debug)]
struct PlanarZ2Codec;

type PlanarZ2Rule = CategoryRule<PlanarZ2, PlanarZ2Codec>;

fn planar_z2() -> PlanarZ2Rule {
    CategoryRule::try_new(PlanarZ2, PlanarZ2Codec).expect("planar Z2 is admitted")
}

impl CategoryCodec<PlanarZ2> for PlanarZ2Codec {
    type Sector = PlanarParity;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a PlanarZ2 + Copy,
        rule: &RuleIdentity,
        label: &PlanarParity,
    ) -> Result<SectorId, FusionAlgebraError> {
        if label.0 < 2 {
            Ok(SectorId::new(usize::from(label.0)))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: format!("planar parity {}", label.0),
            })
        }
    }

    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a PlanarZ2 + Copy,
        sector: SectorId,
    ) -> Result<PlanarParity, FusionAlgebraError> {
        self.object(category, sector)
    }

    fn object<'a>(
        &self,
        _: impl Fn() -> &'a PlanarZ2 + Copy,
        sector: SectorId,
    ) -> Result<PlanarParity, FusionAlgebraError> {
        if sector.id() < 2 {
            Ok(PlanarParity(sector.id() as u8))
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }

    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a PlanarZ2 + Copy,
        _: &RuleIdentity,
        object: &PlanarParity,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(usize::from(object.0)))
    }

    fn rule_identity(&self, _: &PlanarZ2) -> Option<RuleIdentity> {
        Some(RuleIdentity::from_canonical_bytes::<PlanarZ2>(
            0x9a2f_0620_0000_0000,
            Arc::<[u8]>::from(vec![]),
        ))
    }

    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<NoBraiding>,
    ) -> FusionAlgebraError {
        match error {
            CategoryAdapterError::ForeignObject {
                sector: Some(sector),
            } => FusionAlgebraError::InvalidSector { sector },
            other => FusionAlgebraError::CategoryFailure(CategoryFailure::new(other)),
        }
    }
}

// ---------------------------------------------------------------------------
// Non-symmetric braiding boundary of the destination and compact entries
// (#1355, #1372).
// ---------------------------------------------------------------------------

use crate::braiding_probe;
use braiding_probe::{ProbeSector, RealBraidingProbe};

mod compose;
mod construction;
mod contract;
mod diagonal;
mod factorization;
mod laws;
mod matrix_fn;
mod scalar;
mod space;
mod transform;
mod twist_flip;

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
