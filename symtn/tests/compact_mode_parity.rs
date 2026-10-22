//! A compact diagonal (TensorKit `DiagonalTensorMap`) answers every operation
//! the same way in both admission modes (#1866): the same storage form, the
//! same values, and the dense route's answer once materialized.
//!
//! Twist. TensorKit `cfaa073e` `twist(t, inds; inv)`
//! (`src/tensors/indexmanipulations.jl:90`) allocates `similar(t, …)`, which
//! for a `DiagonalTensorMap` is `similar_diagonal` (`src/tensors/diagonal.jl:85`),
//! so the result stays diagonal; `twist!` (`indexmanipulations.jl:62`) scales
//! each block by `Π_{i ∈ inds} θ(f.uncoupled[i])`, conjugated for `inv`. On a
//! bond every uncoupled sector is the coupled `c`, so the hand oracle below is
//! `s_c[i] · θ(c)^{|inds|}` for the real θ of these rules. QSpace has no
//! twist.
//!
//! Compose and contract. TensorKit `cfaa073e` `compose`
//! (`src/tensors/linalg.jl:38`) with `compose_dest(::Diagonal, ::Diagonal)`
//! (`src/tensors/diagonal.jl:325`) keeps `D * D` diagonal, and `t * D` /
//! `D * t` dispatch `mul!` on `block(d, c)::Diagonal` (`diagonal.jl:153`,
//! `330`), i.e. one axis scaled per block. The oracle is the same operation on
//! the materialized operands (the dense route). QSpace has no compact
//! diagonal contraction (`contractQS.cc` expands a diagonal first).
//!
//! Full trace. TensorKit `cfaa073e` has no `DiagonalTensorMap`
//! trace method: the generic `tensortrace!` runs, so the compact result must
//! equal the dense route's. Both modes read the compiled trace coefficients
//! the dense route replays and sum each spectrum block, so the oracle is the
//! dense route (within 1e-12, since the widened accumulation differs) plus
//! the hand supertrace `Σ_c dim(c) θ(c) Σ_i s_c[i]` over a non-dual traced
//! codomain leg. QSpace has no compact trace (`traceQS.cc` has no diagonal
//! format).
//!
//! Capability absences (compile-time, not counted as passing cells): checked
//! × CUDA and Fibonacci × CUDA (CUDA ops are bounded
//! `MultiplicityFreeRigidSymbols<Scalar = f64>`), and Fibonacci twist (the
//! multiplicity-free twist needs `Scalar = f64`). Fibonacci contract is
//! anyonic and asserted as the unchanged `contract` rejection.

#[path = "support/report.rs"]
mod report;
use report::{cause, report};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FermionParityFusionRule, FusionStyleKind, GenericFArray,
    GenericRMatrix, RuleIdentity, SectorId, SectorVec, TypedSectorAdmission, Z2Irrep,
};
use symtn::typed::{
    BlockFusionTrees, Complex64, ContractSpec, Direction, GradedSpace, Runtime, SectorSpectrum,
    TensorMap, TensorScalar, TypedTensorConstructionDispatch, TypedTensorContractDispatch,
    TypedTensorModeDispatch, TypedTensorRootDispatch, TypedTensorTransformDispatch,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::host_runtime;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

const EVEN: SectorId = SectorId::new(0);
const ODD: SectorId = SectorId::new(1);

/// A checked Generic twin of fZ2 (`Z2` fusion, real symbols) whose braiding
/// and odd-sector twist are parameters: fermionic with `θ(odd) = -1` is the
/// checked counterpart of [`FermionParityFusionRule`]; anyonic with a
/// non-unit `θ(odd)` makes the twist a non-sign rescaling.
struct CheckedZ2 {
    braiding: BraidingStyleKind,
    odd_twist: f64,
    /// `R^{odd, odd}` of a non-Bosonic style; `-1` is fZ2's exchange sign.
    odd_r: f64,
    /// Separates otherwise equal rules in the structure caches, so a cold
    /// provider ledger can be read twice.
    salt: u8,
    twist_queries: AtomicUsize,
    /// Every fallible provider query.
    queries: AtomicUsize,
    /// Once set, every fallible provider query fails: an operation that
    /// still succeeds made none.
    fail_queries: AtomicBool,
}

impl CheckedZ2 {
    fn new(braiding: BraidingStyleKind, odd_twist: f64) -> Self {
        Self {
            braiding,
            odd_twist,
            odd_r: -1.0,
            salt: 0,
            twist_queries: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            fail_queries: AtomicBool::new(false),
        }
    }

    fn with_odd_r(self, odd_r: f64) -> Self {
        Self { odd_r, ..self }
    }

    fn with_salt(self, salt: u8) -> Self {
        Self { salt, ..self }
    }

    fn gate(&self) -> Result<(), InvalidSector> {
        self.queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_queries.load(Ordering::Relaxed) {
            Err(InvalidSector)
        } else {
            Ok(())
        }
    }

    fn admissible(left: SectorId, right: SectorId, coupled: SectorId) -> bool {
        (left.id() ^ right.id()) == coupled.id()
    }
}

#[derive(Debug)]
struct InvalidSector;

impl std::fmt::Display for InvalidSector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid Z2 sector")
    }
}

impl std::error::Error for InvalidSector {}

fn checked_z2(sector: SectorId) -> Result<SectorId, InvalidSector> {
    if sector == EVEN || sector == ODD {
        Ok(sector)
    } else {
        Err(InvalidSector)
    }
}

impl symtn_core::testing::Provider for CheckedZ2 {}

impl CheckedGenericFusion for CheckedZ2 {
    type Error = InvalidSector;

    fn rule_identity(&self) -> RuleIdentity {
        let mut bytes = vec![self.braiding as u8];
        bytes.extend(self.odd_twist.to_bits().to_le_bytes());
        bytes.extend(self.odd_r.to_bits().to_le_bytes());
        bytes.push(self.salt);
        RuleIdentity::from_canonical_bytes::<Self>(0x1866, Arc::<[u8]>::from(bytes))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.braiding
    }

    fn vacuum(&self) -> SectorId {
        EVEN
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.gate()?;
        checked_z2(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.gate()?;
        let coupled = SectorId::new(checked_z2(left)?.id() ^ checked_z2(right)?.id());
        Ok([coupled].into_iter().collect())
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.gate()?;
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.gate()?;
        Ok(usize::from(Self::admissible(
            checked_z2(left)?,
            checked_z2(right)?,
            checked_z2(coupled)?,
        )))
    }
}

impl CheckedGenericRigidSymbols for CheckedZ2 {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.gate()?;
        let n = |l, r, c| usize::from(Self::admissible(l, r, c));
        let shape = (n(a, b, e), n(e, c, d), n(b, c, f), n(a, f, d));
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        Ok(GenericFArray::new(vec![1.0; len], shape))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.gate()?;
        let size = usize::from(Self::admissible(a, b, c));
        // Bosonic Z2 is the trivially braided twin; every other style keeps
        // the super-vector-space exchange sign.
        let odd_pair = a == ODD && b == ODD && self.braiding != BraidingStyleKind::Bosonic;
        let value = if odd_pair { self.odd_r } else { 1.0 };
        Ok(GenericRMatrix::new(vec![value; size], size, size))
    }
}

impl CheckedGenericPivotal for CheckedZ2 {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
        self.twist_queries.fetch_add(1, Ordering::Relaxed);
        Ok(if checked_z2(sector)? == ODD {
            self.odd_twist
        } else {
            1.0
        })
    }
}

impl TypedSectorAdmission for CheckedZ2 {
    type Sector = Z2Irrep;
    type Error = InvalidSector;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Z2Irrep) -> Result<SectorId, Self::Error> {
        Ok(SectorId::new(usize::from(sector.parity())))
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Z2Irrep, Self::Error> {
        Ok(Z2Irrep::new(checked_z2(sector)?.id() as u8))
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        checked_z2(sector)
    }
}

/// `(parity, k)`: both sectors, nontrivial and distinct degeneracies.
const SECTORS: [(u8, usize); 2] = [(0, 2), (1, 3)];

/// The stored diagonal; binary fractions, so a ±1 twist is exact.
fn value(parity: u8, i: usize) -> f64 {
    [[0.75, -1.5, 0.0], [0.5, 2.25, -0.125]][usize::from(parity)][i]
}

const LEG_SETS: [&[usize]; 3] = [&[0], &[1], &[0, 1]];

fn close(actual: Complex64, expected: Complex64) -> bool {
    (actual - expected).norm() <= 1e-12 * expected.norm().max(1.0)
}

/// Twists one compact fZ2-shaped diagonal on every leg set in both
/// directions, asserting against the hand oracle and the dense route, and
/// returns the twisted spectra in a fixed order for the cross-mode check.
macro_rules! twist_rows {
    ($rule:expr, $odd_twist:expr, $D:ty, $conv:expr) => {{
        let runtime = host_runtime();
        let conv = $conv;
        let odd_twist: f64 = $odd_twist;
        let leg = GradedSpace::try_new(
            Arc::new($rule),
            SECTORS.iter().map(|&(p, k)| (Z2Irrep::new(p), k)),
        )
        .unwrap();
        let spectrum = SECTORS
            .iter()
            .map(|&(p, k)| SectorSpectrum {
                sector: Z2Irrep::new(p),
                values: (0..k).map(|i| conv(value(p, i))).collect::<Vec<$D>>(),
            })
            .collect::<Vec<_>>();
        let compact = TensorMap::<_, $D>::diagonal(&runtime, &leg, spectrum).unwrap();
        let dense_twin = compact.materialize().unwrap();
        let mut out = Vec::new();
        for legs in LEG_SETS {
            for direction in [Direction::Forward, Direction::Inverse] {
                let twisted = compact.twist(legs, direction).unwrap();
                assert!(
                    twisted.dense_data().is_err(),
                    "twist {legs:?} {direction:?} densified a compact diagonal"
                );
                // TensorKit `twist!`: `inv && (θ = θ')`; a real θ is its own
                // conjugate.
                let power = legs.len() as i32;
                let got = twisted.diagview().unwrap();
                assert_eq!(got.len(), SECTORS.len());
                for (entry, &(p, k)) in got.iter().zip(&SECTORS) {
                    assert_eq!(entry.sector, Z2Irrep::new(p));
                    let theta = if p == 1 { odd_twist } else { 1.0 };
                    let factor = theta.powi(power);
                    for i in 0..k {
                        let expected = Complex64::from(conv(value(p, i))) * factor;
                        assert!(
                            close(Complex64::from(entry.values[i]), expected),
                            "twist {legs:?} {direction:?} sector {p} entry {i}"
                        );
                    }
                }
                let dense = dense_twin.twist(legs, direction).unwrap();
                let materialized = twisted.materialize().unwrap();
                let (lhs, rhs) = (
                    materialized.dense_data().unwrap(),
                    dense.dense_data().unwrap(),
                );
                assert_eq!(lhs.len(), rhs.len());
                for (&a, &b) in lhs.iter().zip(rhs) {
                    assert!(close(Complex64::from(a), Complex64::from(b)));
                }
                out.push(got);
            }
        }
        out
    }};
}

#[test]
fn compact_twist_stays_compact_and_agrees_across_modes() {
    // What: a fermionic compact twist is one mode-free arm: both modes,
    // including the checked Generic fZ2 twin, keep the spectrum and give
    // identical values.
    let real = |x: f64| x;
    let mf = twist_rows!(FermionParityFusionRule, -1.0, f64, real);
    let checked = twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
        -1.0,
        f64,
        real
    );
    assert_eq!(mf, checked);

    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    let mf = twist_rows!(FermionParityFusionRule, -1.0, Complex64, complex);
    let checked = twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
        -1.0,
        Complex64,
        complex
    );
    assert_eq!(mf, checked);
}

#[test]
fn checked_anyonic_compact_twist_scales_by_theta_powers() {
    // What: a non-unit real θ, which only the checked mode can express here,
    // scales the spectrum by θ^{|legs|}, so a two-leg twist is not the
    // identity on the bond.
    let real = |x: f64| x;
    twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6),
        0.6,
        f64,
        real
    );
    let complex = |x: f64| Complex64::new(-0.5 * x, x);
    twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6),
        0.6,
        Complex64,
        complex
    );
}

#[test]
fn checked_compact_twist_queries_the_provider_like_the_dense_route() {
    // What: the compact arm reuses the staged twist values; it adds no
    // provider query beyond the dense route's.
    let runtime = host_runtime();
    let rule = Arc::new(CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6));
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        SECTORS.iter().map(|&(p, k)| (Z2Irrep::new(p), k)),
    )
    .unwrap();
    let spectrum = SECTORS
        .iter()
        .map(|&(p, k)| SectorSpectrum {
            sector: Z2Irrep::new(p),
            values: (0..k).map(|i| value(p, i)).collect::<Vec<f64>>(),
        })
        .collect::<Vec<_>>();
    let compact = TensorMap::<_, f64>::diagonal(&runtime, &leg, spectrum).unwrap();
    let dense = compact.materialize().unwrap();
    let queries = |tensor: &TensorMap<CheckedZ2, f64>| {
        rule.twist_queries.store(0, Ordering::Relaxed);
        black_box(tensor.twist(&[0, 1], Direction::Forward).unwrap());
        rule.twist_queries.load(Ordering::Relaxed)
    };
    let (compact_queries, dense_queries) = (queries(&compact), queries(&dense));
    assert!(dense_queries > 0);
    assert_eq!(compact_queries, dense_queries);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn bosonic_su2_compact_twist_is_the_identity_in_both_modes() {
    // What: SU(2) is bosonic, so twist returns the compact receiver as is in
    // both modes.
    use symtn::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
    let runtime = host_runtime();
    let su2 = [(0usize, 2usize), (1, 3), (2, 1)];
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        su2.iter()
            .map(|&(twice, k)| (SU2Irrep::from_twice_spin(twice), k)),
    )
    .unwrap();
    let checked_leg = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        su2.iter().map(|&(twice, k)| (vec![twice as i64], k)),
    )
    .unwrap();
    let values = |twice: usize, k: usize| (0..k).map(|i| 0.5 + (twice * 4 + i) as f64).collect();
    let mf = TensorMap::<_, f64>::diagonal(
        &runtime,
        &mf_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(twice),
            values: values(twice, k),
        }),
    )
    .unwrap();
    let checked = TensorMap::<_, f64>::diagonal(
        &runtime,
        &checked_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: vec![twice as i64],
            values: values(twice, k),
        }),
    )
    .unwrap();
    for legs in LEG_SETS {
        let mf_twisted = mf.twist(legs, Direction::Forward).unwrap();
        let checked_twisted = checked.twist(legs, Direction::Forward).unwrap();
        assert!(mf_twisted.dense_data().is_err());
        assert!(checked_twisted.dense_data().is_err());
        let mf_values: Vec<_> = mf_twisted
            .diagview()
            .unwrap()
            .into_iter()
            .map(|e| e.values)
            .collect();
        let checked_values: Vec<_> = checked_twisted
            .diagview()
            .unwrap()
            .into_iter()
            .map(|e| e.values)
            .collect();
        assert_eq!(mf_values, checked_values);
        assert_eq!(
            mf_values,
            su2.iter()
                .map(|&(twice, k)| values(twice, k))
                .collect::<Vec<Vec<f64>>>()
        );
    }
}

/// k = 128 makes one dense payload (`k² · 8` bytes) dominate the warm
/// staging bytes, as in `typed_diagonal_allocations.rs`.
const K: usize = 128;

fn odd_diagonal<R>(rule: R, k: usize) -> TensorMap<R, f64>
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: symtn::typed::TypedTensorConstructionDispatch<R, f64>,
{
    let runtime = host_runtime();
    let leg = GradedSpace::try_new(Arc::new(rule), [(Z2Irrep::ODD, k)]).unwrap();
    TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Z2Irrep::ODD,
            values: (0..k).map(|i| 1.0 + i as f64).collect(),
        }],
    )
    .unwrap()
}

fn warmed_twist_bytes<R>(tensor: &TensorMap<R, f64>) -> u64
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorTwistDispatch<R, f64>,
{
    black_box(tensor.twist(&[0], Direction::Forward).unwrap());
    let (twisted, allocs) =
        counting_alloc::measure(|| black_box(tensor.twist(&[0], Direction::Forward).unwrap()));
    assert!(twisted.dense_data().is_err(), "the twist densified");
    allocs.bytes
}

#[test]
fn compact_twist_allocates_linear_in_the_spectrum_in_both_modes() {
    // What: the twist of a compact diagonal costs Σk, never a k² payload: from
    // k to 2k the bytes grow by at most the spectrum's growth, and stay below
    // one dense payload.
    let _measurement = counting_alloc::serial();
    let dense_payload = (K * K * std::mem::size_of::<f64>()) as u64;
    let spectrum_growth = (2 * K * std::mem::size_of::<f64>()) as u64;
    let rows = [
        (
            "multiplicity-free",
            warmed_twist_bytes(&odd_diagonal(FermionParityFusionRule, K)),
            warmed_twist_bytes(&odd_diagonal(FermionParityFusionRule, 2 * K)),
        ),
        (
            "checked",
            warmed_twist_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                K,
            )),
            warmed_twist_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                2 * K,
            )),
        ),
    ];
    for (mode, small, large) in rows {
        assert!(
            large.saturating_sub(small) <= spectrum_growth,
            "{mode} twist is not linear in k: {small} -> {large} bytes"
        );
        assert!(
            large < dense_payload,
            "{mode} twist allocated a dense payload: {large} bytes"
        );
    }
}

// ---- Compose and contract ----

/// The compact operand and its dense twin, plus two dense operands that face
/// the bond: `a: W ⊗ W ← W` and `b: W ← W ⊗ W`.
struct BondFixture<R: TypedSectorAdmission, D: TensorScalar> {
    d: TensorMap<R, D>,
    e: TensorMap<R, D>,
    a: TensorMap<R, D>,
    b: TensorMap<R, D>,
}

fn bond_fixture<R, D>(
    leg: &GradedSpace<R>,
    label: impl Fn(&R::Sector) -> f64,
    conv: impl Fn(f64) -> D,
) -> BondFixture<R, D>
where
    R: TypedSectorAdmission,
    R::Sector: Clone,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    bond_fixture_on(&host_runtime(), leg, leg, label, conv)
}

/// [`bond_fixture`] on `runtime`, with `open` as the dense operands' leg that
/// does not face the bond: `a: W ⊗ open ← W` and `b: W ← open ⊗ W`.
fn bond_fixture_on<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    open: &GradedSpace<R>,
    label: impl Fn(&R::Sector) -> f64,
    conv: impl Fn(f64) -> D,
) -> BondFixture<R, D>
where
    R: TypedSectorAdmission,
    R::Sector: Clone,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    let spectrum = |shift: f64| {
        leg.sectors()
            .unwrap()
            .into_iter()
            .map(|sector| SectorSpectrum {
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(shift + 0.5 * label(&sector) - 0.75 * i as f64))
                    .collect(),
                sector,
            })
            .collect::<Vec<_>>()
    };
    let fill = |trees: &BlockFusionTrees<R::Sector>, index: &[usize]| {
        let sectors: f64 = trees
            .codomain_uncoupled()
            .iter()
            .chain(trees.domain_uncoupled())
            .enumerate()
            .map(|(position, sector)| (position + 1) as f64 * label(sector))
            .sum();
        let offsets: f64 = index
            .iter()
            .enumerate()
            .map(|(axis, &i)| (axis as f64 + 0.5) * i as f64)
            .sum();
        conv(0.25 + 0.375 * sectors - 0.3125 * offsets)
    };
    BondFixture {
        d: TensorMap::diagonal(runtime, leg, spectrum(1.5)).unwrap(),
        e: TensorMap::diagonal(runtime, leg, spectrum(-0.5)).unwrap(),
        a: TensorMap::from_subblock_fn(runtime, [leg, open], [leg], fill).unwrap(),
        b: TensorMap::from_subblock_fn(runtime, [leg], [open, leg], fill).unwrap(),
    }
}

const fn spec<'a>(
    lhs: &'a [usize],
    rhs: &'a [usize],
    codomain: &'a [usize],
    domain: &'a [usize],
) -> ContractSpec<'a> {
    ContractSpec {
        lhs,
        rhs,
        codomain,
        domain,
    }
}

/// `(what, stays compact, lhs, rhs, spec)`; `None` spec is `compose`. Operand
/// codes: `d`/`e` compact, `a`/`b` dense. Every one-leg contraction is listed
/// in its default output order and in at least one reordered one.
type BondCase = (
    &'static str,
    bool,
    char,
    char,
    Option<ContractSpec<'static>>,
);

const COMPOSE_CASES: [BondCase; 3] = [
    ("compose D * D", true, 'd', 'e', None),
    ("compose t * D", false, 'a', 'd', None),
    ("compose D * t", false, 'd', 'b', None),
];

const CONTRACT_CASES: [BondCase; 10] = [
    (
        "contract D . D",
        true,
        'd',
        'e',
        Some(spec(&[1], &[0], &[0], &[1])),
    ),
    (
        "contract D . D bent",
        false,
        'd',
        'e',
        Some(spec(&[1], &[0], &[0, 1], &[])),
    ),
    (
        "contract t . D",
        false,
        'a',
        'd',
        Some(spec(&[2], &[0], &[0, 1], &[2])),
    ),
    (
        "contract t . D swapped",
        false,
        'a',
        'd',
        Some(spec(&[2], &[0], &[1, 0], &[2])),
    ),
    (
        "contract t . D bent",
        false,
        'a',
        'd',
        Some(spec(&[2], &[0], &[0], &[1, 2])),
    ),
    (
        "contract D . t",
        false,
        'd',
        'b',
        Some(spec(&[1], &[0], &[0], &[1, 2])),
    ),
    (
        "contract D . t swapped",
        false,
        'd',
        'b',
        Some(spec(&[1], &[0], &[0], &[2, 1])),
    ),
    (
        "contract D . t bent",
        false,
        'd',
        'b',
        Some(spec(&[1], &[0], &[0, 1], &[2])),
    ),
    // Reordered `D . D` is the product spectrum and one permute (#1870):
    // a swap stays compact (TensorKit `diagonal.jl:291`), a bend is dense.
    (
        "contract D . D swapped",
        true,
        'd',
        'e',
        Some(spec(&[1], &[0], &[1], &[0])),
    ),
    (
        "contract D . D bent to the domain",
        false,
        'd',
        'e',
        Some(spec(&[1], &[0], &[], &[1, 0])),
    ),
];

fn pick<R: TypedSectorAdmission, D: TensorScalar>(
    fixture: &BondFixture<R, D>,
    code: char,
) -> &TensorMap<R, D> {
    match code {
        'd' => &fixture.d,
        'e' => &fixture.e,
        'a' => &fixture.a,
        _ => &fixture.b,
    }
}

/// Runs `cases` on the compact operands and on their materialized twins,
/// asserting the storage form and agreement with the dense route, and
/// returns each result's dense image for the cross-mode comparison. `None`
/// marks a case both storages reject with the same error.
///
/// The one storage-dependent outcome: checked Generic contraction of a
/// non-Bosonic rule is unsupported on the dense route, while the compact
/// arm, which only scales and permutes, answers it as TensorKit does. The
/// caller checks those values against the multiplicity-free mode instead.
fn bond_rows<R, D>(fixture: &BondFixture<R, D>, cases: &[BondCase]) -> Vec<Option<Vec<Complex64>>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D> + TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar + std::fmt::Debug,
    Complex64: From<D>,
{
    let dense = BondFixture {
        d: fixture.d.materialize().unwrap(),
        e: fixture.e.materialize().unwrap(),
        a: fixture.a.clone(),
        b: fixture.b.clone(),
    };
    let run = |operands: &BondFixture<R, D>, lhs, rhs, spec: &Option<ContractSpec<'_>>| {
        let (lhs, rhs) = (pick(operands, lhs), pick(operands, rhs));
        match spec {
            None => lhs.compose(rhs),
            Some(spec) => lhs.contract(rhs, spec),
        }
    };
    let image = |tensor: &TensorMap<R, D>| -> Vec<Complex64> {
        let dense = tensor.materialize().unwrap();
        dense
            .dense_data()
            .unwrap()
            .iter()
            .map(|&x| Complex64::from(x))
            .collect()
    };
    cases
        .iter()
        .map(|(what, compact, lhs, rhs, spec)| {
            match (
                run(fixture, *lhs, *rhs, spec),
                run(&dense, *lhs, *rhs, spec),
            ) {
                (Ok(got), Ok(want)) => {
                    assert_eq!(got.dense_data().is_err(), *compact, "{what}: storage form");
                    assert_eq!(got.codomain(), want.codomain(), "{what}: codomain");
                    assert_eq!(got.domain(), want.domain(), "{what}: domain");
                    let (got, want) = (image(&got), image(&want));
                    assert_eq!(got.len(), want.len(), "{what}");
                    for (&x, &y) in got.iter().zip(&want) {
                        assert!(close(x, y), "{what}: {x} != dense {y}");
                    }
                    Some(got)
                }
                (Ok(got), Err(rejection)) => {
                    assert!(
                        matches!(
                            cause::<symtn::typed::OperationError>(&rejection),
                            Some(symtn::typed::OperationError::UnsupportedTensorContractScope {
                                message
                            }) if message.ends_with("requires Bosonic braiding")
                        ),
                        "{what}: only the U15 boundary may separate the storages: {rejection:?}"
                    );
                    assert_eq!(got.dense_data().is_err(), *compact, "{what}: storage form");
                    Some(image(&got))
                }
                (Err(got), Err(want)) => {
                    assert_eq!(report(&got), report(&want), "{what}");
                    None
                }
                (Err(got), Ok(_)) => panic!("{what}: compact failed where dense did not: {got}"),
            }
        })
        .collect()
}

fn assert_rows_close(mf: &[Option<Vec<Complex64>>], checked: &[Option<Vec<Complex64>>]) {
    assert_eq!(mf.len(), checked.len());
    for (row, (lhs, rhs)) in mf.iter().zip(checked).enumerate() {
        let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
            panic!("row {row}: one mode rejected it");
        };
        assert_eq!(lhs.len(), rhs.len(), "row {row}");
        for (&x, &y) in lhs.iter().zip(rhs) {
            assert!(
                close(x, y),
                "row {row}: multiplicity-free {x} != checked {y}"
            );
        }
    }
}

fn z2_leg<R>(rule: R) -> GradedSpace<R>
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: TypedTensorModeDispatch<R>,
{
    GradedSpace::try_new(
        Arc::new(rule),
        SECTORS.iter().map(|&(p, k)| (Z2Irrep::new(p), k)),
    )
    .unwrap()
}

fn parity(sector: &Z2Irrep) -> f64 {
    f64::from(sector.parity())
}

#[test]
fn compact_compose_and_contract_agree_across_modes() {
    // What: compose and one-leg contract against a compact diagonal are one
    // mode-free arm: both modes give the same storage form and values, equal to
    // the dense route, for f64 and Complex64 and for every output order.
    use symtn::sector::Z2FusionRule;
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    assert_rows_close(
        &bond_rows(&bond_fixture(&z2_leg(Z2FusionRule), parity, real), &cases),
        &bond_rows(
            &bond_fixture(
                &z2_leg(CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0)),
                parity,
                real,
            ),
            &cases,
        ),
    );
    assert_rows_close(
        &bond_rows(
            &bond_fixture(&z2_leg(Z2FusionRule), parity, complex),
            &cases,
        ),
        &bond_rows(
            &bond_fixture(
                &z2_leg(CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0)),
                parity,
                complex,
            ),
            &cases,
        ),
    );
}

#[test]
fn fermionic_compact_compose_and_contract_agree_across_modes() {
    // What: the fermionic row. Composition is admitted in both modes and
    // agrees. Checked Generic fermionic contraction is unsupported on the
    // dense route, so the compact arms' values, every listed pattern
    // included, are checked against the multiplicity-free mode (whose dense
    // route `bond_rows` already checked).
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    for complex in [false, true] {
        let conv = move |x: f64| {
            if complex {
                Complex64::new(x, 0.5 * x - 0.25)
            } else {
                Complex64::new(x, 0.0)
            }
        };
        let mf = bond_rows(
            &bond_fixture(&z2_leg(FermionParityFusionRule), parity, conv),
            &cases,
        );
        let checked = bond_rows(
            &bond_fixture(
                &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
                parity,
                conv,
            ),
            &cases,
        );
        // Every pattern is an arm, reordered `D . D` included (#1870).
        assert_rows_close(&mf, &checked);
    }
}

#[test]
fn multiplicity_free_u1_compact_compose_and_contract_match_the_dense_route() {
    // What: a non-self-dual U(1) bond (charges ±1 swap under duality) takes the
    // same arms, including the bent outputs whose permute moves a dual sector.
    use symtn::sector::{U1FusionRule, U1Irrep};
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let charge = |sector: &U1Irrep| {
        if *sector == U1Irrep::new(1) {
            1.0
        } else {
            -1.0
        }
    };
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    bond_rows(&bond_fixture(&leg, charge, |x: f64| x), &cases);
    bond_rows(
        &bond_fixture(&leg, charge, |x: f64| Complex64::new(-x, 0.25 * x)),
        &cases,
    );
}

#[test]
fn fibonacci_complex_symbol_compact_compose_takes_the_arms() {
    // What: the complex-symbol coefficient lane takes the compact arms
    // (#1866); compose stays compact for `D * D` and scales for
    // `t * D` / `D * t`. Fibonacci is anyonic, so `contract` keeps rejecting
    // it before any arm, compact or not.
    use symtn::sector::{TableRule, TableSector};
    let leg = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 2), (TableSector::new(2), 3)],
    )
    .unwrap();
    let tau = |sector: &TableSector| f64::from(u8::from(*sector == TableSector::new(2)));
    let fixture = bond_fixture(&leg, tau, |x: f64| Complex64::new(x, 0.5 - x));
    bond_rows(&fixture, &COMPOSE_CASES);
    let dense = fixture.d.materialize().unwrap();
    for (what, _, lhs, rhs, spec) in CONTRACT_CASES {
        let spec = spec.unwrap();
        let compact = pick(&fixture, lhs).contract(pick(&fixture, rhs), &spec);
        let swap = |code| {
            if code == 'd' {
                &dense
            } else {
                pick(&fixture, code)
            }
        };
        let materialized = swap(lhs).contract(swap(rhs), &spec);
        assert_eq!(
            format!("{}", compact.unwrap_err()),
            format!("{}", materialized.unwrap_err()),
            "{what}"
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su2_compact_compose_and_contract_agree_across_modes() {
    // What: the non-Abelian row. The bent outputs carry SU(2) bending
    // coefficients through the mode's own permute.
    use symtn::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
    let su2 = [(0usize, 2usize), (1, 3), (2, 1)];
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        su2.iter()
            .map(|&(twice, k)| (SU2Irrep::from_twice_spin(twice), k)),
    )
    .unwrap();
    let checked_leg = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        su2.iter().map(|&(twice, k)| (vec![twice as i64], k)),
    )
    .unwrap();
    let mf_label = |sector: &SU2Irrep| sector.twice_spin() as f64;
    let checked_label = |sector: &Vec<i64>| sector[0] as f64;
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    let real = |x: f64| x;
    assert_rows_close(
        &bond_rows(&bond_fixture(&mf_leg, mf_label, real), &cases),
        &bond_rows(&bond_fixture(&checked_leg, checked_label, real), &cases),
    );
    let complex = |x: f64| Complex64::new(0.5 * x, x - 1.0);
    assert_rows_close(
        &bond_rows(&bond_fixture(&mf_leg, mf_label, complex), &cases),
        &bond_rows(&bond_fixture(&checked_leg, checked_label, complex), &cases),
    );
}

#[test]
fn declined_compact_arms_report_the_dense_route_error() {
    // What: a pair the arms cannot answer — providers with different rule
    // identities, or a dense leg that is not the bond — falls through to the
    // dense route, so compact and materialized operands fail alike.
    macro_rules! assert_same_error {
        ($compact:expr, $dense:expr, $what:expr) => {
            assert_eq!(
                format!("{}", $compact.unwrap_err()),
                format!("{}", $dense.unwrap_err()),
                "{}",
                $what
            )
        };
    }
    let real = |x: f64| x;

    let fixture = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
        parity,
        real,
    );
    let foreign = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -0.5)),
        parity,
        real,
    );
    let (d, foreign_d) = (&fixture.d, &foreign.d);
    let (dense_d, dense_foreign) = (d.materialize().unwrap(), foreign_d.materialize().unwrap());
    assert_same_error!(
        d.compose(foreign_d),
        dense_d.compose(&dense_foreign),
        "D * D'"
    );
    assert_same_error!(
        fixture.a.compose(foreign_d),
        fixture.a.compose(&dense_foreign),
        "t * D'"
    );
    assert_same_error!(
        d.contract(&foreign.b, &spec(&[1], &[0], &[0], &[1, 2])),
        dense_d.contract(&foreign.b, &spec(&[1], &[0], &[0], &[1, 2])),
        "D . t'"
    );

    for (mode, errors) in [
        (
            "multiplicity-free",
            non_composable_errors(|| FermionParityFusionRule),
        ),
        (
            "checked",
            non_composable_errors(|| CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
        ),
    ] {
        for (what, (compact, dense)) in ["t * D", "D . t"].into_iter().zip(errors) {
            assert_eq!(compact, dense, "non-composable {what} ({mode})");
        }
    }
}

/// `t * D` and `D . t` where `t`'s facing leg is not the bond, on compact and
/// on materialized `D`: `[(compact, dense); 2]` error messages.
fn non_composable_errors<R>(rule: impl Fn() -> R) -> [(String, String); 2]
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: TypedTensorContractDispatch<R, f64> + TypedTensorConstructionDispatch<R, f64>,
{
    let other = GradedSpace::try_new(
        Arc::new(rule()),
        [(Z2Irrep::new(0), 1), (Z2Irrep::new(1), 4)],
    )
    .unwrap();
    let t =
        TensorMap::<_, f64>::from_subblock_fn(&host_runtime(), [&other], [&other], |_, index| {
            index[0] as f64 - 0.5 * index[1] as f64
        })
        .unwrap();
    let fixture = bond_fixture(&z2_leg(rule()), parity, |x: f64| x);
    let dense = fixture.d.materialize().unwrap();
    let message = |result: Result<TensorMap<R, f64>, _>| format!("{}", result.unwrap_err());
    let one_leg = spec(&[1], &[0], &[0], &[1]);
    [
        (message(t.compose(&fixture.d)), message(t.compose(&dense))),
        (
            message(fixture.d.contract(&t, &one_leg)),
            message(dense.contract(&t, &one_leg)),
        ),
    ]
}

#[test]
fn checked_compose_and_identity_order_contract_arms_query_no_provider() {
    // What: the compose arms and an arm whose output needs no reordering read
    // only the stored spectrum and the admitted rule identity, so they succeed
    // even when every fallible provider query would fail.
    let rule = CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0);
    let fixture = bond_fixture(&z2_leg(rule), parity, |x: f64| x);
    let provider = fixture.d.provider();
    provider.fail_queries.store(true, Ordering::Relaxed);
    for (what, compact, lhs, rhs, spec) in
        COMPOSE_CASES
            .into_iter()
            .chain([CONTRACT_CASES[0], CONTRACT_CASES[2], CONTRACT_CASES[5]])
    {
        let (lhs, rhs) = (pick(&fixture, lhs), pick(&fixture, rhs));
        let result = match spec {
            None => lhs.compose(rhs),
            Some(spec) => lhs.contract(rhs, &spec),
        };
        let result = result.unwrap_or_else(|error| panic!("{what}: {error}"));
        assert_eq!(result.dense_data().is_err(), compact, "{what}");
    }
    provider.fail_queries.store(false, Ordering::Relaxed);
}

#[test]
fn checked_reordered_contract_arm_failure_is_typed_and_nonpublishing() {
    // What: the reordered `t · D` arm lays its result out with one checked
    // permute staging. A provider failing there is reported as the typed
    // plan error of that staging; the operands are unchanged and no complete
    // layout or transform coefficients are published for the destination.
    const ISOLATED: &str = "SYMTN_COMPACT_CONTRACT_PUBLICATION_ISOLATED";
    const NAME: &str = "checked_reordered_contract_arm_failure_is_typed_and_nonpublishing";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    use symtn::typed::{CheckedGenericPlanError, GenericTensorError};
    use symtn_core::{structure_cache_info, StructureCacheKind};
    symtn_core::clear_structure_caches();
    let fixture = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
        parity,
        |x: f64| x,
    );
    let source = fixture.a.dense_data().unwrap().to_vec();
    let spectrum = symtn::expert::diagonal_spectrum(&fixture.d).unwrap();
    let published = || {
        [
            StructureCacheKind::DegeneracyStructure,
            StructureCacheKind::CompletedTreeTransformer,
            StructureCacheKind::TreeTransformCoefficients,
        ]
        .map(|kind| {
            let info = structure_cache_info(kind);
            (info.admissions(), info.entries(), info.charged_bytes())
        })
    };
    let before = published();
    fixture
        .d
        .provider()
        .fail_queries
        .store(true, Ordering::Relaxed);
    let result = fixture
        .a
        .contract(&fixture.d, &spec(&[2], &[0], &[0], &[1, 2]));
    fixture
        .d
        .provider()
        .fail_queries
        .store(false, Ordering::Relaxed);
    assert!(
        matches!(
            result,
            Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
                InvalidSector
            )))
        ),
        "{result:?}"
    );
    assert_eq!(published(), before);
    assert_eq!(fixture.a.dense_data().unwrap(), source);
    assert_eq!(
        symtn::expert::diagonal_spectrum(&fixture.d).unwrap(),
        spectrum
    );
}

/// Warm `D * D` bytes and whether the result stayed compact. The caller
/// asserts the byte bound before the storage form, so a densifying
/// regression is caught by the bound on its own.
fn warmed_square_bytes<R>(tensor: &TensorMap<R, f64>) -> (u64, bool)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, f64>,
{
    black_box(tensor.compose(tensor).unwrap());
    let (square, allocs) = counting_alloc::measure(|| black_box(tensor.compose(tensor).unwrap()));
    (allocs.bytes, square.dense_data().is_err())
}

#[test]
fn compact_square_allocates_linear_in_the_spectrum_in_both_modes() {
    // What: `D * D` of a compact diagonal costs Σk, never a k² payload.
    let _measurement = counting_alloc::serial();
    let dense_payload = (K * K * std::mem::size_of::<f64>()) as u64;
    let spectrum_growth = (2 * K * std::mem::size_of::<f64>()) as u64;
    let rows = [
        (
            "multiplicity-free",
            warmed_square_bytes(&odd_diagonal(FermionParityFusionRule, K)),
            warmed_square_bytes(&odd_diagonal(FermionParityFusionRule, 2 * K)),
        ),
        (
            "checked",
            warmed_square_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                K,
            )),
            warmed_square_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                2 * K,
            )),
        ),
    ];
    for (mode, (small, _), (large, compact)) in rows {
        assert!(
            large.saturating_sub(small) <= spectrum_growth,
            "{mode} D * D is not linear in k: {small} -> {large} bytes"
        );
        assert!(
            large < dense_payload,
            "{mode} D * D allocated a dense payload: {large} bytes"
        );
        assert!(compact, "{mode} D * D densified");
    }
}

#[test]
fn fermionic_dual_open_leg_compact_compose_and_contract_agree_across_modes() {
    // What: the dense operands carry a dual open leg (`a: W ⊗ W' ← W`), so
    // the reordered arms permute a dual leg past the bond under a fermionic
    // rule; both modes agree with the multiplicity-free dense route.
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    fn fixture<R>(leg: GradedSpace<R>) -> BondFixture<R, Complex64>
    where
        R: TypedSectorAdmission<Sector = Z2Irrep>,
        R::Mode: TypedTensorConstructionDispatch<R, Complex64>,
    {
        let open = leg.try_dual().unwrap();
        bond_fixture_on(&host_runtime(), &leg, &open, parity, |x: f64| {
            Complex64::new(x, 0.25 - x)
        })
    }
    let mf = bond_rows(&fixture(z2_leg(FermionParityFusionRule)), &cases);
    let checked = bond_rows(
        &fixture(z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0))),
        &cases,
    );
    // No row is rejected: reordered `D . D` is an arm too (#1870).
    assert_rows_close(&mf, &checked);
}

// ---- Rank-(1,1) swap and braid ----
//
// TensorKit `cfaa073e` `permute(d::DiagonalTensorMap, ((2,), (1,)))` and
// `transpose` (`src/tensors/diagonal.jl:215-273`) keep the result diagonal on
// `dual(d.domain)`, scaling each block by the one coefficient of the
// rank-one tree pair. `braid` has no diagonal method: `similar(t, T, V)`
// (`src/tensors/abstracttensor.jl:614-618`) makes it a dense `TensorMap`.
// QSpace has no braiding. The per-sector swap coefficients below were read
// from TensorKit at `cfaa073e` (script and output in the #1866
// record): fZ2 permute θ-sign `-1` on the odd sector and transpose `+1`; Z2,
// U(1) and SU(2) (j = 0, 1/2, 1) all `1`.

const TRANSFORMS: [&str; 3] = ["permute", "transpose", "braid"];

fn transform<R, D>(
    tensor: &TensorMap<R, D>,
    op: &str,
) -> Result<TensorMap<R, D>, <R::Mode as TypedTensorModeDispatch<R>>::FacadeError>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    match op {
        "permute" => tensor.permute(&[1], &[0]),
        "transpose" => tensor.transpose(&[1], &[0]),
        _ => tensor.braid(&[1], &[0], &[0, 1]),
    }
}

/// Runs `ops` on the compact `d` and on its materialized twin: a swap stays
/// compact on `dual(leg) ← dual(leg)` and a braid is dense, and either equals
/// the dense route bit for bit (one compiled coefficient authority). Returns
/// each result's dense image.
fn transform_rows<R, D>(d: &TensorMap<R, D>, leg: &GradedSpace<R>, ops: &[&str]) -> Vec<Vec<D>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
    D: TensorScalar + std::fmt::Debug,
{
    let dense = d.materialize().unwrap();
    let dual = leg.try_dual().unwrap();
    ops.iter()
        .map(|&op| {
            let got = transform(d, op).unwrap();
            let want = transform(&dense, op).unwrap();
            assert_eq!(
                got.dense_data().is_err(),
                op != "braid",
                "{op}: storage form"
            );
            assert_eq!(got.codomain()[0], dual, "{op}: codomain is dual(domain)");
            assert_eq!(got.domain()[0], dual, "{op}: domain is dual(codomain)");
            assert_eq!(got.codomain(), want.codomain(), "{op}");
            assert_eq!(got.domain(), want.domain(), "{op}");
            let image = got.materialize().unwrap().dense_data().unwrap().to_vec();
            assert_eq!(
                image,
                want.dense_data().unwrap(),
                "{op}: compact != dense route"
            );
            image
        })
        .collect()
}

fn assert_images_close<D: Copy>(mf: &[Vec<D>], checked: &[Vec<D>])
where
    Complex64: From<D>,
{
    assert_eq!(mf.len(), checked.len());
    for (row, (lhs, rhs)) in mf.iter().zip(checked).enumerate() {
        assert_eq!(lhs.len(), rhs.len(), "row {row}");
        for (&x, &y) in lhs.iter().zip(rhs) {
            let (x, y) = (Complex64::from(x), Complex64::from(y));
            assert!(
                close(x, y),
                "row {row}: multiplicity-free {x} != checked {y}"
            );
        }
    }
}

/// The swapped spectrum of a Z2-graded compact diagonal against the
/// TensorKit coefficient: `-1` only on the odd sector of a fermionic permute.
fn assert_z2_swap_spectrum<R, D>(d: &TensorMap<R, D>, fermionic: bool)
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedTensorRootDispatch<R>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
    D: TensorScalar + std::fmt::Debug,
    Complex64: From<D>,
{
    let source = d.diagview().unwrap();
    for op in ["permute", "transpose"] {
        let swapped = transform(d, op).unwrap().diagview().unwrap();
        assert_eq!(swapped.len(), source.len(), "{op}");
        for (got, want) in swapped.iter().zip(&source) {
            assert_eq!(got.sector, want.sector, "{op}: Z2 is self-dual");
            let sign = if fermionic && op == "permute" && got.sector.parity() == 1 {
                -1.0
            } else {
                1.0
            };
            for (&x, &y) in got.values.iter().zip(&want.values) {
                assert_eq!(
                    Complex64::from(x),
                    Complex64::from(y) * sign,
                    "{op} sector {:?}",
                    got.sector
                );
            }
        }
    }
}

fn z2_diagonal<R, D>(rule: R, conv: impl Fn(f64) -> D) -> (TensorMap<R, D>, GradedSpace<R>)
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    let leg = z2_leg(rule);
    let d = TensorMap::diagonal(
        &host_runtime(),
        &leg,
        SECTORS.iter().map(|&(p, k)| SectorSpectrum {
            sector: Z2Irrep::new(p),
            values: (0..k).map(|i| conv(value(p, i))).collect(),
        }),
    )
    .unwrap();
    (d, leg)
}

#[test]
fn compact_swap_and_braid_agree_across_modes() {
    // What: the rank-(1,1) swap and braid of a compact diagonal are one
    // mode-free arm: both modes keep it compact on
    // `dual(domain)` with TensorKit's coefficient, and both read the braid's
    // dense result from the spectrum, equal to the dense route.
    use symtn::sector::Z2FusionRule;
    macro_rules! rows {
        ($fermionic:expr, $mf:expr, $checked:expr, $conv:expr) => {{
            let (mf, mf_leg) = z2_diagonal($mf, $conv);
            let (checked, checked_leg) = z2_diagonal($checked, $conv);
            assert_z2_swap_spectrum(&mf, $fermionic);
            assert_z2_swap_spectrum(&checked, $fermionic);
            assert_images_close(
                &transform_rows(&mf, &mf_leg, &TRANSFORMS),
                &transform_rows(&checked, &checked_leg, &TRANSFORMS),
            );
            checked
        }};
    }
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    let bosonic = || CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0);
    let fermionic = || CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0);
    for checked in [
        rows!(false, Z2FusionRule, bosonic(), real),
        rows!(true, FermionParityFusionRule, fermionic(), real),
    ] {
        let swapped = checked.permute(&[1], &[0]).unwrap();
        let root = swapped.map_diagonal(|x: f64| x.abs().sqrt()).unwrap();
        assert!(root.dense_data().is_err());
    }
    rows!(false, Z2FusionRule, bosonic(), complex);
    rows!(true, FermionParityFusionRule, fermionic(), complex);
}

#[test]
fn checked_anyonic_compact_swap_and_braid_match_the_dense_route() {
    // What: a checked-only anyonic rule with a non-sign R takes the same arm;
    // the result is the dense route's, whether the dense route answers or
    // rejects.
    let (d, leg) = z2_diagonal(
        CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6).with_odd_r(0.8),
        |x: f64| x,
    );
    let dense = d.materialize().unwrap();
    for op in TRANSFORMS {
        match (transform(&d, op), transform(&dense, op)) {
            (Ok(_), Ok(_)) => {
                transform_rows(&d, &leg, &[op]);
            }
            (Err(got), Err(want)) => assert_eq!(format!("{got}"), format!("{want}"), "{op}"),
            (got, want) => panic!("{op}: compact {got:?} vs dense {want:?}"),
        }
    }
}

#[test]
fn multiplicity_free_u1_compact_swap_moves_each_spectrum_to_the_dual_sector() {
    // What: U(1) charges ±1 trade places under duality, so each swapped
    // spectrum sits on the dual charge with coefficient 1.
    use symtn::sector::{U1FusionRule, U1Irrep};
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    for complex in [false, true] {
        let conv = |x: f64| Complex64::new(x, if complex { 1.0 - x } else { 0.0 });
        let d = TensorMap::diagonal(
            &host_runtime(),
            &leg,
            [(-1, 2), (1, 3)].map(|(charge, k)| SectorSpectrum {
                sector: U1Irrep::new(charge),
                values: (0..k)
                    .map(|i| conv(f64::from(charge) - 0.5 * i as f64))
                    .collect(),
            }),
        )
        .unwrap();
        transform_rows(&d, &leg, &TRANSFORMS);
        let source = d.diagview().unwrap();
        for op in ["permute", "transpose"] {
            for entry in transform(&d, op).unwrap().diagview().unwrap() {
                let partner = source
                    .iter()
                    .find(|s| s.sector == U1Irrep::new(-entry.sector.charge()))
                    .unwrap();
                assert_eq!(entry.values, partner.values, "{op} {:?}", entry.sector);
            }
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su2_compact_swap_and_braid_agree_across_modes() {
    // What: the non-Abelian row. TensorKit's swap coefficient is 1 on
    // j = 0, 1/2 and 1 (up to its rounding); both modes keep the spectrum and
    // agree with each other and with their dense routes.
    use symtn::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
    let su2 = [(0usize, 2usize), (1, 3), (2, 1)];
    let tensorkit_coefficient = [1.0, 1.0, 1.0];
    let values = |twice: usize, k: usize| -> Vec<f64> {
        (0..k).map(|i| 0.5 + (twice * 4 + i) as f64).collect()
    };
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        su2.iter()
            .map(|&(twice, k)| (SU2Irrep::from_twice_spin(twice), k)),
    )
    .unwrap();
    let checked_leg = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        su2.iter().map(|&(twice, k)| (vec![twice as i64], k)),
    )
    .unwrap();
    let mf = TensorMap::<_, f64>::diagonal(
        &host_runtime(),
        &mf_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(twice),
            values: values(twice, k),
        }),
    )
    .unwrap();
    let checked = TensorMap::<_, f64>::diagonal(
        &host_runtime(),
        &checked_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: vec![twice as i64],
            values: values(twice, k),
        }),
    )
    .unwrap();
    assert_images_close(
        &transform_rows(&mf, &mf_leg, &TRANSFORMS),
        &transform_rows(&checked, &checked_leg, &TRANSFORMS),
    );
    for op in ["permute", "transpose"] {
        let mf_swapped = transform(&mf, op).unwrap().diagview().unwrap();
        let checked_swapped = transform(&checked, op).unwrap().diagview().unwrap();
        for (index, &(twice, k)) in su2.iter().enumerate() {
            let mf_entry = mf_swapped
                .iter()
                .find(|e| e.sector == SU2Irrep::from_twice_spin(twice))
                .unwrap();
            let checked_entry = checked_swapped
                .iter()
                .find(|e| e.sector == vec![twice as i64])
                .unwrap();
            for (i, want) in values(twice, k).into_iter().enumerate() {
                let want = Complex64::from(want * tensorkit_coefficient[index]);
                assert!(
                    close(Complex64::from(mf_entry.values[i]), want),
                    "{op} mf j={twice}/2"
                );
                assert!(
                    close(Complex64::from(checked_entry.values[i]), want),
                    "{op} checked j={twice}/2"
                );
            }
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_dual_leg_checked_compact_swap_and_braid_match_the_dense_route() {
    // What: non-self-dual SU(3) irreps on a dual leg (`[1,0]^2 + [0,1]`
    // dualized): the swapped spectrum moves to the destination's dual labels,
    // read from the staged destination, and equals the dense route.
    use symtn::sector::SUNFusionRule;
    let leg = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(3).unwrap()),
        [(vec![1i64, 0], 2), (vec![0, 1], 1)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    for complex in [false, true] {
        let conv = |x: f64| Complex64::new(x, if complex { 0.5 * x } else { 0.0 });
        let d = TensorMap::diagonal(
            &host_runtime(),
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0i64, 1],
                    values: vec![conv(0.5), conv(-1.25)],
                },
                SectorSpectrum {
                    sector: vec![1i64, 0],
                    values: vec![conv(2.0)],
                },
            ],
        )
        .unwrap();
        transform_rows(&d, &leg, &TRANSFORMS);
    }
}

#[test]
fn fibonacci_complex_symbol_compact_transpose_and_braid_take_the_arm() {
    // What: the complex-symbol coefficient lane takes the same arm (#1866
    // X6): transpose stays compact and the braid is read from the spectrum.
    // Fibonacci is anyonic, so `permute` is rejected alike for the compact
    // and the materialized input.
    use symtn::sector::{TableRule, TableSector};
    let leg = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 2), (TableSector::new(2), 3)],
    )
    .unwrap();
    let d = bond_fixture(
        &leg,
        |sector| f64::from(u8::from(*sector == TableSector::new(2))),
        |x: f64| Complex64::new(x, 0.5 - x),
    )
    .d;
    transform_rows(&d, &leg, &["transpose", "braid"]);
    assert_eq!(
        format!("{}", d.permute(&[1], &[0]).unwrap_err()),
        format!(
            "{}",
            d.materialize().unwrap().permute(&[1], &[0]).unwrap_err()
        ),
    );
}

#[test]
fn checked_compact_braid_decline_replays_in_one_staging() {
    // What: an R symbol beyond f32 (1e100 overflows, 1e-100 underflows)
    // declines the spectrum read; the arm replays the densified spectrum with
    // the structure it already staged, so the result is the dense route's
    // bit for bit (NaN placement included) and the provider is queried
    // exactly as often as by the dense route, cold and warm.
    for (salt, odd_r) in [(1u8, 1e100), (3, 1e-100)] {
        let rule = |salt| {
            CheckedZ2::new(BraidingStyleKind::Anyonic, 1.0)
                .with_odd_r(odd_r)
                .with_salt(salt)
        };
        let ledger = |tensor: &TensorMap<CheckedZ2, f32>| {
            let provider = tensor.provider();
            let mut counts = Vec::new();
            let mut result = None;
            for _ in 0..2 {
                provider.queries.store(0, Ordering::Relaxed);
                result = Some(tensor.braid(&[1], &[0], &[0, 1]).unwrap());
                counts.push(provider.queries.load(Ordering::Relaxed));
            }
            (counts, result.unwrap())
        };
        let values = |x: f64| x as f32;
        let (compact, _) = z2_diagonal(rule(salt), values);
        let dense = z2_diagonal(rule(salt + 1), values).0.materialize().unwrap();
        let (compact_counts, got) = ledger(&compact);
        let (dense_counts, want) = ledger(&dense);
        assert!(compact_counts[0] > 0);
        assert_eq!(
            compact_counts, dense_counts,
            "R = {odd_r:e}: provider queries"
        );
        let (got, want) = (got.dense_data().unwrap(), want.dense_data().unwrap());
        assert!(
            want.iter().any(|x| x.is_nan()) || want.iter().filter(|&&x| x != 0.0).count() < 5,
            "R = {odd_r:e}: the fixture must make the coefficient unrepresentable"
        );
        assert_eq!(got.len(), want.len());
        for (&x, &y) in got.iter().zip(want) {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "R = {odd_r:e}: {x} != dense {y}"
            );
        }
    }
}

#[test]
fn checked_compact_swap_queries_the_provider_like_the_dense_route() {
    // What: the swap arm stages, resolves and commits with the dense route's
    // own staging, so cold and warm provider queries are the dense route's.
    let rule = |salt| CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0).with_salt(salt);
    let ledger = |tensor: &TensorMap<CheckedZ2, f64>| {
        let provider = tensor.provider();
        (0..2)
            .map(|_| {
                provider.queries.store(0, Ordering::Relaxed);
                black_box(tensor.permute(&[1], &[0]).unwrap());
                provider.queries.load(Ordering::Relaxed)
            })
            .collect::<Vec<_>>()
    };
    let (compact, _) = z2_diagonal(rule(5), |x: f64| x);
    let dense = z2_diagonal(rule(6), |x: f64| x).0.materialize().unwrap();
    let compact_counts = ledger(&compact);
    assert!(compact_counts[0] > 0);
    assert_eq!(compact_counts, ledger(&dense));
}

fn warmed_swap_bytes<R>(tensor: &TensorMap<R, f64>) -> (u64, bool)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, f64>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
{
    black_box(tensor.permute(&[1], &[0]).unwrap());
    let (swapped, allocs) =
        counting_alloc::measure(|| black_box(tensor.permute(&[1], &[0]).unwrap()));
    (allocs.bytes, swapped.dense_data().is_err())
}

#[test]
fn compact_swap_allocates_linear_in_the_spectrum_in_both_modes() {
    // What: the swap of a compact diagonal costs Σk, never a k² payload.
    let _measurement = counting_alloc::serial();
    let dense_payload = (K * K * std::mem::size_of::<f64>()) as u64;
    let spectrum_growth = (2 * K * std::mem::size_of::<f64>()) as u64;
    let rows = [
        (
            "multiplicity-free",
            warmed_swap_bytes(&odd_diagonal(FermionParityFusionRule, K)),
            warmed_swap_bytes(&odd_diagonal(FermionParityFusionRule, 2 * K)),
        ),
        (
            "checked",
            warmed_swap_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                K,
            )),
            warmed_swap_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                2 * K,
            )),
        ),
    ];
    for (mode, (small, _), (large, compact)) in rows {
        assert!(
            large.saturating_sub(small) <= spectrum_growth,
            "{mode} swap is not linear in k: {small} -> {large} bytes"
        );
        assert!(
            large < dense_payload,
            "{mode} swap allocated a dense payload: {large} bytes"
        );
        assert!(compact, "{mode} swap densified");
    }
}

// ---- Full trace ----

/// The full trace of `d` over `pair`, and the same trace of its dense twin.
fn full_traces<R, D>(d: &TensorMap<R, D>, pair: (usize, usize)) -> (Complex64, Complex64)
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorTraceDispatch<R, D>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
    D: TensorScalar,
    Complex64: From<D>,
{
    let trace = |t: &TensorMap<R, D>| {
        Complex64::from(t.trace_pairs(&[pair]).unwrap().dense_data().unwrap()[0])
    };
    (trace(d), trace(&d.materialize().unwrap()))
}

#[test]
fn compact_full_trace_agrees_across_modes() {
    // What: the full trace of a compact bond is one mode-free arm: both
    // modes read the spectrum with the
    // dense route's compiled coefficients and agree with it, with each other
    // and with the hand supertrace (θ(odd) = -1 for fZ2).
    use symtn::sector::Z2FusionRule;
    macro_rules! rows {
        ($mf:expr, $checked:expr, $odd:expr, $conv:expr) => {{
            let conv = $conv;
            let hand = SECTORS
                .iter()
                .map(|&(p, k)| {
                    let factor = if p == 1 { $odd } else { 1.0 };
                    (0..k)
                        .map(|i| Complex64::from(conv(value(p, i))) * factor)
                        .sum::<Complex64>()
                })
                .sum::<Complex64>();
            let (mf, _) = z2_diagonal($mf, conv);
            let (checked, _) = z2_diagonal($checked, conv);
            for pair in [(0, 1), (1, 0)] {
                let (mf_compact, mf_dense) = full_traces(&mf, pair);
                let (checked_compact, checked_dense) = full_traces(&checked, pair);
                assert!(
                    close(mf_compact, mf_dense),
                    "{pair:?}: mf {mf_compact} vs dense {mf_dense}"
                );
                assert!(
                    close(checked_compact, checked_dense),
                    "{pair:?}: checked {checked_compact} vs dense {checked_dense}"
                );
                assert!(
                    close(mf_compact, checked_compact),
                    "{pair:?}: mf vs checked"
                );
                if pair == (0, 1) {
                    assert!(
                        close(mf_compact, hand),
                        "{pair:?}: {mf_compact} vs hand {hand}"
                    );
                }
            }
        }};
    }
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    let bosonic = || CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0);
    let fermionic = || CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0);
    rows!(Z2FusionRule, bosonic(), 1.0, real);
    rows!(FermionParityFusionRule, fermionic(), -1.0, real);
    rows!(Z2FusionRule, bosonic(), 1.0, complex);
    rows!(FermionParityFusionRule, fermionic(), -1.0, complex);
}

#[test]
fn checked_anyonic_compact_full_trace_matches_the_dense_route() {
    // What: a checked-only non-sign θ(odd) = 0.6 enters the compiled
    // coefficient; the compact trace is the dense route's on either pair
    // order.
    let (d, _) = z2_diagonal(
        CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6).with_odd_r(0.8),
        |x: f64| Complex64::new(x, 1.0 - x),
    );
    for pair in [(0, 1), (1, 0)] {
        match (
            d.trace_pairs(&[pair]),
            d.materialize().unwrap().trace_pairs(&[pair]),
        ) {
            (Ok(compact), Ok(dense)) => {
                let (compact, dense) = (
                    compact.dense_data().unwrap()[0],
                    dense.dense_data().unwrap()[0],
                );
                assert!(
                    close(compact, dense),
                    "{pair:?}: {compact} vs dense {dense}"
                );
            }
            (Err(got), Err(want)) => assert_eq!(format!("{got}"), format!("{want}"), "{pair:?}"),
            (got, want) => panic!("{pair:?}: compact {got:?} vs dense {want:?}"),
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su2_compact_full_trace_agrees_across_modes() {
    // What: the non-Abelian row; SU(2) is bosonic, so the hand trace is
    // `Σ_j (2j + 1) Σ_i s_j[i]`.
    use symtn::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
    let su2 = [(0usize, 2usize), (1, 3), (2, 1)];
    let values = |twice: usize, k: usize| -> Vec<f64> {
        (0..k).map(|i| 0.5 + (twice * 4 + i) as f64).collect()
    };
    let hand: f64 = su2
        .iter()
        .map(|&(twice, k)| (twice + 1) as f64 * values(twice, k).iter().sum::<f64>())
        .sum();
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        su2.iter()
            .map(|&(twice, k)| (SU2Irrep::from_twice_spin(twice), k)),
    )
    .unwrap();
    let checked_leg = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        su2.iter().map(|&(twice, k)| (vec![twice as i64], k)),
    )
    .unwrap();
    let mf = TensorMap::<_, f64>::diagonal(
        &host_runtime(),
        &mf_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(twice),
            values: values(twice, k),
        }),
    )
    .unwrap();
    let checked = TensorMap::<_, f64>::diagonal(
        &host_runtime(),
        &checked_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: vec![twice as i64],
            values: values(twice, k),
        }),
    )
    .unwrap();
    for pair in [(0, 1), (1, 0)] {
        let (mf_compact, mf_dense) = full_traces(&mf, pair);
        let (checked_compact, checked_dense) = full_traces(&checked, pair);
        for (what, got, want) in [
            ("mf vs dense", mf_compact, mf_dense),
            ("checked vs dense", checked_compact, checked_dense),
            ("mf vs checked", mf_compact, checked_compact),
            ("mf vs hand", mf_compact, Complex64::from(hand)),
        ] {
            assert!(close(got, want), "{pair:?} {what}: {got} vs {want}");
        }
    }
}

#[test]
fn checked_compact_full_trace_queries_the_provider_like_the_dense_route() {
    // What: the trace arm only reads the dense route's compiled terms, so
    // cold and warm provider queries are the dense route's.
    let rule = |salt| CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0).with_salt(salt);
    let ledger = |tensor: &TensorMap<CheckedZ2, f64>| {
        let provider = tensor.provider();
        (0..2)
            .map(|_| {
                provider.queries.store(0, Ordering::Relaxed);
                black_box(tensor.trace_pairs(&[(0, 1)]).unwrap());
                provider.queries.load(Ordering::Relaxed)
            })
            .collect::<Vec<_>>()
    };
    let (compact, _) = z2_diagonal(rule(7), |x: f64| x);
    let dense = z2_diagonal(rule(8), |x: f64| x).0.materialize().unwrap();
    let compact_counts = ledger(&compact);
    assert!(compact_counts[0] > 0);
    assert_eq!(compact_counts, ledger(&dense));
}

#[test]
fn checked_compact_full_trace_failure_is_typed_and_nonpublishing() {
    // What: a provider failing during the trace compile fails the compact
    // trace with the dense route's typed error, and nothing is published.
    const ISOLATED: &str = "SYMTN_COMPACT_TRACE_PUBLICATION_ISOLATED";
    const NAME: &str = "checked_compact_full_trace_failure_is_typed_and_nonpublishing";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    use symtn::typed::{CheckedGenericPlanError, GenericTensorError};
    use symtn_core::{structure_cache_info, StructureCacheKind};
    let (compact, _) = z2_diagonal(
        CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0).with_salt(9),
        |x: f64| x,
    );
    let dense = compact.materialize().unwrap();
    let spectrum = symtn::expert::diagonal_spectrum(&compact).unwrap();
    symtn_core::clear_structure_caches();
    let published = || {
        StructureCacheKind::ALL.map(|kind| {
            let info = structure_cache_info(kind);
            (info.admissions(), info.entries(), info.charged_bytes())
        })
    };
    let before = published();
    let provider = compact.provider();
    provider.fail_queries.store(true, Ordering::Relaxed);
    let result = compact.trace_pairs(&[(0, 1)]);
    let dense_result = dense.trace_pairs(&[(0, 1)]);
    provider.fail_queries.store(false, Ordering::Relaxed);
    assert!(
        matches!(
            result,
            Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
                InvalidSector
            )))
        ),
        "{result:?}"
    );
    assert_eq!(
        format!("{}", result.unwrap_err()),
        format!("{}", dense_result.unwrap_err())
    );
    assert_eq!(published(), before);
    assert_eq!(
        symtn::expert::diagonal_spectrum(&compact).unwrap(),
        spectrum
    );
}

fn warmed_trace_bytes<R>(tensor: &TensorMap<R, f64>) -> u64
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorTraceDispatch<R, f64>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
{
    black_box(tensor.trace_pairs(&[(0, 1)]).unwrap());
    counting_alloc::measure(|| black_box(tensor.trace_pairs(&[(0, 1)]).unwrap()))
        .1
        .bytes
}

#[test]
fn compact_full_trace_allocates_no_spectrum_sized_buffer_in_both_modes() {
    // What: the full trace of a compact diagonal reads the spectrum in place:
    // from k to 2k the bytes do not grow by a spectrum, and stay below one
    // dense payload.
    let _measurement = counting_alloc::serial();
    let dense_payload = (K * K * std::mem::size_of::<f64>()) as u64;
    let spectrum_growth = (K * std::mem::size_of::<f64>()) as u64;
    let rows = [
        (
            "multiplicity-free",
            warmed_trace_bytes(&odd_diagonal(FermionParityFusionRule, K)),
            warmed_trace_bytes(&odd_diagonal(FermionParityFusionRule, 2 * K)),
        ),
        (
            "checked",
            warmed_trace_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                K,
            )),
            warmed_trace_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                2 * K,
            )),
        ),
    ];
    for (mode, small, large) in rows {
        assert!(
            large.saturating_sub(small) < spectrum_growth,
            "{mode} trace grows with k: {small} -> {large} bytes"
        );
        assert!(
            large < dense_payload,
            "{mode} trace allocated a dense payload: {large} bytes"
        );
    }
}

/// The Host-vs-CUDA column of the matrix: the multiplicity-free rows of every
/// compact operation run on a device copy (`to_cuda` densifies the compact
/// operand) and come back equal to the Host result densified. Checked × CUDA
/// and Fibonacci × CUDA are compile-time capability absences (the device
/// operations are bounded `MultiplicityFreeRigidSymbols<Scalar = f64>`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device"]
fn cuda_column_matches_the_host_for_every_compact_row() {
    use symtn::sector::{U1FusionRule, U1Irrep, Z2FusionRule};
    fn assert_same<D: TensorScalar + Copy>(
        what: &str,
        host: &TensorMap<impl TypedSectorAdmission, D>,
        device: &[D],
    ) where
        Complex64: From<D>,
    {
        let host = host.materialize().unwrap();
        let host = host.dense_data().unwrap();
        assert_eq!(host.len(), device.len(), "{what}");
        for (&x, &y) in host.iter().zip(device) {
            let (x, y) = (Complex64::from(x), Complex64::from(y));
            assert!(close(y, x), "{what}: device {y} != host {x}");
        }
    }
    macro_rules! cuda_rows {
        ($row:expr, $leg:expr, $label:expr, $conv:expr) => {{
            let row: &str = $row;
            let leg = $leg;
            let host = bond_fixture_on(&fixtures::cuda_runtime(), &leg, &leg, $label, $conv);
            let device = [&host.d, &host.e, &host.a, &host.b].map(|t| t.to_cuda().unwrap());
            let pick_device = |code: char| match code {
                'd' => &device[0],
                'e' => &device[1],
                'a' => &device[2],
                _ => &device[3],
            };
            for op in TRANSFORMS {
                let expected = transform(&host.d, op).unwrap();
                let actual = match op {
                    "permute" => device[0].permute(&[1], &[0]),
                    "transpose" => device[0].transpose(&[1], &[0]),
                    _ => device[0].braid(&[1], &[0], &[0, 1]),
                }
                .unwrap()
                .to_host()
                .unwrap();
                assert_same(
                    &format!("{row} {op}"),
                    &expected,
                    actual.dense_data().unwrap(),
                );
            }
            for legs in LEG_SETS {
                for direction in [Direction::Forward, Direction::Inverse] {
                    let expected = host.d.twist(legs, direction).unwrap();
                    let actual = device[0].twist(legs, direction).unwrap().to_host().unwrap();
                    assert_same(
                        &format!("{row} twist {legs:?} {direction:?}"),
                        &expected,
                        actual.dense_data().unwrap(),
                    );
                }
            }
            for (what, _, lhs, rhs, spec) in COMPOSE_CASES.into_iter().chain(CONTRACT_CASES) {
                let (host_lhs, host_rhs) = (pick(&host, lhs), pick(&host, rhs));
                let (device_lhs, device_rhs) = (pick_device(lhs), pick_device(rhs));
                let (expected, actual) = match spec {
                    None => (host_lhs.compose(host_rhs), device_lhs.compose(device_rhs)),
                    Some(spec) => (
                        host_lhs.contract(host_rhs, &spec),
                        device_lhs.contract(device_rhs, &spec),
                    ),
                };
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => assert_same(
                        &format!("{row} {what}"),
                        &expected,
                        actual.to_host().unwrap().dense_data().unwrap(),
                    ),
                    (Err(_), Err(_)) => {}
                    (expected, actual) => panic!(
                        "{row} {what}: host {:?} vs device {:?}",
                        expected.err().map(|e| e.to_string()),
                        actual.err().map(|e| e.to_string())
                    ),
                }
            }
        }};
    }
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    let u1 = || {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(-1), 2), (U1Irrep::new(1), 3)],
        )
        .unwrap()
    };
    let charge = |sector: &U1Irrep| f64::from(sector.charge());
    cuda_rows!("Z2 f64", z2_leg(Z2FusionRule), parity, real);
    cuda_rows!("Z2 c64", z2_leg(Z2FusionRule), parity, complex);
    cuda_rows!("fZ2 f64", z2_leg(FermionParityFusionRule), parity, real);
    cuda_rows!("fZ2 c64", z2_leg(FermionParityFusionRule), parity, complex);
    cuda_rows!("U(1) f64", u1(), charge, real);
    cuda_rows!("U(1) c64", u1(), charge, complex);
}

/// Checked tree-transform `_into` (#1870): which error a call with two
/// faults reports does not depend on cache history. A warm checked transform
/// defers its plan preflight from staging to resolution (#2154), so the
/// destination's space is compared only after resolution in that mode;
/// otherwise a `SpaceMismatch` would hide the preflight error only when warm.
/// Lives here for the `CheckedZ2` rule, whose braiding is a parameter.
#[test]
fn checked_transform_into_reports_the_same_error_cold_and_warm() {
    let runtime = host_runtime();
    for (salt, braiding, expected) in [
        // Non-planar on an anyonic rule: the preflight error wins.
        (0x71, BraidingStyleKind::Anyonic, "UnsupportedBraidingStyle"),
        // Admissible motion: only the destination is wrong.
        (0x72, BraidingStyleKind::Bosonic, "SpaceMismatch"),
    ] {
        let leg = z2_leg(CheckedZ2::new(braiding, 0.6).with_salt(salt));
        let source =
            TensorMap::<_, f64>::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, index| {
                index.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        // `[W, W] <- [W]`, where the result is `[W] <- [W, W]`.
        let mut wrong = source.axpby(0.0, &source, 0.0).unwrap();
        let mut into = || {
            format!(
                "{:?}",
                source
                    .permute_into(&[0], &[1, 2], &mut wrong, 1.0, 0.0)
                    .unwrap_err()
            )
        };
        let cold = into();
        // Warms the destination memo of these axes; a braid is admitted
        // where the permute is not.
        drop(source.braid(&[0], &[1, 2], &[0, 1, 2]).unwrap());
        let warm = into();
        assert!(cold.contains(expected), "{braiding:?}: {cold}");
        assert_eq!(warm, cold, "{braiding:?}");
    }
}

// ---- compact-source transform `_into` ----

/// Every rank-(1, 1) transform a compact source's `_into` takes.
const BOND_INTO_OPS: [&str; 6] = [
    "identity",
    "permute",
    "transpose",
    "braid",
    "repartition (0, 2)",
    "repartition (2, 0)",
];

fn bond_eager<R, D>(
    tensor: &TensorMap<R, D>,
    op: &str,
) -> Result<TensorMap<R, D>, <R::Mode as TypedTensorModeDispatch<R>>::FacadeError>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    match op {
        "identity" => tensor.permute(&[0], &[1]),
        "repartition (0, 2)" => tensor.repartition(0),
        "repartition (2, 0)" => tensor.repartition(2),
        _ => transform(tensor, op),
    }
}

fn bond_into<R, D>(
    tensor: &TensorMap<R, D>,
    op: &str,
    destination: &mut TensorMap<R, D>,
    alpha: D,
    beta: D,
) -> Result<(), <R::Mode as TypedTensorModeDispatch<R>>::FacadeError>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    match op {
        "identity" => tensor.permute_into(&[0], &[1], destination, alpha, beta),
        "permute" => tensor.permute_into(&[1], &[0], destination, alpha, beta),
        "transpose" => tensor.transpose_into(&[1], &[0], destination, alpha, beta),
        "braid" => tensor.braid_into(&[1], &[0], &[0, 1], destination, alpha, beta),
        _ => tensor.repartition_into(destination, alpha, beta),
    }
}

/// A zero destination on the legs of `op`'s eager result.
fn bond_destination<R, D>(tensor: &TensorMap<R, D>, op: &str) -> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedTensorConstructionDispatch<R, D>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
    D: TensorScalar,
{
    let eager = bond_eager(tensor, op).unwrap();
    TensorMap::from_subblock_fn(
        tensor.runtime(),
        &eager.codomain(),
        &eager.domain(),
        |_, _| D::from_real(0.0),
    )
    .unwrap()
}

/// The warm `_into` of `op` from `tensor`: its allocations, counting those
/// of at least `size` bytes as matched.
fn warmed_bond_into_allocs<R, D>(
    tensor: &TensorMap<R, D>,
    op: &str,
    size: usize,
) -> counting_alloc::Allocs
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedTensorConstructionDispatch<R, D>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
    D: TensorScalar,
{
    let mut destination = bond_destination(tensor, op);
    let (one, half) = (D::from_real(1.0), D::from_real(0.5));
    bond_into(tensor, op, &mut destination, one, half).unwrap();
    let (result, allocs) = counting_alloc::measure_matching(size..=usize::MAX, || {
        bond_into(tensor, op, &mut destination, one, half)
    });
    result.unwrap();
    assert!(tensor.dense_data().is_err(), "{op}: the source densified");
    allocs
}

#[test]
fn compact_transform_into_allocates_no_dense_source_in_both_modes() {
    // What: a compact source's `_into` reads its diagonal from the resolved
    // structure into the caller's destination: no allocation as large as the
    // `k²` dense image the decline route would build, in either mode.
    let _measurement = counting_alloc::serial();
    let dense_payload = K * K * std::mem::size_of::<f64>();
    let mf = odd_diagonal(FermionParityFusionRule, K);
    let checked = odd_diagonal(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0), K);
    for op in BOND_INTO_OPS {
        for (mode, allocs) in [
            (
                "multiplicity-free",
                warmed_bond_into_allocs(&mf, op, dense_payload),
            ),
            (
                "checked",
                warmed_bond_into_allocs(&checked, op, dense_payload),
            ),
        ] {
            assert_eq!(allocs.matched_calls, 0, "{mode} {op}: {allocs:?}");
            assert!(
                allocs.bytes < dense_payload as u64,
                "{mode} {op}: {allocs:?}"
            );
        }
    }
}

#[test]
fn compact_braid_into_decline_replays_the_densified_source() {
    // What: a braid coefficient beyond f32 declines the diagonal read, so the
    // `_into` replays the densified spectrum through the structure it staged:
    // the dense route's value (NaN placement included), and the one
    // operation-local `k²` buffer the read otherwise avoids.
    let _measurement = counting_alloc::serial();
    for (salt, odd_r) in [(0x81u8, 1e100), (0x83, 1e-100)] {
        let rule = CheckedZ2::new(BraidingStyleKind::Anyonic, 1.0)
            .with_odd_r(odd_r)
            .with_salt(salt);
        let leg = GradedSpace::try_new(Arc::new(rule), [(Z2Irrep::ODD, K)]).unwrap();
        let compact = TensorMap::<_, f32>::diagonal(
            &host_runtime(),
            &leg,
            [SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: (0..K).map(|i| 1.0 + i as f32).collect(),
            }],
        )
        .unwrap();
        let want = compact
            .materialize()
            .unwrap()
            .braid(&[1], &[0], &[0, 1])
            .unwrap();
        let mut destination = bond_destination(&compact, "braid");
        bond_into(&compact, "braid", &mut destination, 1.0, 0.0).unwrap();
        let (got, want) = (
            destination.dense_data().unwrap(),
            want.dense_data().unwrap(),
        );
        assert_eq!(got.len(), want.len());
        assert!(
            want.iter().any(|x| x.is_nan()) || want.iter().filter(|&&x| x != 0.0).count() < K,
            "R = {odd_r:e}: the fixture must make the coefficient unrepresentable"
        );
        for (&x, &y) in got.iter().zip(want) {
            assert!(
                x == y || (x.is_nan() && y.is_nan()),
                "R = {odd_r:e}: {x} != dense {y}"
            );
        }
        let allocs = warmed_bond_into_allocs(&compact, "braid", K * K * std::mem::size_of::<f32>());
        assert!(allocs.matched_calls >= 1, "R = {odd_r:e}: {allocs:?}");
    }
}

#[test]
fn checked_compact_transform_into_queries_the_provider_like_eager() {
    // What: the compact `_into` stages and resolves with the eager compact
    // arm's own staging, so cold and warm provider queries are eager's.
    let ledger =
        |tensor: &TensorMap<CheckedZ2, f64>, leg: &GradedSpace<CheckedZ2>, op: &str, into: bool| {
            let provider = tensor.provider();
            // Built on the swapped legs directly: deriving them through the eager
            // op would warm the very caches the cold count reads.
            let dual = leg.try_dual().unwrap();
            let mut destination = into.then(|| {
                TensorMap::from_subblock_fn(tensor.runtime(), [&dual], [&dual], |_, _| 0.0).unwrap()
            });
            (0..2)
                .map(|_| {
                    provider.queries.store(0, Ordering::Relaxed);
                    match &mut destination {
                        Some(destination) => bond_into(tensor, op, destination, 1.0, 0.5).unwrap(),
                        None => drop(black_box(bond_eager(tensor, op).unwrap())),
                    }
                    provider.queries.load(Ordering::Relaxed)
                })
                .collect::<Vec<_>>()
        };
    for (index, op) in ["permute", "braid"].into_iter().enumerate() {
        let salt = 0x90 + 2 * index as u8;
        let rule = |salt| CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0).with_salt(salt);
        // Two rule identities, so each ledger starts cold.
        let (eager, eager_leg) = z2_diagonal(rule(salt), |x: f64| x);
        let (into, into_leg) = z2_diagonal(rule(salt + 1), |x: f64| x);
        let eager_counts = ledger(&eager, &eager_leg, op, false);
        let into_counts = ledger(&into, &into_leg, op, true);
        assert!(eager_counts[0] > 0, "{op}");
        assert_eq!(into_counts, eager_counts, "{op}");
    }
}

fn checked_fermionic(salt: u8) -> CheckedZ2 {
    CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0).with_salt(salt)
}

/// `legs` rebuilt on a fresh checked fZ2 rule `salt`: a destination built on
/// them warms none of that rule's transforms.
fn z2_legs_on(legs: Vec<GradedSpace<CheckedZ2>>, salt: u8) -> Vec<GradedSpace<CheckedZ2>> {
    legs.iter()
        .map(|leg| {
            let fresh = z2_leg(checked_fermionic(salt));
            if leg.is_dual() {
                fresh.try_dual().unwrap()
            } else {
                fresh
            }
        })
        .collect()
}

/// The lazy adjoint `[W] ← [W, W]` of a checked fZ2 tensor on rule `salt`.
fn lazy_z2(runtime: &Runtime, salt: u8) -> TensorMap<CheckedZ2, f64> {
    let leg = z2_leg(checked_fermionic(salt));
    TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg], |_, index| {
        index.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap()
    .adjoint()
    .unwrap()
}

/// A zero destination for `lazy.permute(&[1, 0], &[2])` on rule `salt`.
fn lazy_z2_destination(runtime: &Runtime, salt: u8) -> TensorMap<CheckedZ2, f64> {
    let shape = lazy_z2(runtime, 0xff).permute(&[1, 0], &[2]).unwrap();
    TensorMap::from_subblock_fn(
        runtime,
        &z2_legs_on(shape.codomain(), salt),
        &z2_legs_on(shape.domain(), salt),
        |_, _| 0.0,
    )
    .unwrap()
}

#[test]
fn checked_lazy_transform_into_queries_no_more_than_eager() {
    // What: eager transforms the parent and re-derives the adjoint result
    // space; the `_into` reads the parent's storage through the logical
    // source, with no result space to derive. Cold it asks the provider at
    // most what eager asks; warm neither asks anything.
    let runtime = host_runtime();
    let ledger = |salt: u8, into: bool| {
        let lazy = lazy_z2(&runtime, salt);
        let mut destination = lazy_z2_destination(&runtime, salt);
        let provider = lazy.provider();
        (0..2)
            .map(|_| {
                provider.queries.store(0, Ordering::Relaxed);
                if into {
                    lazy.permute_into(&[1, 0], &[2], &mut destination, 1.0, 0.0)
                        .unwrap();
                } else {
                    drop(black_box(lazy.permute(&[1, 0], &[2]).unwrap()));
                }
                assert!(lazy.dense_data().is_err(), "the lazy source materialized");
                provider.queries.load(Ordering::Relaxed)
            })
            .collect::<Vec<_>>()
    };
    let (eager, into) = (ledger(0xa2, false), ledger(0xa3, true));
    assert!(
        into[0] > 0 && into[0] <= eager[0],
        "eager {eager:?} into {into:?}"
    );
    assert_eq!(into[1], eager[1], "warm");
}

#[test]
fn checked_compact_and_lazy_transform_into_failure_is_typed_and_nonpublishing() {
    // What: a provider failing in the staging of a compact or lazy-adjoint
    // source's `_into` is the typed plan error, leaves the destination bit
    // for bit, and publishes no layout, transformer or coefficients; the
    // succeeding call after it does (#2063).
    const ISOLATED: &str = "SYMTN_COMPACT_TRANSFORM_INTO_PUBLICATION_ISOLATED";
    const NAME: &str = "checked_compact_and_lazy_transform_into_failure_is_typed_and_nonpublishing";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    use symtn::typed::{CheckedGenericPlanError, GenericTensorError};
    use symtn_core::{structure_cache_info, StructureCacheKind};
    symtn_core::clear_structure_caches();
    let published = || {
        [
            StructureCacheKind::CompletedTreeTransformer,
            StructureCacheKind::TreeTransformCoefficients,
        ]
        .map(|kind| {
            let info = structure_cache_info(kind);
            (info.admissions(), info.entries(), info.charged_bytes())
        })
    };
    let runtime = host_runtime();
    let (compact, leg) = z2_diagonal(checked_fermionic(0xc0), |x: f64| x);
    let dual = leg.try_dual().unwrap();
    let mut compact_destination =
        TensorMap::from_subblock_fn(compact.runtime(), [&dual], [&dual], |_, index| {
            index[0] as f64 + 0.25
        })
        .unwrap();
    let lazy = lazy_z2(&runtime, 0xc1);
    let mut lazy_destination = lazy_z2_destination(&runtime, 0xc1);
    let bits = |t: &TensorMap<CheckedZ2, f64>| {
        t.dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    macro_rules! fails_then_publishes {
        ($name:expr, $provider:expr, $destination:expr, |$d:ident| $into:expr) => {{
            let (name, before, destination) = ($name, published(), bits(&$destination));
            $provider.fail_queries.store(true, Ordering::Relaxed);
            let result = {
                let $d = &mut $destination;
                $into
            };
            $provider.fail_queries.store(false, Ordering::Relaxed);
            assert!(
                matches!(
                    result,
                    Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
                        InvalidSector
                    )))
                ),
                "{name}: {result:?}"
            );
            assert_eq!(bits(&$destination), destination, "{name}: destination");
            assert_eq!(published(), before, "{name}: the failure published");
            let $d = &mut $destination;
            $into.unwrap();
            assert_ne!(published(), before, "{name}: the success published nothing");
        }};
    }
    fails_then_publishes!(
        "compact braid",
        compact.provider(),
        compact_destination,
        |d| compact.braid_into(&[1], &[0], &[0, 1], d, 1.0, 0.5)
    );
    fails_then_publishes!("lazy permute", lazy.provider(), lazy_destination, |d| lazy
        .permute_into(&[1, 0], &[2], d, 1.0, 0.5));
}

// ---- contraction `_into` ----

/// A zero destination on the legs of `case`'s contraction of `fixture`,
/// built on those legs directly: the shape is read from a throwaway rule, as
/// deriving it through the eager contraction would warm the caches a cold
/// count reads.
fn contract_destination(
    fixture: &BondFixture<CheckedZ2, f64>,
    case: BondCase,
) -> TensorMap<CheckedZ2, f64> {
    let (_, _, lhs, rhs, spec) = case;
    let scratch = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0).with_salt(0xff)),
        parity,
        |x: f64| x,
    );
    let shape = pick(&scratch, lhs)
        .contract(pick(&scratch, rhs), &spec.unwrap())
        .unwrap();
    let leg = &fixture.d.codomain()[0];
    let on = |legs: Vec<GradedSpace<CheckedZ2>>| {
        legs.iter()
            .map(|l| {
                if l.is_dual() {
                    leg.try_dual().unwrap()
                } else {
                    leg.clone()
                }
            })
            .collect::<Vec<_>>()
    };
    TensorMap::from_subblock_fn(
        fixture.d.runtime(),
        &on(shape.codomain()),
        &on(shape.domain()),
        |_, _| 0.0,
    )
    .unwrap()
}

#[test]
fn checked_contract_into_queries_the_provider_like_eager() {
    // What: the dense `contract_into` runs eager's staging and planning, and
    // the compact one eager's arm (the braiding gate, plus one permute
    // staging when reordered), so cold and warm provider queries equal
    // eager's on every pattern both take alike.
    let ledger = |salt: u8, case: usize, into: bool| {
        let rule = CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0).with_salt(salt);
        let fixture = bond_fixture(&z2_leg(rule), parity, |x: f64| x);
        let (what, _, lhs, rhs, spec) = DENSE_AND_ARM_CASES[case];
        let mut destination = contract_destination(&fixture, DENSE_AND_ARM_CASES[case]);
        let (lhs, rhs, spec) = (pick(&fixture, lhs), pick(&fixture, rhs), spec.unwrap());
        let provider = fixture.d.provider();
        let counts = (0..2)
            .map(|_| {
                provider.queries.store(0, Ordering::Relaxed);
                if into {
                    lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5)
                        .unwrap_or_else(|error| panic!("{what}: {error}"));
                } else {
                    drop(black_box(lhs.contract(rhs, &spec).unwrap()));
                }
                provider.queries.load(Ordering::Relaxed)
            })
            .collect::<Vec<_>>();
        (what, counts)
    };
    let mut cold_queries = Vec::new();
    for case in 0..DENSE_AND_ARM_CASES.len() {
        let salt = 0xd0 + 2 * case as u8;
        let (what, eager) = ledger(salt, case, false);
        let (_, into) = ledger(salt + 1, case, true);
        assert_eq!(into, eager, "{what}");
        cold_queries.push(eager[0]);
    }
    // The ledger is not vacuous: the reordered rows stage on the provider.
    assert!(
        cold_queries.iter().any(|&queries| queries > 0),
        "{cold_queries:?}"
    );
}

/// The dense `a · b` and every compact arm pattern: eager and `_into` take
/// each alike.
const DENSE_AND_ARM_CASES: [BondCase; 12] = [
    (
        "contract a . b",
        false,
        'a',
        'b',
        Some(spec(&[2], &[0], &[0, 1], &[2, 3])),
    ),
    (
        "contract a . b reordered",
        false,
        'a',
        'b',
        Some(spec(&[2], &[0], &[0, 2], &[1, 3])),
    ),
    CONTRACT_CASES[0],
    CONTRACT_CASES[2],
    CONTRACT_CASES[3],
    CONTRACT_CASES[4],
    CONTRACT_CASES[5],
    CONTRACT_CASES[6],
    CONTRACT_CASES[7],
    CONTRACT_CASES[1],
    CONTRACT_CASES[8],
    CONTRACT_CASES[9],
];

#[test]
fn checked_contract_into_failure_is_typed_and_nonpublishing() {
    // What: a provider failing in the staging of a dense `contract_into`, or
    // in the permute staging of a reordered compact arm, is the typed plan
    // error, leaves the destination bit for bit and publishes nothing; so
    // does a shared destination rejected after staging. The succeeding call
    // after them publishes (#2063).
    const ISOLATED: &str = "SYMTN_CONTRACT_INTO_PUBLICATION_ISOLATED";
    const NAME: &str = "checked_contract_into_failure_is_typed_and_nonpublishing";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    use symtn::typed::{CheckedGenericPlanError, Error, GenericTensorError};
    use symtn_core::{structure_cache_info, StructureCacheKind};
    symtn_core::clear_structure_caches();
    let published = || {
        [
            StructureCacheKind::DegeneracyStructure,
            StructureCacheKind::CompletedTreeTransformer,
            StructureCacheKind::TreeTransformCoefficients,
        ]
        .map(|kind| {
            let info = structure_cache_info(kind);
            (info.admissions(), info.entries(), info.charged_bytes())
        })
    };
    let bits = |t: &TensorMap<CheckedZ2, f64>| {
        t.dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    for (salt, case) in [(0xe0u8, DENSE_AND_ARM_CASES[1]), (0xe1, CONTRACT_CASES[4])] {
        let (what, _, lhs, rhs, spec) = case;
        let rule = CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0).with_salt(salt);
        let fixture = bond_fixture(&z2_leg(rule), parity, |x: f64| x);
        let mut destination = contract_destination(&fixture, case);
        let (lhs, rhs, spec) = (pick(&fixture, lhs), pick(&fixture, rhs), spec.unwrap());
        let provider = fixture.d.provider();
        let (before, original) = (published(), bits(&destination));

        provider.fail_queries.store(true, Ordering::Relaxed);
        let result = lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5);
        provider.fail_queries.store(false, Ordering::Relaxed);
        assert!(
            matches!(
                result,
                Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
                    InvalidSector
                )))
            ),
            "{what}: {result:?}"
        );
        assert_eq!(bits(&destination), original, "{what}: provider failure");
        assert_eq!(published(), before, "{what}: the failure published");

        let shared = destination.clone();
        let result = lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5);
        assert!(
            matches!(
                result,
                Err(GenericTensorError::Facade(Error::DestinationShared))
            ),
            "{what}: {result:?}"
        );
        drop(shared);
        assert_eq!(bits(&destination), original, "{what}: shared");
        assert_eq!(published(), before, "{what}: the shared call published");

        lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5)
            .unwrap();
        assert_ne!(published(), before, "{what}: the success published nothing");
    }
}

#[test]
fn checked_contract_into_reports_the_same_error_cold_and_warm() {
    // What: staging (where the checked dense route refuses a fermionic rule)
    // runs before the destination match, and no contraction staging step is
    // deferred to planning on a memo hit, so a call reports eager's staging
    // error cold and warm, on the result's space and on a wrong one alike.
    let rule = CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0).with_salt(0xf0);
    let fixture = bond_fixture(&z2_leg(rule), parity, |x: f64| x);
    let case = DENSE_AND_ARM_CASES[1];
    let (_, _, lhs, rhs, spec) = case;
    let (lhs, rhs, spec) = (pick(&fixture, lhs), pick(&fixture, rhs), spec.unwrap());
    let eager = lhs.contract(rhs, &spec).unwrap_err().to_string();
    let mut right = contract_destination(&fixture, case);
    let mut wrong = TensorMap::from_subblock_fn(
        fixture.a.runtime(),
        &fixture.a.codomain(),
        &fixture.a.domain(),
        |_, _| 0.0,
    )
    .unwrap();
    for pass in ["cold", "warm"] {
        for destination in [&mut right, &mut wrong] {
            let error = lhs
                .contract_into(rhs, &spec, destination, 1.0, 0.5)
                .unwrap_err();
            assert_eq!(error.to_string(), eager, "{pass}");
        }
        drop(black_box(lhs.contract(rhs, &spec)));
    }
}

/// `contract_into` of every compact arm case equals `α·eager + β·dst₀`
/// (tolerance oracle) for β ∈ {0 (NaN-seeded), 0.5, 1}.
fn contract_into_rows<R, D>(fixture: &BondFixture<R, D>, alpha: D)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D> + TypedTensorConstructionDispatch<R, D>,
    <R::Mode as TypedTensorModeDispatch<R>>::FacadeError: std::fmt::Debug,
    D: TensorScalar + std::fmt::Debug,
    Complex64: From<D>,
{
    for (what, _, lhs, rhs, spec) in CONTRACT_CASES {
        let (lhs, rhs, spec) = (pick(fixture, lhs), pick(fixture, rhs), spec.unwrap());
        let eager = lhs.contract(rhs, &spec).unwrap().materialize().unwrap();
        for beta in [0.0, 0.5, 1.0] {
            let mut next = 0.0;
            let mut destination = TensorMap::from_subblock_fn(
                fixture.d.runtime(),
                &eager.codomain(),
                &eager.domain(),
                |_, _| {
                    next += 0.375;
                    D::from_real(if beta == 0.0 { f64::NAN } else { next })
                },
            )
            .unwrap();
            let before: Vec<Complex64> = destination
                .dense_data()
                .unwrap()
                .iter()
                .map(|&x| Complex64::from(x))
                .collect();
            lhs.contract_into(rhs, &spec, &mut destination, alpha, D::from_real(beta))
                .unwrap();
            let alpha = Complex64::from(alpha);
            for ((&got, &e), &b) in destination
                .dense_data()
                .unwrap()
                .iter()
                .zip(eager.dense_data().unwrap())
                .zip(&before)
            {
                let want = alpha * Complex64::from(e)
                    + if beta == 0.0 {
                        Complex64::new(0.0, 0.0)
                    } else {
                        b * beta
                    };
                assert!(
                    close(Complex64::from(got), want),
                    "{what} beta {beta}: {got:?} vs {want}"
                );
            }
        }
    }
}

#[test]
fn compact_contract_into_is_alpha_eager_plus_beta_destination_in_every_mode() {
    // What: eager and `_into` take one route on every arm pattern, reordered
    // `D . D` included, so `_into` is eager's value with `α`/`β` in the
    // multiplicity-free and checked modes, bosonic and fermionic.
    use symtn::sector::Z2FusionRule;
    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    let alpha = Complex64::new(1.5, -0.5);
    contract_into_rows(
        &bond_fixture(&z2_leg(Z2FusionRule), parity, |x: f64| x),
        1.5,
    );
    contract_into_rows(&bond_fixture(&z2_leg(Z2FusionRule), parity, complex), alpha);
    contract_into_rows(
        &bond_fixture(&z2_leg(FermionParityFusionRule), parity, |x: f64| x),
        1.5,
    );
    contract_into_rows(
        &bond_fixture(&z2_leg(FermionParityFusionRule), parity, complex),
        alpha,
    );
    for braiding in [BraidingStyleKind::Bosonic, BraidingStyleKind::Fermionic] {
        let rule = || {
            CheckedZ2::new(
                braiding,
                if braiding == BraidingStyleKind::Bosonic {
                    1.0
                } else {
                    -1.0
                },
            )
        };
        contract_into_rows(&bond_fixture(&z2_leg(rule()), parity, |x: f64| x), 1.5);
        contract_into_rows(&bond_fixture(&z2_leg(rule()), parity, complex), alpha);
    }
}

// ---- trace `_into` ----

/// The trace sources `_into` reads: a dense `[W, W] ← [W, W]` and its lazy
/// adjoint (partial trace over `(0, 2)`), and a compact diagonal (full trace
/// over `(0, 1)`), each on a fresh checked fZ2 rule `salt`.
const TRACE_INTO_SOURCES: [&str; 3] = ["dense", "lazy", "compact"];

fn trace_into_source(runtime: &Runtime, salt: u8, kind: &str) -> TensorMap<CheckedZ2, f64> {
    if kind == "compact" {
        return z2_diagonal(checked_fermionic(salt), |x: f64| x).0;
    }
    let leg = z2_leg(checked_fermionic(salt));
    let dense = TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg, &leg], |_, index| {
        index.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    if kind == "lazy" {
        dense.adjoint().unwrap()
    } else {
        dense
    }
}

fn trace_into_pairs(kind: &str) -> &'static [(usize, usize)] {
    if kind == "compact" {
        &[(0, 1)]
    } else {
        &[(0, 2)]
    }
}

/// A destination for `kind`'s trace on rule `salt`, built without tracing
/// that source: `[W] ← [W]` on the legs directly; the scalar one as the full
/// trace of a differently shaped dense `[W₁] ← [W₁]`.
fn trace_into_destination(runtime: &Runtime, salt: u8, kind: &str) -> TensorMap<CheckedZ2, f64> {
    if kind == "compact" {
        let leg = GradedSpace::try_new(
            Arc::new(checked_fermionic(salt)),
            [(Z2Irrep::new(0), 1), (Z2Irrep::new(1), 1)],
        )
        .unwrap();
        return TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |_, _| 1.0)
            .unwrap()
            .trace_pairs(&[(0, 1)])
            .unwrap();
    }
    let leg = z2_leg(checked_fermionic(salt));
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |_, _| 0.0).unwrap()
}

#[test]
fn checked_trace_pairs_into_queries_the_provider_like_eager() {
    // What: `trace_pairs_into` runs eager's staging and compiled terms (the
    // compact read included) and compares the destination provider-free, so
    // cold and warm provider queries equal eager's for every source kind.
    let runtime = host_runtime();
    let mut cold = Vec::new();
    for (case, kind) in TRACE_INTO_SOURCES.into_iter().enumerate() {
        let ledger = |salt: u8, into: bool| {
            let source = trace_into_source(&runtime, salt, kind);
            let mut destination = trace_into_destination(source.runtime(), salt, kind);
            let pairs = trace_into_pairs(kind);
            let provider = source.provider();
            (0..2)
                .map(|_| {
                    provider.queries.store(0, Ordering::Relaxed);
                    if into {
                        source
                            .trace_pairs_into(pairs, &mut destination, 1.0, 0.5)
                            .unwrap();
                    } else {
                        drop(black_box(source.trace_pairs(pairs).unwrap()));
                    }
                    provider.queries.load(Ordering::Relaxed)
                })
                .collect::<Vec<_>>()
        };
        let salt = 0x40 + 2 * case as u8;
        let eager = ledger(salt, false);
        let into = ledger(salt + 1, true);
        assert_eq!(into, eager, "{kind}");
        cold.push(eager[0]);
    }
    // Not vacuous: every cold trace stages on the provider.
    assert!(cold.iter().all(|&queries| queries > 0), "{cold:?}");
}

#[test]
fn checked_trace_pairs_into_failure_is_typed_and_nonpublishing() {
    // What: a provider failing in the staging of `trace_pairs_into` is the
    // typed plan error, leaves the destination bit for bit and publishes
    // nothing; so does a shared destination rejected after staging. The
    // succeeding call after them publishes (#2063).
    const ISOLATED: &str = "SYMTN_TRACE_INTO_PUBLICATION_ISOLATED";
    const NAME: &str = "checked_trace_pairs_into_failure_is_typed_and_nonpublishing";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    use symtn::typed::{CheckedGenericPlanError, Error, GenericTensorError};
    use symtn_core::{structure_cache_info, StructureCacheKind};
    let runtime = host_runtime();
    let bits = |t: &TensorMap<CheckedZ2, f64>| {
        t.dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    for (case, kind) in TRACE_INTO_SOURCES.into_iter().enumerate() {
        let salt = 0x50 + case as u8;
        let source = trace_into_source(&runtime, salt, kind);
        let mut destination = trace_into_destination(source.runtime(), salt, kind);
        let pairs = trace_into_pairs(kind);
        symtn_core::clear_structure_caches();
        // What #2063 stages: the destination layout and the coefficients
        // (the sector algebra the preflight reads is not staged, as for
        // contraction).
        let published = || {
            [
                StructureCacheKind::DegeneracyStructure,
                StructureCacheKind::CompletedTreeTransformer,
                StructureCacheKind::TreeTransformCoefficients,
            ]
            .map(|kind| {
                let info = structure_cache_info(kind);
                (info.admissions(), info.entries(), info.charged_bytes())
            })
        };
        let provider = source.provider();
        let (before, original) = (published(), bits(&destination));

        provider.fail_queries.store(true, Ordering::Relaxed);
        let result = source.trace_pairs_into(pairs, &mut destination, 1.0, 0.5);
        provider.fail_queries.store(false, Ordering::Relaxed);
        assert!(
            matches!(
                result,
                Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
                    InvalidSector
                )))
            ),
            "{kind}: {result:?}"
        );
        assert_eq!(bits(&destination), original, "{kind}: provider failure");
        assert_eq!(published(), before, "{kind}: the failure published");

        let shared = destination.clone();
        let result = source.trace_pairs_into(pairs, &mut destination, 1.0, 0.5);
        assert!(
            matches!(
                result,
                Err(GenericTensorError::Facade(Error::DestinationShared))
            ),
            "{kind}: {result:?}"
        );
        drop(shared);
        assert_eq!(bits(&destination), original, "{kind}: shared");
        assert_eq!(published(), before, "{kind}: the shared call published");

        source
            .trace_pairs_into(pairs, &mut destination, 1.0, 0.5)
            .unwrap();
        assert_ne!(published(), before, "{kind}: the success published nothing");
    }
}

#[test]
fn checked_trace_pairs_into_reports_the_same_error_cold_and_warm() {
    // What: the HomSpace comparison and the pair duality are staging, which
    // runs in full on every call, so a non-dual pair reports eager's
    // duality error (on the result's space) or the destination's space
    // error (on a wrong one), cold and warm alike.
    let runtime = host_runtime();
    let leg = z2_leg(checked_fermionic(0x58));
    let dual = leg.try_dual().unwrap();
    // `(0, 2)` joins `W` with `W'`: the untraced legs are those of a valid
    // trace, the pair is not dual.
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&dual, &leg], |_, index| {
        index.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let eager = report(&source.trace_pairs(&[(0, 2)]).unwrap_err());
    let mut right = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 0.0).unwrap();
    let mut wrong =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 0.0).unwrap();
    let mut seen = Vec::new();
    for pass in ["cold", "warm"] {
        let right_error = report(
            &source
                .trace_pairs_into(&[(0, 2)], &mut right, 1.0, 0.5)
                .unwrap_err(),
        );
        let wrong_error = source
            .trace_pairs_into(&[(0, 2)], &mut wrong, 1.0, 0.5)
            .unwrap_err();
        assert!(
            matches!(
                cause::<symtn::typed::OperationError>(&wrong_error),
                Some(symtn::typed::OperationError::SpaceMismatch { message }) if message.contains("destination")
            ),
            "{pass}: {wrong_error:?}"
        );
        let wrong_error = report(&wrong_error);
        assert_eq!(right_error, eager, "{pass}");
        seen.push((right_error, wrong_error));
        // Warms the source's trace structures on a valid pair.
        drop(black_box(source.trace_pairs(&[(1, 3)]).unwrap()));
    }
    assert_eq!(seen[0], seen[1]);
}
