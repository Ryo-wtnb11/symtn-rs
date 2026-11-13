//! #1389 evidence probe, kept here uncompiled: copy it to
//! `tenet/examples/streaming_factor_digest.rs` of the revision under test and
//! build with `cargo build --release -p tenet-rs --example
//! streaming_factor_digest`. It runs every streaming
//! factorization site of `tenet-matrixalgebra` on fixed inputs and prints, per
//! site, a bit digest of every output, the live-heap peak above the call's
//! entry (process-wide), and the TeNeT session count. Built unchanged at the
//! base and at the change; the digests must match bit for bit.
//!
//! Usage: `streaming_factor_digest [threads]` (omit for default threads).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use tenet::prelude::*;
use tenet_core::{
    BraidingStyleKind, CheckedGenericFusion, FusionProductSpace, FusionRule, FusionStyleKind,
    FusionTensorMapSpace, FusionTreeHomSpace, RuleIdentity, SectorId, SectorLeg, SectorVec,
    TensorMap as CoreTensorMap, TensorMapSpace,
};
use tenet_dense::{cpu_session_stats, DefaultDenseExecutor};
use tenet_matrixalgebra::{
    eigh_full_dyn, eigh_full_dyn_checked_generic, left_null_dyn, left_null_dyn_checked_generic,
    left_polar_dyn_checked_generic, lq_compact_dyn, lq_compact_dyn_checked_generic,
    lq_compact_dyn_generic, pinv_direct_into_dyn, right_null_dyn, right_null_dyn_checked_generic,
    svd_compact_dyn_checked_generic, svd_compact_factors_dyn, svd_compact_factors_dyn_generic,
    BoundDynFactor, BoundDynamicTensorRef, BoundTensorMap, FactorScalar, SectorSpectrum,
};
use tenet_tensors::BoundDynamicFusionMapSpace;

struct PeakAllocator;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live =
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed) + layout.size() as isize;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

trait Sample: FactorScalar {
    fn sample(i: usize) -> Self;
    fn push_bits(&self, out: &mut Vec<u64>);
}

impl Sample for f64 {
    fn sample(i: usize) -> Self {
        ((i * 11 + 5) % 29) as f64 * 0.25 - 3.0
    }
    fn push_bits(&self, out: &mut Vec<u64>) {
        out.push(self.to_bits());
    }
}

impl Sample for Complex64 {
    fn sample(i: usize) -> Self {
        Complex64::new(f64::sample(i), ((i * 5 + 2) % 13) as f64 * 0.125 - 0.75)
    }
    fn push_bits(&self, out: &mut Vec<u64>) {
        out.extend([self.re.to_bits(), self.im.to_bits()]);
    }
}

fn fnv(bits: &[u64]) -> u64 {
    bits.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &word| {
        (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[derive(Default)]
struct Digest(Vec<u64>);

impl Digest {
    fn data<D: Sample>(&mut self, data: &[D]) -> &mut Self {
        self.0.push(data.len() as u64);
        data.iter().for_each(|x| x.push_bits(&mut self.0));
        self
    }
    fn factor<R, D: Sample>(&mut self, factor: &BoundDynFactor<R, D>) -> &mut Self {
        self.data(factor.data())
    }
    fn spectrum(&mut self, spectrum: &[SectorSpectrum]) -> &mut Self {
        for entry in spectrum {
            self.0.push(entry.sector.id() as u64);
            self.0.extend(entry.values.iter().map(|x| x.to_bits()));
        }
        self
    }
}

fn run(site: &str, call: impl FnOnce() -> Digest) {
    let sessions = cpu_session_stats().sessions_opened;
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let digest = call();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    println!(
        "{site:<44} digest={:016x} peak_bytes={peak:>7} sessions={}",
        fnv(&digest.0),
        cpu_session_stats().sessions_opened - sessions
    );
}

fn symmetrize<D: Sample>(
    structure: &tenet_core::BlockStructure,
    nout: usize,
    data: &mut [D],
) -> bool {
    let Some(regions) = structure.coupled_sector_regions(nout).unwrap() else {
        return false;
    };
    let half = D::from_real(0.5);
    for region in regions.iter() {
        let n = region.rows();
        let block = &mut data[region.range()];
        for c in 0..n {
            for r in 0..=c {
                let value = (block[r + n * c] + FactorScalar::adjoint(block[c + n * r])) * half;
                block[r + n * c] = value;
                block[c + n * r] = FactorScalar::adjoint(value);
            }
        }
    }
    true
}

fn u1_tensor<D: Sample>(hermitian: bool) -> BoundTensorMap<U1FusionRule, D, 2, 2> {
    let rule = U1FusionRule;
    let charges = [-1, 0, 1];
    let degeneracy = 2usize;
    let leg = || {
        SectorLeg::new(
            charges
                .iter()
                .map(|&q| (U1Irrep::new(q).sector_id(), degeneracy)),
            false,
        )
    };
    let dim = charges.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([dim, dim], [dim, dim]).unwrap(),
        homspace,
        &rule,
        vec![vec![degeneracy; 4]; keys],
    )
    .unwrap();
    let data: Vec<D> = (0..space.required_len().unwrap()).map(D::sample).collect();
    let mut tensor = CoreTensorMap::<D, 2, 2>::from_vec_with_fusion_space(data, space).unwrap();
    if hermitian {
        let mut data = tensor.data().to_vec();
        assert!(symmetrize(tensor.structure(), 2, &mut data));
        let space = tensor.fusion_space().unwrap().as_ref().clone();
        tensor = CoreTensorMap::from_vec_with_fusion_space(data, space).unwrap();
    }
    BoundTensorMap::try_new(Arc::new(rule), tensor).unwrap()
}

#[derive(Clone, Copy)]
struct ToyGenericRule;

impl FusionRule for ToyGenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

struct CheckedToyRule;

impl CheckedGenericFusion for CheckedToyRule {
    type Error = std::convert::Infallible;
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        ToyGenericRule.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(ToyGenericRule.dual(sector))
    }
    fn try_fusion_channels(&self, l: SectorId, r: SectorId) -> Result<SectorVec, Self::Error> {
        Ok(ToyGenericRule.fusion_channels(l, r))
    }
    fn try_fusion_channels_in_table(
        &self,
        l: SectorId,
        r: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(ToyGenericRule.fusion_channels(l, r))
    }
    fn try_nsymbol(&self, l: SectorId, r: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        Ok(ToyGenericRule.nsymbol(l, r, c))
    }
}

fn toy_leg(d0: usize, d1: usize) -> SectorLeg {
    SectorLeg::new([(SectorId::new(0), d0), (SectorId::new(1), d1)], false)
}

/// Codomain `[x(2,2), x(1,2)]`, domain `[x(2,1)]`: every coupled sector is
/// tall, and `x ⊗ x → x` has multiplicity 2.
fn toy_homspace() -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new([toy_leg(2, 2), toy_leg(1, 2)]),
        FusionProductSpace::new([toy_leg(2, 1)]),
    )
}

fn toy_endomorphism() -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new([toy_leg(1, 2), toy_leg(2, 1)]),
        FusionProductSpace::new([toy_leg(1, 2), toy_leg(2, 1)]),
    )
}

fn samples<D: Sample>(len: usize, seed: usize) -> Vec<D> {
    (0..len).map(|i| D::sample(i * 3 + seed)).collect()
}

fn sites<D: Sample>(threads: Option<usize>, dtype: &str) {
    let mut dense = match threads {
        Some(threads) => DefaultDenseExecutor::with_threads(threads).unwrap(),
        None => DefaultDenseExecutor::new(),
    };
    let dense = &mut dense;

    // Multiplicity-free U(1): canonical (direct-region) and adjoint-view
    // (matricization) layouts.
    let u1 = u1_tensor::<D>(false);
    let u1_direct = BoundDynamicTensorRef::try_new(u1.space(), u1.data()).unwrap();
    let adjoint = u1.space().adjoint_view().unwrap();
    let u1_matricized = BoundDynamicTensorRef::try_new(&adjoint, u1.data()).unwrap();
    run(&format!("{dtype} u1 lq direct"), || {
        let (l, q) = lq_compact_dyn(dense, &u1_direct).unwrap();
        let mut d = Digest::default();
        d.factor(&l).factor(&q);
        d
    });
    run(&format!("{dtype} u1 lq matricized"), || {
        let (l, q) = lq_compact_dyn(dense, &u1_matricized).unwrap();
        let mut d = Digest::default();
        d.factor(&l).factor(&q);
        d
    });
    run(&format!("{dtype} u1 svd matricized"), || {
        let (u, vh, s) = svd_compact_factors_dyn(dense, &u1_matricized).unwrap();
        let mut d = Digest::default();
        d.factor(&u).factor(&vh).spectrum(&s);
        d
    });
    run(&format!("{dtype} u1 left_null"), || {
        let mut d = Digest::default();
        d.factor(&left_null_dyn(dense, &u1_matricized).unwrap());
        d
    });
    run(&format!("{dtype} u1 right_null"), || {
        let mut d = Digest::default();
        d.factor(&right_null_dyn(dense, &u1_direct).unwrap());
        d
    });
    let hermitian = u1_tensor::<D>(true);
    let h_direct = BoundDynamicTensorRef::try_new(hermitian.space(), hermitian.data()).unwrap();
    let h_adjoint = hermitian.space().adjoint_view().unwrap();
    let h_matricized = BoundDynamicTensorRef::try_new(&h_adjoint, hermitian.data()).unwrap();
    for (label, input) in [("direct", &h_direct), ("matricized", &h_matricized)] {
        run(&format!("{dtype} u1 eigh {label}"), || {
            let (v, values) = eigh_full_dyn(dense, input, HermitianTol::DEFAULT).unwrap().into_parts();
            let mut d = Digest::default();
            d.factor(&v).spectrum(&values);
            d
        });
    }

    // Generic (unchecked) toy rule with fusion multiplicity.
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(ToyGenericRule),
        toy_homspace(),
    )
    .unwrap();
    let data = samples::<D>(space.space().required_len().unwrap(), 1);
    let generic_direct = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let generic_adjoint = space.adjoint_view().unwrap();
    let generic_matricized = BoundDynamicTensorRef::try_new(&generic_adjoint, &data).unwrap();
    for (label, input) in [
        ("direct", &generic_direct),
        ("matricized", &generic_matricized),
    ] {
        run(&format!("{dtype} generic lq {label}"), || {
            let (l, q) = lq_compact_dyn_generic(dense, input).unwrap();
            let mut d = Digest::default();
            d.factor(&l).factor(&q);
            d
        });
    }
    run(&format!("{dtype} generic svd matricized"), || {
        let (u, vh, s) = svd_compact_factors_dyn_generic(dense, &generic_matricized).unwrap();
        let mut d = Digest::default();
        d.factor(&u).factor(&vh).spectrum(&s);
        d
    });

    // Checked Generic toy rule.
    let provider = Arc::new(CheckedToyRule);
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        toy_homspace(),
    )
    .unwrap();
    let data = samples::<D>(checked.space().required_len().unwrap(), 2);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    run(&format!("{dtype} checked lq"), || {
        let (l, q) = lq_compact_dyn_checked_generic(dense, &input).unwrap();
        let mut d = Digest::default();
        d.factor(&l).factor(&q);
        d
    });
    run(&format!("{dtype} checked svd"), || {
        let (u, s, vh) = svd_compact_dyn_checked_generic(dense, &input).unwrap();
        let mut d = Digest::default();
        d.factor(&u).factor(&s).factor(&vh);
        d
    });
    run(&format!("{dtype} checked left_null"), || {
        let mut d = Digest::default();
        d.factor(&left_null_dyn_checked_generic(dense, &input).unwrap());
        d
    });
    run(&format!("{dtype} checked right_null"), || {
        let mut d = Digest::default();
        d.factor(&right_null_dyn_checked_generic(dense, &input).unwrap());
        d
    });
    // Polar and pinv need canonical coupled-sector storage: a 1 <- 1 map.
    let matrix = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([toy_leg(3, 2)]),
            FusionProductSpace::new([toy_leg(2, 1)]),
        ),
    )
    .unwrap();
    let data = samples::<D>(matrix.space().required_len().unwrap(), 4);
    let input = BoundDynamicTensorRef::try_new(&matrix, &data).unwrap();
    run(&format!("{dtype} checked left_polar"), || {
        let (w, p) = left_polar_dyn_checked_generic(dense, &input).unwrap();
        let mut d = Digest::default();
        d.factor(&w).factor(&p);
        d
    });
    let hom = matrix.space().homspace();
    let swapped = FusionTreeHomSpace::new(hom.domain().clone(), hom.codomain().clone());
    run(&format!("{dtype} checked pinv"), || {
        let output = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            swapped,
        )
        .unwrap();
        let mut d = Digest::default();
        d.factor(&pinv_direct_into_dyn(dense, &input, output, 1e-12).unwrap());
        d
    });
    let endo = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        toy_endomorphism(),
    )
    .unwrap();
    let mut data = samples::<D>(endo.space().required_len().unwrap(), 3);
    assert!(symmetrize(
        endo.space().structure(),
        endo.space().nout(),
        &mut data
    ));
    let input = BoundDynamicTensorRef::try_new(&endo, &data).unwrap();
    run(&format!("{dtype} checked eigh"), || {
        let (v, values) = eigh_full_dyn_checked_generic(dense, &input, HermitianTol::DEFAULT)
            .unwrap()
            .into_parts();
        let mut d = Digest::default();
        d.factor(&v).spectrum(&values);
        d
    });
}

/// Facade multiplicity-free fixtures over three symmetries.
macro_rules! facade {
    ($runtime:expr, $label:expr, $dtype:ty, $rule:expr, $sectors:expr) => {{
        let leg = GradedSpace::try_new($rule, $sectors.into_iter().map(|s| (s, 2))).unwrap();
        let a =
            TensorMap::<_, $dtype>::rand_with_seed($runtime, [&leg, &leg], [&leg], 1389).unwrap();
        run(&format!("{} lq", $label), || {
            let (l, q) = a.lq_compact().unwrap();
            let mut d = Digest::default();
            d.data(l.data()).data(q.data());
            d
        });
        run(&format!("{} left_null", $label), || {
            let mut d = Digest::default();
            d.data(a.left_null().unwrap().data());
            d
        });
        run(&format!("{} right_null", $label), || {
            let mut d = Digest::default();
            d.data(a.right_null().unwrap().data());
            d
        });
        let h = a.compose(&a.adjoint().unwrap()).unwrap();
        run(&format!("{} eigh", $label), || {
            let (w, v) = h.eigh_full().unwrap();
            let mut d = Digest::default();
            d.data(w.data()).data(v.data());
            d
        });
    }};
}

macro_rules! facades {
    ($runtime:expr, $dtype:ty, $name:expr) => {{
        type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        facade!(
            $runtime,
            format!("{} facade u1", $name),
            $dtype,
            U1FusionRule,
            (-3..=3).map(U1Irrep::new).collect::<Vec<_>>()
        );
        facade!(
            $runtime,
            format!("{} facade su2", $name),
            $dtype,
            SU2FusionRule,
            (0..4).map(SU2Irrep::from_twice_spin).collect::<Vec<_>>()
        );
        facade!(
            $runtime,
            format!("{} facade fz2xu1", $name),
            $dtype,
            Fz2U1::new(FermionParityFusionRule::new(), U1FusionRule),
            (-2..=3)
                .map(|q: i32| {
                    let parity = if q.rem_euclid(2) == 0 {
                        FermionParityIrrep::EVEN
                    } else {
                        FermionParityIrrep::ODD
                    };
                    ProductSector::new(parity, U1Irrep::new(q))
                })
                .collect::<Vec<_>>()
        );
    }};
}

fn main() {
    let threads = std::env::args().nth(1).map(|t| t.parse::<usize>().unwrap());
    sites::<f64>(threads, "f64");
    sites::<Complex64>(threads, "c64");
    let mut builder = Runtime::builder();
    if let Some(threads) = threads {
        builder = builder.dense_threads(threads);
    }
    let runtime = builder.build().unwrap();
    facades!(&runtime, f64, "f64");
    facades!(&runtime, Complex64, "c64");
}
