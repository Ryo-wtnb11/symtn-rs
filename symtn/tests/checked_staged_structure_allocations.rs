//! #2050: a pre-commit plan compiled against a staged checked destination
//! lends the live canonical structure on a complete-cache hit instead of
//! wrapping a cloned preview in a fresh `Arc`. Warm allocation calls of the
//! public checked permute and contraction (non-identity, so a destination is
//! staged) are pinned as upper bounds. #2129: the contraction's core plan
//! compiles from coupled regions; #2063: its staged and core intermediates
//! are committed and its transformers reused, so warm planning is
//! O(S + G + P_in·r) and its warm calls are bounded by `c·(S + G)`.

use std::sync::Arc;

use symtn::sector::TypedSectorAdmission;
use symtn::typed::{
    ContractSpec, GradedSpace, Runtime, TensorMap, TypedTensorConstructionDispatch,
    TypedTensorContractDispatch, TypedTensorModeDispatch, TypedTensorTransformDispatch,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[path = "../../tests/support/toy_rules.rs"]
mod toy_rules;

use toy_rules::{GenericLabel, GenericToy};

/// The dispatch a provider needs for the fixture below.
trait Mode<R: TypedSectorAdmission>:
    TypedTensorModeDispatch<R>
    + TypedTensorConstructionDispatch<R, f64>
    + TypedTensorContractDispatch<R, f64>
    + TypedTensorTransformDispatch<R, f64>
{
}

impl<R: TypedSectorAdmission, M> Mode<R> for M where
    M: TypedTensorModeDispatch<R>
        + TypedTensorConstructionDispatch<R, f64>
        + TypedTensorContractDispatch<R, f64>
        + TypedTensorTransformDispatch<R, f64>
{
}

fn tensor<R: TypedSectorAdmission>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    rank: usize,
) -> TensorMap<R, f64>
where
    R::Mode: Mode<R>,
{
    let legs = vec![leg; rank / 2];
    TensorMap::from_subblock_fn(runtime, legs.clone(), legs, |trees, indices| {
        trees.codomain_vertices().first().map_or(0, |v| v.get()) as f64
            + indices.iter().sum::<usize>() as f64
    })
    .unwrap()
}

/// Warm allocation calls and bytes of one public checked call.
struct Warm {
    permute: (u64, u64),
    /// Recoupling groups `G_t` of the permute: composed-coefficient lookups
    /// of its cold call, one per source fusion-tree group (at least its
    /// multi-term groups).
    permute_groups: u64,
    contract: (u64, u64),
    /// Output blocks `P` of the contraction.
    out_blocks: usize,
    /// Blocks `P_c` and coupled sectors `S` of its core product.
    core_blocks: usize,
    core_sectors: usize,
    /// Recoupling groups `G`: composed-coefficient lookups of the cold call
    /// (a warm call makes none).
    groups: u64,
    /// Input blocks `P_in` of one operand.
    in_blocks: usize,
    /// Completed-transformer hits of one warm call.
    transformer_hits: u64,
}

fn lookups(kind: symtn::cache::StructureCacheKind) -> u64 {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .map(|info| info.hits() + info.misses())
        .unwrap()
}

fn hits(kind: symtn::cache::StructureCacheKind) -> u64 {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .map(|info| info.hits())
        .unwrap()
}

fn warm<R: TypedSectorAdmission>(leg: &GradedSpace<R>, rank: usize) -> Warm
where
    R::Mode: Mode<R>,
{
    let _serial = counting_alloc::serial();
    symtn::cache::clear();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = leg.clone();
    let lhs = tensor(&runtime, &leg, rank);
    let rhs = tensor(&runtime, &leg, rank);
    let half = rank / 2;
    let reversed: Vec<usize> = (0..rank).rev().collect();
    let (codomain, domain) = reversed.split_at(half);

    let open = 2 * rank - 2;
    let codomain_open: Vec<usize> = (0..open / 2).rev().collect();
    let domain_open: Vec<usize> = (open / 2..open).collect();
    let (lhs_axes, rhs_axes) = ([rank - 1], [0]);
    let spec = ContractSpec {
        lhs: &lhs_axes,
        rhs: &rhs_axes,
        codomain: &codomain_open,
        domain: &domain_open,
    };

    use symtn::cache::StructureCacheKind::{CompletedTreeTransformer, TreeTransformCoefficients};
    let fresh = lookups(TreeTransformCoefficients);
    drop(lhs.permute(codomain, domain).unwrap());
    let cold = lookups(TreeTransformCoefficients);
    let permute_groups = cold - fresh;
    let out_blocks = lhs.contract(&rhs, &spec).unwrap().subblock_count();
    let groups = lookups(TreeTransformCoefficients) - cold;
    let (out, permute) = counting_alloc::measure(|| lhs.permute(codomain, domain).unwrap());
    drop(out);
    let warm_lookups = lookups(TreeTransformCoefficients);
    let warm_hits = hits(CompletedTreeTransformer);
    let (out, contract) = counting_alloc::measure(|| lhs.contract(&rhs, &spec).unwrap());
    assert_eq!(lookups(TreeTransformCoefficients), warm_lookups);
    let transformer_hits = hits(CompletedTreeTransformer) - warm_hits;
    drop(out);
    let lhs_core: Vec<usize> = (0..rank - 1).collect();
    let rhs_core: Vec<usize> = (1..rank).collect();
    let core = lhs
        .permute(&lhs_core, &[rank - 1])
        .unwrap()
        .compose(&rhs.permute(&[0], &rhs_core).unwrap())
        .unwrap();
    Warm {
        permute: (permute.calls, permute.bytes),
        permute_groups,
        contract: (contract.calls, contract.bytes),
        out_blocks,
        core_blocks: core.subblock_count(),
        core_sectors: core.blocks().unwrap().len(),
        groups,
        in_blocks: lhs.subblock_count(),
        transformer_hits,
    }
}

impl Warm {
    fn describe(&self, case: &str, rank: usize) -> String {
        format!(
            "{case} rank {rank}: permute {:?} G_t={} contract {:?} P={} P_c={} P_in={} S={} G={} \
             cache-3 hits={}",
            self.permute,
            self.permute_groups,
            self.contract,
            self.out_blocks,
            self.core_blocks,
            self.in_blocks,
            self.core_sectors,
            self.groups,
            self.transformer_hits
        )
    }

    /// #2154: a warm permute hits its source's destination memo and its
    /// completed transformer, so it re-admits nothing and derives nothing:
    /// `calls ≤ 2·G_t + 8`, independent of `P_in` and the source sectors.
    /// The `G_t` term is the dense backend's per-recoupling-group scratch
    /// (Tenferro); the constant is the owned output, the committed space and
    /// the preview keys.
    fn assert_permute_bound(&self, what: &str) {
        assert!(self.permute.0 <= 2 * self.permute_groups + 8, "{what}");
    }

    /// #2063: a warm contraction binds its completed transformers and
    /// allocates `calls ≤ 2·(S + G) + 32`. The `G` term is the dense
    /// backend's per-recoupling-GEMM scratch (out of scope, Tenferro); the
    /// constant is the owned output and payload buffers and per-call keys.
    /// The per-output-block relation is the anti-regression pin.
    fn assert_contract_bounds(&self, what: &str) {
        // Rank 2 transforms nothing (`G = 0`); otherwise all three stages do.
        let transforms = if self.groups == 0 { 0 } else { 3 };
        assert_eq!(self.transformer_hits, transforms, "{what}");
        let groups = self.core_sectors as u64 + self.groups;
        assert!(self.contract.0 <= 2 * groups + 32, "{what}");
        assert!(self.contract.0 <= 3 * self.out_blocks as u64 + 64, "{what}");
    }
}

/// Permute: a warm permute does not re-admit its source (#2154). Contract:
/// the region core plan (#2129) and committed intermediates (#2063) keep the
/// count linear in sectors and groups. Bounds, not
/// equalities: platform allocators and the toy provider may shift
/// incidental sizes.
#[test]
fn warm_staged_checked_plans_lend_the_canonical_structure() {
    let leg = GradedSpace::try_new(
        Arc::new(GenericToy),
        [(GenericLabel::Vacuum, 1), (GenericLabel::X, 2)],
    )
    .unwrap();
    for rank in [2, 4, 6] {
        let w = warm(&leg, rank);
        let what = w.describe("GenericToy", rank);
        eprintln!("{what}");
        w.assert_permute_bound(&what);
        w.assert_contract_bounds(&what);
    }
}

/// The same permute and contraction bounds for the fusion-symbols SU(N) providers
/// (SU(2) and SU(3), whose adjoint carries outer multiplicity). SU(3) stops
/// at rank 4: its rank-6 cold build exceeds ten minutes in a debug test.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_racah_checked_contraction_is_bounded_by_sectors_and_groups() {
    use symtn::sector::LieGroupFusionRule;
    let su2 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap());
    let su2 = GradedSpace::try_new(su2, [(vec![0i64], 1), (vec![1], 2)]).unwrap();
    let su3 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let su3 = GradedSpace::try_new(su3, [(vec![0i64, 0], 1), (vec![1, 1], 1)]).unwrap();
    for (case, leg, ranks) in [
        ("SU(2)", &su2, &[2usize, 4, 6][..]),
        ("SU(3)", &su3, &[2, 4]),
    ] {
        for &rank in ranks {
            let w = warm(leg, rank);
            let what = w.describe(case, rank);
            eprintln!("{what}");
            w.assert_permute_bound(&what);
            w.assert_contract_bounds(&what);
        }
    }
}
