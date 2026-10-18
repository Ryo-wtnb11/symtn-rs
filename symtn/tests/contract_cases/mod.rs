//! General-axes contraction fixtures and their independent oracles, shared by
//! the Host pin (`typed_contract_host_oracle.rs`, ungated) and the device gate
//! (`typed_cuda_contract.rs`, #1345).
//!
//! Two oracles, neither of which passes through the Host `DynamicTree`
//! compiler whose device replay is under test:
//!
//! * [`blas_contract_oracle`] — TensorKit's `blas_contract!` step sequence
//!   (tensoroperations.jl:383-455 @cfaa073) on Host typed operations: permute
//!   A to `(open; contracted)`, permute B to `(contracted; open)`, canonical
//!   `compose` (TensorKit `mul!`), permute the result by `pAB`. Bosonic
//!   providers only: TensorKit inserts a `twist!` for fermions here.
//! * [`dense_oracle`] — the physical-basis expansion (U(1), SU(2)): an index
//!   sum over the contracted legs followed by an axis permutation, for the
//!   cases where A contracts only domain legs and B only codomain legs, so no
//!   leg bends and the physical array only moves.
//!
//! Fixture entries are dyadic rationals, exact in `f32`. The payload set is
//! the shared `common::Payload` harness, so every including test binary also
//! declares `mod common;`.

#![allow(dead_code)]

use std::sync::Arc;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, PhysicalFusionBasis, SectorCodec,
};
use symtn::typed::{BlockFusionTrees, ContractSpec, GradedSpace, Runtime, TensorMap};

pub use crate::common::Payload;

/// Dyadic counting fill, distinct per `salt`.
pub fn fill<S, D: Payload>(salt: usize) -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> D {
    let mut next = salt;
    move |_, _| {
        next += 1;
        let real = ((next * 37) % 17) as f64 / 8.0 - 1.0;
        let imaginary = ((next * 11) % 13) as f64 / 16.0 - 0.375;
        D::entry(real, imaginary)
    }
}

pub fn u1(charges: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        charges
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

pub fn su2() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

pub type U1Su2 = ProductFusionRule<U1FusionRule, SU2FusionRule>;

pub fn u1_su2() -> GradedSpace<U1Su2> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule.product(SU2FusionRule)),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                2,
            ),
        ],
    )
    .unwrap()
}

pub type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
pub type FermionSu2 = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

/// fZ2 x U(1): odd sectors of both charge signs, degeneracy > 1.
pub fn fermion_u1() -> GradedSpace<FermionU1> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(1)), 1),
        ],
    )
    .unwrap()
}

/// fZ2 (x) SU(2): fermionic signs over non-Abelian recoupling.
pub fn fermion_su2() -> GradedSpace<FermionSu2> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(SU2FusionRule)),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                2,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap()
}

/// One contraction: operands, axes, and whether the physical-basis oracle
/// applies (A contracts only domain legs, B only codomain legs).
pub struct Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    pub name: &'static str,
    pub lhs: TensorMap<R, D>,
    pub rhs: TensorMap<R, D>,
    pub lhs_axes: Vec<usize>,
    pub rhs_axes: Vec<usize>,
    pub output_axes: Vec<usize>,
    pub dense: bool,
}

impl<R, D> Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    /// The longest sum an output entry combines: the contracted physical
    /// dimension (a bound, not the reduced count).
    pub fn terms(&self) -> usize {
        self.lhs_axes.len().max(1) * 64
    }

    /// The spec with TensorOperations' default split: every open leg of
    /// `lhs` in the codomain.
    pub fn spec(&self) -> ContractSpec<'_> {
        let (codomain, domain) = self
            .output_axes
            .split_at(self.lhs.rank() - self.lhs_axes.len());
        ContractSpec {
            lhs: &self.lhs_axes,
            rhs: &self.rhs_axes,
            codomain,
            domain,
        }
    }

    pub fn host(&self) -> TensorMap<R, D> {
        self.lhs.contract(&self.rhs, &self.spec()).unwrap()
    }
}

fn tensor<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
    salt: usize,
) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    TensorMap::from_subblock_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        fill(salt),
    )
    .unwrap()
}

/// Rank 5 against rank 4: a codomain leg against a domain leg and a domain
/// leg against a codomain leg, contracted out of order, output permuted
/// across both sources.
pub fn u1_rank_five<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    Case {
        name: "U(1) rank 5 x 4, mixed sides",
        lhs: tensor(runtime, &[&v, &v, &v], &[&v, &v], 1),
        rhs: tensor(runtime, &[&v, &v], &[&v, &v], 2),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
        dense: false,
    }
}

/// Whole domain against whole codomain in reversed order, codomain output
/// permuted; `v` carries charge 2 that `w` lacks, so a coupled sector of the
/// result has no contributing GEMM.
pub fn u1_reordered<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(0, 1), (1, 2), (2, 1)]);
    let w = u1(&[(-1, 1), (0, 2), (1, 2)]);
    Case {
        name: "U(1) reordered whole-side",
        lhs: tensor(runtime, &[&v, &v], &[&w, &w], 3),
        rhs: tensor(runtime, &[&w, &w], &[&v], 4),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
        dense: true,
    }
}

/// SU(2) recoupling of the reordered whole-side contraction.
pub fn su2_reordered<D: Payload>(runtime: &Runtime) -> Case<SU2FusionRule, D> {
    let s = su2();
    Case {
        name: "SU(2) reordered whole-side",
        lhs: tensor(runtime, &[&s, &s], &[&s, &s], 5),
        rhs: tensor(runtime, &[&s, &s], &[&s], 6),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
        dense: true,
    }
}

/// SU(2): a codomain leg of A against a domain leg of B.
pub fn su2_bent<D: Payload>(runtime: &Runtime) -> Case<SU2FusionRule, D> {
    let s = su2();
    Case {
        name: "SU(2) codomain against domain",
        lhs: tensor(runtime, &[&s, &s], &[&s], 7),
        rhs: tensor(runtime, &[&s], &[&s, &s], 8),
        lhs_axes: vec![0],
        rhs_axes: vec![2],
        output_axes: vec![3, 0, 2, 1],
        dense: false,
    }
}

/// Bosonic product provider, two pairs across sides.
pub fn product_general<D: Payload>(runtime: &Runtime) -> Case<U1Su2, D> {
    let p = u1_su2();
    Case {
        name: "U(1) x SU(2) mixed sides",
        lhs: tensor(runtime, &[&p, &p], &[&p], 9),
        rhs: tensor(runtime, &[&p, &p], &[&p], 10),
        lhs_axes: vec![2, 1],
        rhs_axes: vec![0, 2],
        output_axes: vec![1, 0],
        dense: false,
    }
}

/// A already `(open; contracted)`: its source transform is the identity and
/// it can be read in place; B needs one.
pub fn u1_lhs_identity<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 2), (1, 1)]);
    Case {
        name: "U(1) identity lhs transform",
        lhs: tensor(runtime, &[&v, &v], &[&w], 11),
        rhs: tensor(runtime, &[&v, &w], &[&v], 12),
        lhs_axes: vec![2],
        rhs_axes: vec![1],
        output_axes: vec![0, 1, 2, 3],
        dense: false,
    }
}

/// B already `(contracted; open)`: its source transform is the identity.
pub fn u1_rhs_identity<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 2), (1, 1)]);
    let w_dual = w.try_dual().unwrap();
    Case {
        name: "U(1) identity rhs transform",
        lhs: tensor(runtime, &[&v, &w], &[&v], 13),
        rhs: tensor(runtime, &[&w_dual], &[&v, &v], 14),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1, 2, 3],
        dense: false,
    }
}

/// U(1) with `-1` and `1` at different degeneracies: `V != V*`.
pub fn u1_non_self_dual() -> GradedSpace<U1FusionRule> {
    u1(&[(-1, 2), (0, 1), (1, 3)])
}

/// The contraction-candidate audit probes (#1468) on `V⊗V ← V⊗V` operands,
/// each paired with its literal Core control's name. They are free in
/// TensorKit only after the paired-axis sort or the operand swap; `P'` is a
/// lazy adjoint:
///
/// - C0 `A[2,3]·B[0,1]` (literal control), C1 `A[3,2]·B[1,0]`,
///   C2 `A[0,1]·B[2,3] → [2,3,0,1]`;
/// - L4 `P'[2,3]·B[0,1]` (literal control), L3 `P'[3,2]·B[1,0]`,
///   L5 `A[0,1]·P'[2,3] → [2,3,0,1]`, L7 `A[3,2]·P'[1,0]`.
pub fn candidate_core_probes<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
) -> Vec<(Case<R, D>, &'static str)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let a: TensorMap<R, D> = tensor(runtime, &[v, v], &[v, v], 61);
    let b: TensorMap<R, D> = tensor(runtime, &[v, v], &[v, v], 62);
    let lazy = tensor::<R, D>(runtime, &[v, v], &[v, v], 63)
        .adjoint()
        .unwrap();
    let case = |name,
                lhs: &TensorMap<R, D>,
                rhs: &TensorMap<R, D>,
                l: [usize; 2],
                r: [usize; 2],
                swap: bool| Case {
        name,
        lhs: lhs.clone(),
        rhs: rhs.clone(),
        lhs_axes: l.to_vec(),
        rhs_axes: r.to_vec(),
        output_axes: if swap {
            vec![2, 3, 0, 1]
        } else {
            vec![0, 1, 2, 3]
        },
        dense: false,
    };
    vec![
        (case("C0", &a, &b, [2, 3], [0, 1], false), "C0"),
        (case("C1", &a, &b, [3, 2], [1, 0], false), "C0"),
        (case("C2", &a, &b, [0, 1], [2, 3], true), "C0"),
        (case("L4", &lazy, &b, [2, 3], [0, 1], false), "L4"),
        (case("L3", &lazy, &b, [3, 2], [1, 0], false), "L4"),
        (case("L5", &a, &lazy, [0, 1], [2, 3], true), "L4"),
        (case("L7", &a, &lazy, [3, 2], [1, 0], false), "L4"),
    ]
}

/// Zero-copy candidates under a requested output order that is neither the
/// identity nor `pAB′` (#1475): TensorKit `blas_contract!` then copies only
/// C (`copyC`: `mul!` into a temporary, then a permuting `tensoradd!`), and
/// `has_shared_permute(::AdjointTensorMap)` keeps `P'` free. Each probe is
/// paired with whether that zero-copy candidate is the swap `B·A`:
///
/// - `L3p` `P'[3,2]·B[1,0] → [1,0,2,3]` (the issue's example, sort);
/// - `L5p` `A[0,1]·P'[2,3] → [3,2,0,1]` (swap);
/// - `L7p` `A[3,2]·P'[1,0] → [0,1,3,2]` (sort, lazy rhs);
/// - `C1p` `A[3,2]·B[1,0] → [1,0,3,2]`, `C2p` `A[0,1]·B[2,3] → [3,2,1,0]`
///   (owned sort and swap).
pub fn copy_c_probes<R, D>(runtime: &Runtime, v: &GradedSpace<R>) -> Vec<(Case<R, D>, bool)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let tensor = |salt| tensor::<R, D>(runtime, &[v, v], &[v, v], salt);
    let (a, b, lazy) = (tensor(81), tensor(82), tensor(83).adjoint().unwrap());
    let case = |name,
                lhs: &TensorMap<R, D>,
                rhs: &TensorMap<R, D>,
                l: [usize; 2],
                r: [usize; 2],
                out: [usize; 4]| Case {
        name,
        lhs: lhs.clone(),
        rhs: rhs.clone(),
        lhs_axes: l.to_vec(),
        rhs_axes: r.to_vec(),
        output_axes: out.to_vec(),
        dense: l == [3, 2],
    };
    vec![
        (case("L3p", &lazy, &b, [3, 2], [1, 0], [1, 0, 2, 3]), false),
        (case("L5p", &a, &lazy, [0, 1], [2, 3], [3, 2, 0, 1]), true),
        (case("L7p", &a, &lazy, [3, 2], [1, 0], [0, 1, 3, 2]), false),
        (case("C1p", &a, &b, [3, 2], [1, 0], [1, 0, 3, 2]), false),
        (case("C2p", &a, &b, [0, 1], [2, 3], [3, 2, 1, 0]), true),
    ]
}

/// A destination of `case`'s result space holding NaN everywhere: a retained
/// buffer `contract_into` must rewrite element by element.
pub fn poisoned_destination<R, D>(case: &Case<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    case.host().scale(D::entry(f64::NAN, f64::NAN)).unwrap()
}

/// Destinations with a block no GEMM writes (`v` carries charge 2 that `w`
/// lacks): the canonical core route, a transformed lhs with an identity
/// output (the core GEMMs write the destination directly in both), and an
/// output transform over the same inactive core block. The resolution of
/// each geometry is pinned in
/// `symtn-tensors/src/contract/storage_contract_tests.rs::overwrite_cases`.
pub fn u1_inactive_cases<D: Payload>(runtime: &Runtime) -> [Case<U1FusionRule, D>; 3] {
    let v = u1(&[(0, 1), (1, 2), (2, 1)]);
    let w = u1(&[(0, 2), (1, 1)]);
    [
        Case {
            name: "U(1) core route, inactive block",
            lhs: tensor(runtime, &[&v, &v], &[&w], 51),
            rhs: tensor(runtime, &[&w], &[&v], 52),
            lhs_axes: vec![2],
            rhs_axes: vec![0],
            output_axes: vec![0, 1, 2],
            dense: false,
        },
        Case {
            name: "U(1) transformed lhs, identity output, inactive block",
            lhs: tensor(runtime, &[&v], &[&w, &v], 53),
            rhs: tensor(runtime, &[&w], &[&v], 54),
            lhs_axes: vec![1],
            rhs_axes: vec![0],
            output_axes: vec![0, 1, 2],
            dense: false,
        },
        Case {
            name: "U(1) output transform over an inactive core block",
            lhs: tensor(runtime, &[&v, &v], &[&w], 55),
            rhs: tensor(runtime, &[&w], &[&v], 56),
            lhs_axes: vec![2],
            rhs_axes: vec![0],
            output_axes: vec![1, 0, 2],
            dense: false,
        },
    ]
}

/// `L^H` contracted with `L`, and `L` with `L^H`, on a subset of legs in a
/// non-canonical order. Every leg of `L^H` is the dual of the matching leg
/// of `L`, so the pairing is valid for any provider; `L^H` stays a lazy
/// adjoint over `L`'s payload.
pub fn lazy_cases<R, D>(lhs: &TensorMap<R, D>, name: &'static str) -> [Case<R, D>; 2]
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let nout = lhs.codomain_rank();
    let nin = lhs.domain_rank();
    assert!(nout >= 2 && nin >= 1, "{name}: fixture too small");
    // `L^H` axis `nin + j` is `L` codomain leg `j`; `L^H` axis `i < nin` is
    // `L` domain leg `nout + i`.
    let adjoint_axes = vec![nin + 1, 0];
    let direct_axes = vec![1, nout];
    let output: Vec<usize> = (0..lhs.rank() * 2 - 4).rev().collect();
    let adjoint = lhs.adjoint().unwrap();
    [
        Case {
            name,
            lhs: adjoint.clone(),
            rhs: lhs.clone(),
            lhs_axes: adjoint_axes.clone(),
            rhs_axes: direct_axes.clone(),
            output_axes: output.clone(),
            dense: false,
        },
        Case {
            name,
            lhs: lhs.clone(),
            rhs: adjoint,
            lhs_axes: direct_axes,
            rhs_axes: adjoint_axes,
            output_axes: output,
            dense: false,
        },
    ]
}

/// #2147: a lazy adjoint `a = t'` (`t: [V, V] <- [V]`) that is already a core
/// operand (identity tree transform) against a partner that must be permuted,
/// so the route is DynamicTree with the adjoint borrowed and only the partner
/// copied: `a` over its whole domain against `u = s.permute([0], [1, 2])`, and
/// `w = s.permute([2, 0], [1])`'s leg 0 against `a`'s codomain; and (#2159,
/// candidate selection) `a`'s domain against `r: [V, V] <- []`'s codomain in
/// reverse order. Returns each case with whether its lazy operand is the lhs.
pub fn identity_adjoint_cases<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
) -> [(Case<R, D>, bool); 3]
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let t: TensorMap<R, D> = tensor(runtime, &[v, v], &[v], 61);
    let a = t.adjoint().unwrap();
    let u = tensor::<R, D>(runtime, &[v, v], &[v], 62)
        .permute(&[0], &[1, 2])
        .unwrap();
    let w = tensor::<R, D>(runtime, &[v, v], &[v], 63)
        .permute(&[2, 0], &[1])
        .unwrap();
    // #2159: `a`'s domain against `r`'s codomain in reverse order. Keeping
    // `a`'s order permutes only `r` (|r| < |t|), keeping `r`'s permutes `a`:
    // TensorKit `contract_memcost` takes the former.
    let r = tensor::<R, D>(runtime, &[v, v], &[], 64);
    [
        (
            Case {
                name: "identity adjoint lhs, permuted rhs",
                lhs: a.clone(),
                rhs: u,
                lhs_axes: vec![1, 2],
                rhs_axes: vec![0, 1],
                output_axes: vec![0, 1],
                dense: false,
            },
            true,
        ),
        (
            Case {
                name: "identity adjoint lhs, reversed rhs order",
                lhs: a.clone(),
                rhs: r,
                lhs_axes: vec![1, 2],
                rhs_axes: vec![1, 0],
                output_axes: vec![0],
                dense: false,
            },
            true,
        ),
        (
            Case {
                name: "permuted lhs, identity adjoint rhs",
                lhs: w,
                rhs: a,
                lhs_axes: vec![0],
                rhs_axes: vec![0],
                output_axes: vec![0, 1, 2, 3],
                dense: false,
            },
            false,
        ),
    ]
}

/// A lazy adjoint over SU(2) (every sector self-dual), contracted in
/// core-form source order (lhs whole domain, rhs whole codomain, in order),
/// with a non-identity output. Lazy lhs, then lazy rhs; multi-block,
/// degeneracy > 1. Typed Host eager and the device both take TensorKit's
/// `copyC` here: the zero-copy core into a temporary, then one permute.
/// Without the `copyC` predicate the context's prelowered entry resolves this
/// geometry to its dense `Structure` route (pinned in
/// `symtn-tensors/src/contract/storage_contract_tests.rs`).
pub fn su2_core_form_lazy_cases<D: Payload>(runtime: &Runtime) -> [Case<SU2FusionRule, D>; 2] {
    let s = su2();
    let x: TensorMap<_, D> = tensor(runtime, &[&s, &s], &[&s, &s], 15);
    let y: TensorMap<_, D> = tensor(runtime, &[&s], &[&s, &s], 17);
    [
        Case {
            name: "SU(2) core-form lazy lhs (copyC)",
            lhs: x.adjoint().unwrap(),
            rhs: tensor(runtime, &[&s, &s], &[&s], 16),
            lhs_axes: vec![2, 3],
            rhs_axes: vec![0, 1],
            output_axes: vec![2, 0, 1],
            dense: false,
        },
        Case {
            name: "SU(2) core-form lazy rhs (copyC)",
            lhs: tensor(runtime, &[&s], &[&s, &s], 18),
            rhs: y.adjoint().unwrap(),
            lhs_axes: vec![1, 2],
            rhs_axes: vec![0, 1],
            output_axes: vec![1, 0],
            dense: false,
        },
    ]
}

/// Rank 4 against rank 4 over mixed sides, out of order. B's codomain leg 1
/// is `v*` (a dual contracted leg on B's codomain side) and B's domain leg 3
/// is `u`, which reaches the permuted B's codomain as `u*`: dual — a dual
/// contracted leg from B's domain side — unless `u` is itself dual. With both
/// twisted, θ of a core-right block is the parity of its coupled sector
/// (uniform per coupled sector); with `u_dual` only `v*` is twisted, so
/// the row trees of one coupled sector carry both signs.
pub fn fermionic_general<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    u_dual: bool,
    name: &'static str,
    salt: usize,
) -> Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let v_dual = v.try_dual().unwrap();
    let u = if u_dual { v_dual.clone() } else { v.clone() };
    Case {
        name,
        lhs: tensor(runtime, &[v, &u], &[v, v], salt),
        rhs: tensor(runtime, &[v, &v_dual], &[v, &u], salt + 1),
        lhs_axes: vec![1, 0],
        rhs_axes: vec![3, 1],
        output_axes: vec![2, 0, 3, 1],
        dense: false,
    }
}

/// The canonical `mul!` form (A's whole domain against B's whole codomain,
/// in order, identity output) with B's codomain `(v, v*)`: only the second
/// leg is twisted, so θ varies within one coupled-sector matrix and no
/// per-job GEMM alpha expresses it.
pub fn fermionic_canonical_nonuniform<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    name: &'static str,
    salt: usize,
) -> Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let v_dual = v.try_dual().unwrap();
    Case {
        name,
        lhs: tensor(runtime, &[v, v], &[v, &v_dual], salt),
        rhs: tensor(runtime, &[v, &v_dual], &[v], salt + 1),
        lhs_axes: vec![2, 3],
        rhs_axes: vec![0, 1],
        output_axes: vec![0, 1, 2],
        dense: false,
    }
}

/// The `mul!` form with B's whole codomain `(v*, v*)` contracted: θ of a
/// core-right block is the product of its two uncoupled parities, the parity
/// of its coupled sector, so it is uniform within each coupled-sector matrix
/// and the core carries it as per-job GEMM alpha (#1858), where TensorKit's
/// `blas_contract!` copies an operand to twist it.
pub fn fermionic_canonical_uniform<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    name: &'static str,
    salt: usize,
) -> Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let v_dual = v.try_dual().unwrap();
    Case {
        name,
        lhs: tensor(runtime, &[v, v], &[&v_dual, &v_dual], salt),
        rhs: tensor(runtime, &[&v_dual, &v_dual], &[v], salt + 1),
        lhs_axes: vec![2, 3],
        rhs_axes: vec![0, 1],
        output_axes: vec![0, 1, 2],
        dense: false,
    }
}

/// One fixture per operand-copy case of TensorKit's `blas_contract!` twist
/// choice (tensoroperations.jl:398-409 @cfaa073), each with a dual leg among
/// B's contracted legs, and A smaller than B where size decides; the cases
/// name the `LhsRhs` layout:
///
/// - `a_copied`: A's contracted leg is in its codomain (A is permuted), B is
///   already `[v*] <- [..]` (read in place): the twist goes on A.
/// - `canonical`: both already in `mul!` form: the smaller, A, is copied.
/// - `b_copied`: A already in form, B's contracted leg is in its domain.
/// - `both_copied`: both permuted: the smaller, A, is twisted.
///
/// Returned in that order under `names`.
pub fn fermionic_twist_roles<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    names: [&'static str; 4],
    salt: usize,
) -> [Case<R, D>; 4]
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let v_dual = v.try_dual().unwrap();
    [
        Case {
            name: names[0],
            lhs: tensor(runtime, &[v, v], &[v], salt),
            rhs: tensor(runtime, &[&v_dual], &[v, v], salt + 1),
            lhs_axes: vec![1],
            rhs_axes: vec![0],
            output_axes: vec![0, 1, 2, 3],
            dense: false,
        },
        Case {
            name: names[1],
            lhs: tensor(runtime, &[v], &[v, &v_dual], salt + 2),
            rhs: tensor(runtime, &[v, &v_dual], &[v, v], salt + 3),
            lhs_axes: vec![1, 2],
            rhs_axes: vec![0, 1],
            output_axes: vec![0, 1, 2],
            dense: false,
        },
        Case {
            name: names[2],
            lhs: tensor(runtime, &[v, v], &[&v_dual], salt + 4),
            rhs: tensor(runtime, &[v], &[v, v], salt + 5),
            lhs_axes: vec![2],
            rhs_axes: vec![1],
            output_axes: vec![0, 1, 2, 3],
            dense: false,
        },
        Case {
            name: names[3],
            lhs: tensor(runtime, &[v, v], &[v], salt + 6),
            rhs: tensor(runtime, &[v, &v_dual], &[v, v], salt + 7),
            lhs_axes: vec![1, 0],
            rhs_axes: vec![3, 1],
            output_axes: vec![0, 1, 2],
            dense: false,
        },
    ]
}

/// Which operand TensorKit's `blas_contract!` twists for a fermionic
/// contraction (tensoroperations.jl:398-431 @cfaa073).
#[derive(Clone, Copy, Debug)]
pub enum TwistRole {
    /// No twist: the bosonic sequence, or the twist-free control.
    None,
    /// `twist!(Anew, filter(!isdual ∘ space(Anew), domainind(Anew)))`.
    A,
    /// `twist!(Bnew, filter(isdual ∘ space(Bnew), codomainind(Bnew)))`.
    B,
}

/// TensorKit `blas_contract!` on Host typed operations, with the fermionic
/// twist placed on `role`. TensorKit's `space(Anew, i)` of a domain index is
/// the dual of the stored domain leg, so `!isdual(space(Anew, i))` is
/// `Anew.domain()[j].is_dual()`.
///
/// `twist` is the typed `TensorMap::twist`, passed in because its dispatch
/// bound is per provider mode.
pub fn fermionic_blas_contract_oracle<R, D>(
    case: &Case<R, D>,
    role: TwistRole,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D>,
) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let split = case.lhs.rank() - case.lhs_axes.len();
    fermionic_blas_contract_oracle_partitioned(case, role, split, twist)
}

pub fn fermionic_blas_contract_oracle_partitioned<R, D>(
    case: &Case<R, D>,
    role: TwistRole,
    split: usize,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D>,
) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let open = |rank: usize, contracted: &[usize]| -> Vec<usize> {
        (0..rank)
            .filter(|axis| !contracted.contains(axis))
            .collect()
    };
    let lhs_open = open(case.lhs.rank(), &case.lhs_axes);
    let rhs_open = open(case.rhs.rank(), &case.rhs_axes);
    let mut a = case.lhs.permute(&lhs_open, &case.lhs_axes).unwrap();
    let mut b = case.rhs.permute(&case.rhs_axes, &rhs_open).unwrap();
    match role {
        TwistRole::None => {}
        TwistRole::A => {
            let nout = a.codomain_rank();
            let legs: Vec<usize> = a
                .domain()
                .iter()
                .enumerate()
                .filter(|(_, leg)| leg.is_dual())
                .map(|(j, _)| nout + j)
                .collect();
            a = twist(&a, &legs);
        }
        TwistRole::B => {
            let legs: Vec<usize> = b
                .codomain()
                .iter()
                .enumerate()
                .filter(|(_, leg)| leg.is_dual())
                .map(|(i, _)| i)
                .collect();
            b = twist(&b, &legs);
        }
    }
    let c = a.compose(&b).unwrap();
    c.permute(&case.output_axes[..split], &case.output_axes[split..])
        .unwrap()
}

/// TensorKit `blas_contract!` on Host typed operations.
pub fn blas_contract_oracle<R, D>(case: &Case<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let open = |rank: usize, contracted: &[usize]| -> Vec<usize> {
        (0..rank)
            .filter(|axis| !contracted.contains(axis))
            .collect()
    };
    let lhs_open = open(case.lhs.rank(), &case.lhs_axes);
    let rhs_open = open(case.rhs.rank(), &case.rhs_axes);
    let a = case.lhs.permute(&lhs_open, &case.lhs_axes).unwrap();
    let b = case.rhs.permute(&case.rhs_axes, &rhs_open).unwrap();
    let c = a.compose(&b).unwrap();
    let split = lhs_open.len();
    c.permute(&case.output_axes[..split], &case.output_axes[split..])
        .unwrap()
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut running = 1;
    for &dim in shape {
        strides.push(running);
        running *= dim;
    }
    strides
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    let total: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        visit(&index);
        for (axis, value) in index.iter_mut().enumerate() {
            *value += 1;
            if *value < shape[axis] {
                break;
            }
            *value = 0;
        }
    }
}

/// Physical-basis contraction: `C[out] = sum_k A[.., k, ..] B[.., k, ..]`,
/// output axes permuted by `output_axes`, all arrays column-major.
pub fn dense_oracle<R, D>(case: &Case<R, D>) -> (Vec<usize>, Vec<D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    let a = case.lhs.to_physical_dense().unwrap();
    let b = case.rhs.to_physical_dense().unwrap();
    let a_strides = column_major_strides(&a.shape);
    let b_strides = column_major_strides(&b.shape);
    let a_open: Vec<usize> = (0..a.shape.len())
        .filter(|axis| !case.lhs_axes.contains(axis))
        .collect();
    let b_open: Vec<usize> = (0..b.shape.len())
        .filter(|axis| !case.rhs_axes.contains(axis))
        .collect();
    let joint_shape: Vec<usize> = a_open
        .iter()
        .map(|&axis| a.shape[axis])
        .chain(b_open.iter().map(|&axis| b.shape[axis]))
        .collect();
    let contracted_shape: Vec<usize> = case.lhs_axes.iter().map(|&axis| a.shape[axis]).collect();
    let output_shape: Vec<usize> = case
        .output_axes
        .iter()
        .map(|&axis| joint_shape[axis])
        .collect();
    let output_strides = column_major_strides(&output_shape);
    let mut output = vec![D::entry(0.0, 0.0); output_shape.iter().product()];
    for_each_index(&joint_shape, |joint| {
        let mut total = D::entry(0.0, 0.0);
        for_each_index(&contracted_shape, |k| {
            let mut a_offset = 0;
            for (position, &axis) in a_open.iter().enumerate() {
                a_offset += joint[position] * a_strides[axis];
            }
            let mut b_offset = 0;
            for (position, &axis) in b_open.iter().enumerate() {
                b_offset += joint[a_open.len() + position] * b_strides[axis];
            }
            for (pair, (&a_axis, &b_axis)) in case.lhs_axes.iter().zip(&case.rhs_axes).enumerate() {
                a_offset += k[pair] * a_strides[a_axis];
                b_offset += k[pair] * b_strides[b_axis];
            }
            total = total + a.data[a_offset] * b.data[b_offset];
        });
        let mut offset = 0;
        for (position, &axis) in case.output_axes.iter().enumerate() {
            offset += joint[axis] * output_strides[position];
        }
        output[offset] = total;
    });
    (output_shape, output)
}

/// Degeneracy-1 FZ2 map `V <- V` with `even`/`odd` block values: the tensor
/// `build(f)` produces in the TensorKit reference (values checked against
/// TensorKit 0.17.1 in `symtn-network/tests/tk_fermionic_correspondence.rs`).
pub fn fz2_map(runtime: &Runtime, even: f64, odd: f64) -> TensorMap<FermionParityFusionRule, f64> {
    let v = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&v], [&v], move |trees, _| {
        if *trees.coupled() == Z2Irrep::EVEN {
            even
        } else {
            odd
        }
    })
    .unwrap()
}

/// The TensorKit-valued FZ2 closed loops of `tk_fermionic_correspondence`,
/// written as explicit `contract` calls so they run through the general
/// contraction on whichever storage `lift` places the operands: every loop
/// closes a codomain leg against a domain leg, which bends a leg into B's
/// codomain as a dual leg and fires the supertrace twist. `S` is the dense
/// `diag(3, 2)` the reference's SVD produces. Returns `(name, value,
/// TensorKit value)`.
pub fn fz2_tensorkit_loops<T>(
    runtime: &Runtime,
    lift: impl Fn(TensorMap<FermionParityFusionRule, f64>) -> T,
    contract: impl Fn(&T, &T, &ContractSpec<'_>) -> T,
    scalar: impl Fn(T) -> f64,
) -> Vec<(&'static str, f64, f64)> {
    let a = lift(fz2_map(runtime, 1.0, 4.0));
    let b = lift(fz2_map(runtime, 2.0, 1.5));
    let c = lift(fz2_map(runtime, 0.5, 2.5));
    let s = lift(fz2_map(runtime, 3.0, 2.0));
    // `x[i; j] y[j; i]`: x's domain against y's codomain, x's codomain
    // against y's domain.
    let close = |x: &T, y: &T| {
        let spec = ContractSpec {
            lhs: &[1, 0],
            rhs: &[0, 1],
            codomain: &[],
            domain: &[],
        };
        scalar(contract(x, y, &spec))
    };
    let chain = |x: &T, y: &T| {
        let spec = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        };
        contract(x, y, &spec)
    };
    vec![
        ("tr(A B)", close(&a, &b), -4.0),
        ("tr(A B C)", close(&chain(&a, &b), &c), -14.0),
        ("tr(A S B)", close(&chain(&a, &s), &b), -6.0),
        ("tr(S A B)", close(&chain(&s, &a), &b), -6.0),
    ]
}

/// Every fermionic fixture (#1347), handed to `check` with the
/// provider's typed twist; shared by the Host pin and the device gate. A
/// macro because each provider's twist has its own dispatch bound.
macro_rules! for_each_fermionic_fixture {
    ($runtime:expr, $payload:ty, $check:ident) => {{
        let runtime = $runtime;
        let fu1 = $crate::contract_cases::fermion_u1();
        let fsu2 = $crate::contract_cases::fermion_su2();
        let twist_u1 =
            |t: &symtn::typed::TensorMap<$crate::contract_cases::FermionU1, $payload>,
             legs: &[usize]| t.twist(legs, symtn::typed::Direction::Forward).unwrap();
        let twist_su2 =
            |t: &symtn::typed::TensorMap<$crate::contract_cases::FermionSu2, $payload>,
             legs: &[usize]| t.twist(legs, symtn::typed::Direction::Forward).unwrap();
        $check(
            $crate::contract_cases::fermionic_general(
                runtime,
                &fu1,
                false,
                "fZ2xU1 both sides",
                31,
            ),
            twist_u1,
        );
        $check(
            $crate::contract_cases::fermionic_general(runtime, &fu1, true, "fZ2xU1 mixed θ", 33),
            twist_u1,
        );
        $check(
            $crate::contract_cases::fermionic_general(
                runtime,
                &fsu2,
                false,
                "fZ2xSU2 both sides",
                35,
            ),
            twist_su2,
        );
        $check(
            $crate::contract_cases::fermionic_general(runtime, &fsu2, true, "fZ2xSU2 mixed θ", 37),
            twist_su2,
        );
        $check(
            $crate::contract_cases::fermionic_canonical_nonuniform(
                runtime,
                &fu1,
                "fZ2xU1 canonical nonuniform",
                39,
            ),
            twist_u1,
        );
        $check(
            $crate::contract_cases::fermionic_canonical_nonuniform(
                runtime,
                &fsu2,
                "fZ2xSU2 canonical nonuniform",
                41,
            ),
            twist_su2,
        );
        $check(
            $crate::contract_cases::fermionic_canonical_uniform(
                runtime,
                &fu1,
                "fZ2xU1 canonical uniform",
                43,
            ),
            twist_u1,
        );
        $check(
            $crate::contract_cases::fermionic_canonical_uniform(
                runtime,
                &fsu2,
                "fZ2xSU2 canonical uniform",
                45,
            ),
            twist_su2,
        );
        for case in $crate::contract_cases::fermionic_twist_roles(
            runtime,
            &fu1,
            [
                "fZ2xU1 A copied",
                "fZ2xU1 canonical small A",
                "fZ2xU1 B copied",
                "fZ2xU1 both copied small A",
            ],
            47,
        ) {
            $check(case, twist_u1);
        }
        for case in $crate::contract_cases::fermionic_twist_roles(
            runtime,
            &fsu2,
            [
                "fZ2xSU2 A copied",
                "fZ2xSU2 canonical small A",
                "fZ2xSU2 B copied",
                "fZ2xSU2 both copied small A",
            ],
            55,
        ) {
            $check(case, twist_su2);
        }
        // `u` not dual: both lazy orderings then contract a leg that reaches
        // B's codomain dual, so each one is twisted.
        let lazy_u1: symtn::typed::TensorMap<$crate::contract_cases::FermionU1, $payload> =
            $crate::contract_cases::fermionic_general(runtime, &fu1, false, "", 43).lhs;
        for case in $crate::contract_cases::lazy_cases(&lazy_u1, "fZ2xU1 lazy") {
            $check(case, twist_u1);
        }
        let lazy_su2: symtn::typed::TensorMap<$crate::contract_cases::FermionSu2, $payload> =
            $crate::contract_cases::fermionic_general(runtime, &fsu2, false, "", 45).lhs;
        for case in $crate::contract_cases::lazy_cases(&lazy_su2, "fZ2xSU2 lazy") {
            $check(case, twist_su2);
        }
    }};
}
