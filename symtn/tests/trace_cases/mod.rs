//! Partial-trace fixtures and their independent oracles, shared by the Host
//! pin (`typed_trace_host_oracle.rs`, ungated) and the device gate
//! (`typed_cuda_trace.rs`).
//!
//! Two oracles, neither of which passes through the trace engine whose device
//! replay is under test:
//!
//! * [`dense_trace`] — the physical-basis expansion (U(1), SU(2)): a
//!   diagonal index sum over each traced pair, for pairs of a codomain leg
//!   with the equal domain leg, so no leg bends;
//! * the identity contraction — `trace(T, (a, b))` as a general `contract` of
//!   `T` with `id_V` (TensorKit's `@tensor T[i, i, ...]` against
//!   `T[i, k, ...] * id[k, i]`), for twist-free providers only: a fermionic
//!   supertrace differs from it by the twist.
//!
//! Every fixture's traced leg carries several sectors of degeneracy > 1, so
//! each destination block has several source blocks as producers; an
//! overwriting replay (instead of accumulating) would lose all but one.
//! Fermionic fixtures have only the Host as oracle here; the fZ2 supertrace
//! is pinned against a hand value in `typed_cuda_trace.rs`.

#![allow(dead_code)]

use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

use crate::contract_cases::{fermion_su2, fermion_u1, fill, su2, u1, u1_su2, Payload};

/// One trace: the operand, its pairs, and the independent oracle that
/// applies, computed on the Host when the fixture is built.
pub struct TraceCase<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    pub name: &'static str,
    pub tensor: TensorMap<R, D>,
    pub pairs: Vec<(usize, usize)>,
    /// The physical-basis diagonal sum applies (U(1)/SU(2), unbent pairs).
    pub dense: bool,
    /// `contract(T, id)` for twist-free providers.
    pub identity: Option<TensorMap<R, D>>,
}

impl<R, D> TraceCase<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    pub fn host(&self) -> TensorMap<R, D> {
        let engine = &mut symtn_dense::CpuEngine::serial();
        self.tensor
            .trace_pairs(engine, &self.pairs)
            .unwrap_or_else(|e| panic!("{}: {e:?}", self.name))
    }

    /// The longest sum an output entry combines is the traced physical
    /// dimension; this bounds it for every fixture here.
    pub fn terms(&self) -> usize {
        64
    }
}

fn tensor<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
    salt: usize,
) -> TensorMap<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    TensorMap::from_subblock_fn(
        engine,
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        fill(salt),
    )
    .unwrap()
}

/// `T[.., a, .., b, ..]` contracted against `id_V` over `(a, b)` with
/// `rhs_axes` naming which identity leg meets which traced leg.
fn identity_oracle<R, D>(
    runtime: &Runtime,
    tensor: &TensorMap<R, D>,
    space: &GradedSpace<R>,
    lhs_axes: [usize; 2],
    rhs_axes: [usize; 2],
) -> TensorMap<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let identity: TensorMap<R, D> =
        TensorMap::isomorphism(engine, runtime, [space], [space]).unwrap();
    let open = tensor.rank() - 2;
    let axes: Vec<usize> = (0..open).collect();
    // The default split puts every open leg of `T` in the codomain; the
    // trace keeps each leg on its side, so bend the result to that split.
    let joint = tensor
        .contract(
            engine,
            &identity,
            &ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &axes,
                domain: &[],
            },
        )
        .unwrap();
    let codomain = tensor.codomain_rank()
        - lhs_axes
            .iter()
            .filter(|&&axis| axis < tensor.codomain_rank())
            .count();
    joint
        .permute(engine, &axes[..codomain], &axes[codomain..])
        .unwrap()
}

/// The unbent pair shapes over one bosonic leg `v` (and a spectator `w`):
/// one pair, two pairs, and the full trace to a scalar.
fn unbent<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    w: &GradedSpace<R>,
    names: [&'static str; 3],
    dense: bool,
    salt: usize,
) -> [TraceCase<R, D>; 3]
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let rank_five = tensor(runtime, &[v, w, v], &[v, w], salt);
    [
        TraceCase {
            name: names[0],
            tensor: rank_five.clone(),
            pairs: vec![(0, 3)],
            dense,
            identity: None,
        },
        TraceCase {
            name: names[1],
            tensor: rank_five,
            pairs: vec![(0, 3), (1, 4)],
            dense,
            identity: None,
        },
        TraceCase {
            name: names[2],
            tensor: tensor(runtime, &[v, w], &[v, w], salt + 1),
            pairs: vec![(1, 3), (0, 2)],
            dense,
            identity: None,
        },
    ]
}

/// A dual leg traced against its codomain partner — the leg bends — and the
/// same pair in the domain, each with its identity-contraction oracle.
fn bent<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    w: &GradedSpace<R>,
    names: [&'static str; 2],
    salt: usize,
) -> [TraceCase<R, D>; 2]
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let dual = v.try_dual().unwrap();
    let codomain = tensor(runtime, &[w, v, &dual], &[w], salt);
    let domain = tensor(runtime, &[w], &[&dual, w, v], salt + 1);
    [
        TraceCase {
            name: names[0],
            identity: Some(identity_oracle(runtime, &codomain, v, [1, 2], [1, 0])),
            tensor: codomain,
            pairs: vec![(1, 2)],
            dense: false,
        },
        TraceCase {
            name: names[1],
            identity: Some(identity_oracle(runtime, &domain, v, [1, 3], [1, 0])),
            tensor: domain,
            pairs: vec![(1, 3)],
            dense: false,
        },
    ]
}

pub fn u1_cases<D: Payload>(runtime: &Runtime) -> Vec<TraceCase<symtn::sector::U1FusionRule, D>> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 1), (1, 2)]);
    let mut cases: Vec<_> = unbent(
        runtime,
        &v,
        &w,
        ["U(1) one pair", "U(1) two pairs", "U(1) full trace"],
        true,
        11,
    )
    .into();
    cases.extend(bent(
        runtime,
        &v,
        &w,
        ["U(1) dual codomain pair", "U(1) dual domain pair"],
        13,
    ));
    cases
}

pub fn su2_cases<D: Payload>(runtime: &Runtime) -> Vec<TraceCase<symtn::sector::SU2FusionRule, D>> {
    let s = su2();
    let mut cases: Vec<_> = unbent(
        runtime,
        &s,
        &s,
        ["SU(2) one pair", "SU(2) two pairs", "SU(2) full trace"],
        true,
        21,
    )
    .into();
    cases.extend(bent(
        runtime,
        &s,
        &s,
        ["SU(2) dual codomain pair", "SU(2) dual domain pair"],
        23,
    ));
    cases
}

pub fn u1_su2_cases<D: Payload>(
    runtime: &Runtime,
) -> Vec<TraceCase<crate::contract_cases::U1Su2, D>> {
    let p = u1_su2();
    let mut cases: Vec<_> = unbent(
        runtime,
        &p,
        &p,
        [
            "U(1)xSU(2) one pair",
            "U(1)xSU(2) two pairs",
            "U(1)xSU(2) full trace",
        ],
        false,
        31,
    )
    .into();
    cases.extend(bent(
        runtime,
        &p,
        &p,
        [
            "U(1)xSU(2) dual codomain pair",
            "U(1)xSU(2) dual domain pair",
        ],
        33,
    ));
    cases
}

/// Fermionic traces over plain and dual legs, codomain against domain and
/// codomain against codomain, several pairs at once: the supertrace twist of
/// every non-dual traced leg after the first is part of the Host coefficient.
fn fermionic<R, D>(runtime: &Runtime, f: &GradedSpace<R>, salt: usize) -> Vec<TraceCase<R, D>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let dual = f.try_dual().unwrap();
    let t = tensor(runtime, &[f, &dual, f], &[f, &dual], salt);
    let bent = tensor(runtime, &[&dual, f, f], &[f], salt + 1);
    [
        ("fermionic plain pair", t.clone(), vec![(0, 3)]),
        ("fermionic dual pair", t.clone(), vec![(1, 4)]),
        ("fermionic two pairs", t, vec![(1, 4), (0, 3)]),
        ("fermionic dual codomain pair", bent.clone(), vec![(0, 1)]),
        (
            "fermionic full trace",
            tensor(runtime, &[f, &dual], &[f, &dual], salt + 2),
            vec![(0, 2), (1, 3)],
        ),
        ("fermionic reversed codomain pair", bent, vec![(1, 0)]),
    ]
    .into_iter()
    .map(|(name, tensor, pairs)| TraceCase {
        name,
        tensor,
        pairs,
        dense: false,
        identity: None,
    })
    .collect()
}

pub fn fermion_u1_cases<D: Payload>(
    runtime: &Runtime,
) -> Vec<TraceCase<crate::contract_cases::FermionU1, D>> {
    fermionic(runtime, &fermion_u1(), 41)
}

pub fn fermion_su2_cases<D: Payload>(
    runtime: &Runtime,
) -> Vec<TraceCase<crate::contract_cases::FermionSu2, D>> {
    fermionic(runtime, &fermion_su2(), 51)
}

/// Lazy-adjoint operands of `cases`, each traced over the pairs that name the
/// same legs of the adjoint (codomain axis `i` of `T` is axis `nin + i` of
/// `T†`, domain axis `nout + j` is axis `j`): the trace reads the parent
/// through its adjoint axes with a conjugated read.
pub fn lazy<R, D>(cases: Vec<TraceCase<R, D>>) -> Vec<TraceCase<R, D>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    cases
        .into_iter()
        .map(|case| TraceCase {
            name: case.name,
            pairs: adjoint_pairs(&case.tensor, &case.pairs),
            tensor: case.tensor.adjoint(engine).unwrap(),
            dense: false,
            identity: None,
        })
        .collect()
}

/// The pairs of `tensor.adjoint()` naming the legs `pairs` names on `tensor`.
pub fn adjoint_pairs<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    pairs: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let (nout, nin) = (tensor.codomain_rank(), tensor.domain_rank());
    let axis = |axis: usize| if axis < nout { nin + axis } else { axis - nout };
    pairs.iter().map(|&(a, b)| (axis(a), axis(b))).collect()
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

/// Physical-basis partial trace: `out[o] = sum_k T[o with every pair at its
/// own k_p]`, the untraced axes kept in order, all arrays column-major.
pub fn dense_trace<R, D>(case: &TraceCase<R, D>) -> Vec<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + symtn::sector::PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let t = case.tensor.to_physical_dense(engine).unwrap();
    let strides = column_major_strides(&t.shape);
    let traced: Vec<usize> = case.pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let open: Vec<usize> = (0..t.shape.len())
        .filter(|axis| !traced.contains(axis))
        .collect();
    let open_shape: Vec<usize> = open.iter().map(|&axis| t.shape[axis]).collect();
    let trace_shape: Vec<usize> = case.pairs.iter().map(|&(a, _)| t.shape[a]).collect();
    let mut output = Vec::new();
    crate::common::for_each_index(&open_shape, |o| {
        let mut total = D::entry(0.0, 0.0);
        crate::common::for_each_index(&trace_shape, |k| {
            let mut offset = 0;
            for (position, &axis) in open.iter().enumerate() {
                offset += o[position] * strides[axis];
            }
            for (pair, &(a, b)) in case.pairs.iter().enumerate() {
                offset += k[pair] * (strides[a] + strides[b]);
            }
            total = total + t.data[offset];
        });
        output.push(total);
    });
    if open_shape.is_empty() {
        // A full trace: `for_each_index` over the empty shape visits once.
        assert_eq!(output.len(), 1);
    }
    output
}
