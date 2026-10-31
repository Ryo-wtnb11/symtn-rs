//! Shared fixtures and the independent oracle for the compose plan/workspace
//! gates (#1639).
//!
//! Every fixture composes `A: V ⊗ V ← W` with `B: W ← V`. `W` lacks a sector
//! of `V` that `V ⊗ V` reaches, so the destination `V ⊗ V ← V` has coupled
//! sectors no member writes (inactive blocks), next to several active ones.
//!
//! The oracle never touches the plan under test: it reads the operands
//! through the public block API and computes TensorKit `mul!` literally, per
//! destination block `(X, Z)`, `Σ_Y A[X, Y] · B[Y, Z]` over fusion-tree keys
//! and degeneracy indices, with zero for a block no `Y` reaches.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, Fz2SectorLayout, MultiplicityFreeRigidSymbols, PackedProductCodec,
    SectorCodec, U1SectorLayout,
};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::common::{for_each_index, Payload};

pub mod eigh;

pub type Fz2U1Rule = ProductFusionRule<
    FermionParityFusionRule,
    U1FusionRule,
    PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
>;

pub fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let q = U1Irrep::new;
    (
        GradedSpace::try_new(Arc::new(U1FusionRule), [(q(-1), 2), (q(0), 1), (q(1), 3)]).unwrap(),
        GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap(),
    )
}

pub fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let j = SU2Irrep::from_twice_spin;
    (
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap(),
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 1), (j(2), 2)]).unwrap(),
    )
}

pub fn fz2u1_legs() -> (GradedSpace<Fz2U1Rule>, GradedSpace<Fz2U1Rule>) {
    let rule = Arc::new(Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap());
    let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
    let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
    (
        GradedSpace::try_new(Arc::clone(&rule), [(even(0), 2), (odd(1), 1), (odd(-1), 2)]).unwrap(),
        GradedSpace::try_new(rule, [(even(0), 1), (odd(1), 2)]).unwrap(),
    )
}

/// `count` tensors `codomain ← domain` with dyadic entries, distinct per
/// member and per `salt`.
pub fn members<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
    count: usize,
    salt: usize,
) -> Vec<TensorMap<R, D>>
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
    (0..count)
        .map(|member| {
            let mut next = 1000 * member + 100 * salt;
            TensorMap::from_subblock_fn(
                engine,
                runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                |_, _| {
                    next += 1;
                    let real = ((next * 37) % 17) as f64 / 8.0 - 1.0;
                    let imaginary = ((next * 11) % 13) as f64 / 16.0 - 0.375;
                    D::entry(real, imaginary)
                },
            )
            .unwrap()
        })
        .collect()
}

/// `count` tensors `codomain ← domain` filled with `value`.
pub fn filled<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
    count: usize,
    value: D,
) -> Vec<TensorMap<R, D>>
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
    (0..count)
        .map(|_| {
            TensorMap::from_subblock_fn(
                engine,
                runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                |_, _| value,
            )
            .unwrap()
        })
        .collect()
}

fn tree_key<S: Debug>(
    coupled: &S,
    uncoupled: &[S],
    innerlines: &[S],
    vertices: &impl Debug,
) -> String {
    format!("{coupled:?}|{uncoupled:?}|{innerlines:?}|{vertices:?}")
}

/// Block `index` of `tensor` as its `(codomain key, domain key)` and a
/// column-major `rows x cols` matrix, read through the block's own strides.
pub fn block_matrix<R, D>(
    tensor: &TensorMap<R, D>,
    index: usize,
) -> (String, String, usize, usize, Vec<D>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let trees = tensor.subblock_fusion_trees(index).unwrap();
    let codomain = tree_key(
        trees.coupled(),
        trees.codomain_uncoupled(),
        trees.codomain_innerlines(),
        &trees.codomain_vertices(),
    );
    let domain = tree_key(
        trees.coupled(),
        trees.domain_uncoupled(),
        trees.domain_innerlines(),
        &trees.domain_vertices(),
    );
    let block = tensor.subblock(index).unwrap();
    let nout = tensor.codomain_rank();
    let rows: usize = block.shape()[..nout].iter().product();
    let cols: usize = block.shape()[nout..].iter().product();
    let materialized = tensor.materialize(engine).unwrap();
    let data = materialized.dense_data().unwrap();
    let mut matrix = Vec::with_capacity(rows * cols);
    for_each_index(block.shape(), |index| {
        let position = block.offset()
            + index
                .iter()
                .zip(block.strides())
                .map(|(i, s)| i * s)
                .sum::<usize>();
        matrix.push(data[position]);
    });
    (codomain, domain, rows, cols, matrix)
}

/// TensorKit `mul!(C, A, B)` per fusion-tree block, laid out like
/// `template` (the destination's structure; its payload is not read).
/// Also returns how many destination blocks no inner tree reaches.
pub fn compose_oracle<R, D>(
    a: &TensorMap<R, D>,
    b: &TensorMap<R, D>,
    template: &TensorMap<R, D>,
) -> (Vec<D>, usize)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
    D: Payload,
{
    let zero = D::entry(0.0, 0.0);
    let a_blocks: Vec<_> = (0..a.subblock_count())
        .map(|i| block_matrix(a, i))
        .collect();
    let mut b_by_rows: HashMap<String, Vec<_>> = HashMap::new();
    for index in 0..b.subblock_count() {
        let block = block_matrix(b, index);
        b_by_rows.entry(block.0.clone()).or_default().push(block);
    }
    let mut out = vec![zero; template.dense_data().unwrap().len()];
    let mut unreached = 0;
    for index in 0..template.subblock_count() {
        let (x, z, rows, cols, _) = block_matrix(template, index);
        let mut product = vec![zero; rows * cols];
        let mut reached = false;
        for (ax, ay, a_rows, inner, a_matrix) in &a_blocks {
            if *ax != x {
                continue;
            }
            for (_, bz, b_inner, b_cols, b_matrix) in b_by_rows.get(ay).into_iter().flatten() {
                if *bz != z {
                    continue;
                }
                assert_eq!((*a_rows, *inner, *b_cols), (rows, *b_inner, cols));
                reached = true;
                for col in 0..cols {
                    for k in 0..*inner {
                        let factor = b_matrix[k + inner * col];
                        for row in 0..rows {
                            let term = mul(a_matrix[row + rows * k], factor);
                            product[row + rows * col] = add(product[row + rows * col], term);
                        }
                    }
                }
            }
        }
        unreached += usize::from(!reached);
        let block = template.subblock(index).unwrap();
        let mut linear = 0;
        for_each_index(block.shape(), |index| {
            let position = block.offset()
                + index
                    .iter()
                    .zip(block.strides())
                    .map(|(i, s)| i * s)
                    .sum::<usize>();
            out[position] = product[linear];
            linear += 1;
        });
    }
    (out, unreached)
}

fn mul<D: Payload>(a: D, b: D) -> D {
    let ((ar, ai), (br, bi)) = (a.parts(), b.parts());
    D::entry(ar * br - ai * bi, ar * bi + ai * br)
}

fn add<D: Payload>(a: D, b: D) -> D {
    let ((ar, ai), (br, bi)) = (a.parts(), b.parts());
    D::entry(ar + br, ai + bi)
}
