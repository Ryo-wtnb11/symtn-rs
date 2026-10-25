//! Checked Generic `trace_pairs` against an independent contraction oracle
//! (#2054): tracing legs `(i, j)` equals contracting them with the identity
//! `isomorphism(S ← S)` on `S = space(t, i)`, as TensorKit's `@tensor`
//! lowers `t[a, b, a]` either way. The contraction shares no trace lowering
//! (split, channel match, channel factor). SU(2) and SU(3) through the
//! checked SU(N) provider, with outer multiplicity, several sectors and
//! degeneracies above one on every leg (so several fusion-tree groups
//! interleave in block order), `f64` and `Complex64`, codomain–domain and
//! codomain–codomain pairs, two pairs at once, and a lazy adjoint source
//! (its oracle contracts the materialized adjoint).

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::LieGroupFusionRule;
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap, TensorScalar};

use crate::numerics;

type Space = GradedSpace<LieGroupFusionRule>;
type Sectors = Vec<(Vec<i64>, usize)>;

/// Contract the codomain leg `lhs` of `x` with the leg `rhs` through
/// `id(space(x, lhs))`, keeping the remaining legs in their partitions.
fn contract_pair<D>(
    runtime: &Runtime,
    x: &TensorMap<LieGroupFusionRule, D>,
    lhs: usize,
    rhs: usize,
) -> TensorMap<LieGroupFusionRule, D>
where
    D: TensorScalar,
{
    let nout = x.codomain().len();
    let rank = nout + x.domain().len();
    assert!(lhs < nout, "the oracle pairs a codomain leg first");
    let space = x.codomain()[lhs].clone();
    let id = TensorMap::<_, D>::isomorphism(runtime, [&space], [&space]).unwrap();
    let open: Vec<usize> = (0..rank)
        .filter(|&axis| axis != lhs && axis != rhs)
        .collect();
    let codomain: Vec<usize> = (0..open.len()).filter(|&k| open[k] < nout).collect();
    let domain: Vec<usize> = (0..open.len()).filter(|&k| open[k] >= nout).collect();
    x.contract(
        &id,
        &ContractSpec {
            lhs: &[lhs, rhs],
            rhs: &[1, 0],
            codomain: &codomain,
            domain: &domain,
        },
    )
    .unwrap()
}

fn assert_trace_matches<D>(
    runtime: &Runtime,
    what: &str,
    source: &TensorMap<LieGroupFusionRule, D>,
    pairs: &[(usize, usize)],
    source_len: usize,
) where
    D: TensorScalar + numerics::Numeric,
{
    let got = source.trace_pairs(pairs).unwrap();
    // Contract the pairs one at a time, renumbering the later pairs past
    // the two legs each contraction removes. A lazy adjoint is materialized
    // first, so the oracle contracts owned operands only.
    let owned = source.materialize().unwrap();
    let mut oracle = contract_pair(runtime, &owned, pairs[0].0, pairs[0].1);
    let mut removed = vec![pairs[0].0, pairs[0].1];
    for &(lhs, rhs) in &pairs[1..] {
        let shift = |axis: usize| axis - removed.iter().filter(|&&gone| gone < axis).count();
        oracle = contract_pair(runtime, &oracle, shift(lhs), shift(rhs));
        removed.extend([lhs, rhs]);
    }
    assert_eq!(got.codomain(), oracle.codomain(), "{what}: codomain");
    assert_eq!(got.domain(), oracle.domain(), "{what}: domain");
    let (got, want) = (got.dense_data().unwrap(), oracle.dense_data().unwrap());
    assert!(
        want.iter().any(|value| value.wide().norm() > 0.5),
        "{what}: nonzero oracle"
    );
    // A trace is linear: each entry sums at most every source entry.
    numerics::assert_slices_close(what, got, want, source_len);
}

fn spaces(n: usize) -> (Space, Space) {
    let rule =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let (v, w): (Sectors, Sectors) = if n == 2 {
        (
            vec![(vec![0], 2), (vec![1], 1), (vec![2], 2)],
            vec![(vec![0], 1), (vec![1], 2), (vec![3], 1)],
        )
    } else {
        (
            vec![(vec![0, 0], 2), (vec![1, 0], 1), (vec![1, 1], 2)],
            vec![(vec![0, 0], 1), (vec![0, 1], 2), (vec![1, 1], 1)],
        )
    };
    (
        GradedSpace::try_new(Arc::clone(&rule), v).unwrap(),
        GradedSpace::try_new(rule, w).unwrap(),
    )
}

fn cases<D>(n: usize)
where
    D: TensorScalar + numerics::Numeric,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = spaces(n);
    let vd = v.try_dual().unwrap();
    let mixed = TensorMap::<_, D>::rand_with_seed(&runtime, [&v, &w, &v], [&v, &w], 11).unwrap();
    let mixed_len = mixed.dense_data().unwrap().len();
    assert_trace_matches(
        &runtime,
        &format!("SU({n}) codomain-domain"),
        &mixed,
        &[(0, 3)],
        mixed_len,
    );
    assert_trace_matches(
        &runtime,
        &format!("SU({n}) two pairs"),
        &mixed,
        &[(0, 3), (1, 4)],
        mixed_len,
    );
    let adjoint = mixed.adjoint().unwrap();
    assert_trace_matches(
        &runtime,
        &format!("SU({n}) lazy adjoint"),
        &adjoint,
        &[(0, 2)],
        mixed_len,
    );

    let bent = TensorMap::<_, D>::rand_with_seed(&runtime, [&v, &w, &vd], [&w], 13).unwrap();
    let bent_len = bent.dense_data().unwrap().len();
    assert_trace_matches(
        &runtime,
        &format!("SU({n}) codomain-codomain"),
        &bent,
        &[(0, 2)],
        bent_len,
    );
}

#[test]
fn checked_su2_trace_matches_identity_contraction() {
    cases::<f64>(2);
    cases::<Complex64>(2);
}

#[test]
fn checked_su3_trace_matches_identity_contraction() {
    cases::<f64>(3);
    cases::<Complex64>(3);
}
