//! Operations that read a compact diagonal or an owned lazy adjoint in
//! place, and the `dense_data` refusal of a payload it would materialize.

#[cfg(test)]
use crate::su2::{su2_irrep, su2_rule, Su2Rule};
use std::sync::Arc;

use num_complex::Complex64;
use symtn_core::{U1FusionRule, U1Irrep};

use super::probe::{self, Event};
use super::{GradedSpace, Side, TensorMap};
use crate::error::{Alternative, Error};
use crate::runtime::Runtime;

fn runtime() -> Runtime {
    Runtime::new()
}

/// Runs `f` and returns its value with the number of entries into adjoint
/// payload materialization it made.
fn adjoint_entries<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let adjoint = probe::record();
    let value = f();
    (value, adjoint.get(Event::AdjointMaterialization))
}

/// The refusal names the operation and the `materialize` remedy.
fn is_unsupported(error: &impl std::fmt::Display, operation: &'static str) -> bool {
    let expected = Error::Unsupported {
        operation,
        alternative: Alternative::Materialize,
    }
    .to_string();
    assert!(expected.contains(operation) && expected.contains("materialize"));
    error.to_string() == expected
}

fn u1(pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        pairs.iter().map(|&(q, d)| (U1Irrep::new(q).unwrap(), d)),
    )
    .unwrap()
}

fn su2(pairs: &[(usize, usize)]) -> GradedSpace<Su2Rule> {
    GradedSpace::try_new(
        Arc::new(su2_rule()),
        pairs.iter().map(|&(j, d)| (su2_irrep(j), d)),
    )
    .unwrap()
}

/// `cat` reads a compact diagonal and a lazy adjoint directly; the result is
/// bitwise the cat of their materialized forms.
#[test]
fn cat_reads_compact_operand_without_densifying() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = runtime();
    let bond = u1(&[(-1, 2), (0, 1), (1, 3)]);
    let spectra = bond
        .sectors()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, sector)| super::SectorSpectrum {
            values: (0..bond.degeneracy(&sector).unwrap())
                .map(|k| 1.25 + i as f64 + 0.5 * k as f64)
                .collect(),
            sector,
        })
        .collect::<Vec<_>>();
    let d: TensorMap<_, f64> = TensorMap::diagonal(engine, &runtime, &bond, spectra).unwrap();
    let x: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&bond], [&bond], 7).unwrap();
    let y: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&bond], [&bond], 8).unwrap();
    let lazy = y.adjoint(engine).unwrap();
    let eager = lazy.materialize(engine).unwrap();
    let bits = |t: &TensorMap<_, f64>| {
        t.dense_data()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    for side in [Side::Domain, Side::Codomain] {
        for (name, lhs, rhs, compact) in [
            ("d|lazy", &d, &lazy, 0usize),
            ("lazy|d", &lazy, &d, 0),
            ("x|lazy", &x, &lazy, 0),
            ("lazy|x", &lazy, &x, 0),
        ] {
            let oracle_lhs = if std::ptr::eq(lhs, &lazy) {
                &eager
            } else {
                lhs
            };
            let oracle_rhs = if std::ptr::eq(rhs, &lazy) {
                &eager
            } else {
                rhs
            };
            let oracle_rhs_materialize = &oracle_rhs.materialize(engine).unwrap();
            let oracle = oracle_lhs
                .materialize(engine)
                .unwrap()
                .cat(engine, oracle_rhs_materialize, side)
                .unwrap();
            let diagonal = probe::record();
            let result = lhs.cat(engine, rhs, side).unwrap();
            let entries = diagonal.get(Event::DiagonalMaterialization);
            assert_eq!(bits(&result), bits(&oracle), "{name} {side:?}: vs oracle");
            assert_eq!(result.codomain(), oracle.codomain(), "{name} {side:?}");
            assert_eq!(result.domain(), oracle.domain(), "{name} {side:?}");
            assert_eq!(entries, compact, "{name} {side:?}: densifications");
        }
    }
}

/// `dense_data` borrows dense Host storage and refuses a lazy
/// adjoint or a compact diagonal with `Unsupported { Materialize }`,
/// entering neither adjoint materialization nor diagonal densification.
#[test]
fn dense_data_borrows_dense_storage_and_never_materializes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = runtime();
    let v = su2(&[(0, 2), (1, 1)]);
    let t: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 3).unwrap();
    let (dense, entries) = adjoint_entries(|| t.dense_data().unwrap());
    assert_eq!(entries, 0);
    assert_eq!(dense.as_ptr(), t.dense_data().unwrap().as_ptr());

    let adjoint = t.adjoint(engine).unwrap();
    let (refused, entries) = adjoint_entries(|| adjoint.dense_data().map(<[_]>::len));
    assert_eq!(entries, 0, "dense_data entered adjoint materialization");
    assert!(is_unsupported(&refused.unwrap_err(), "dense_data"));
    // Positive control: the explicit path enters the probe.
    let (materialized, entries) = adjoint_entries(|| adjoint.materialize(engine).unwrap());
    assert_eq!(entries, 1);
    assert_eq!(materialized.dense_data().unwrap().len(), dense.len());

    let bond = u1(&[(0, 2), (1, 1)]);
    let spectra = bond
        .sectors()
        .unwrap()
        .into_iter()
        .map(|sector| super::SectorSpectrum {
            values: vec![1.5; bond.degeneracy(&sector).unwrap()],
            sector,
        })
        .collect::<Vec<_>>();
    let d: TensorMap<_, f64> = TensorMap::diagonal(engine, &runtime, &bond, spectra).unwrap();
    let densified = probe::record();
    assert!(is_unsupported(&d.dense_data().unwrap_err(), "dense_data"));
    assert!(is_unsupported(
        &d.subblocks().map(|_| ()).unwrap_err(),
        "subblocks"
    ));
    assert_eq!(
        densified.get(Event::DiagonalMaterialization),
        0,
        "dense_data densified a compact diagonal"
    );
    let dense_d = d.materialize(engine).unwrap();
    assert_eq!(dense_d.dense_data().unwrap().len(), 2 * 2 + 1);
    assert!(dense_d.subblocks().is_ok());
    let (refused, entries) = adjoint_entries(|| adjoint.subblocks().map(|_| ()));
    assert_eq!(entries, 0, "subblocks entered adjoint materialization");
    assert!(is_unsupported(&refused.unwrap_err(), "subblocks"));
}
