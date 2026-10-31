//! Factorizations take leg roles `(rows, cols)`: `t.op(rows, cols)` is
//! `t.permute(rows, cols)?.op(current split)` on the same engine. The oracle
//! is that composition, compared bit for bit (the same kernels on the same
//! bytes), and the factors rebuild the Host-permuted tensor.

use super::*;
use crate::legs::{su2_leg, u1_leg};
use num_complex::Complex64;
use std::sync::Arc;
use symtn::sector::{product_sector, FermionParityFusionRule, U1FusionRule, U1Irrep, Z2Irrep};
use symtn::typed::{Eigh, GradedSpace, HermitianTol, OperationError, Qr, Runtime, Svd};

/// Spaces and every stored value of two downloaded results, bit for bit.
macro_rules! same {
    ($engine:ident, $got:expr, $want:expr, $what:expr) => {{
        let (got, want) = (
            $got.materialize($engine).unwrap(),
            $want.materialize($engine).unwrap(),
        );
        assert!(
            got.codomain() == want.codomain() && got.domain() == want.domain(),
            "{}",
            $what
        );
        assert!(
            got.dense_data().unwrap() == want.dense_data().unwrap(),
            "{}",
            $what
        );
    }};
}

macro_rules! close {
    ($engine:ident, $got:expr, $want:expr, $what:expr) => {{
        let (got, want) = (&$got, &$want);
        let residual = got
            .axpby($engine, 1.0.into(), want, (-1.0).into())
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        assert!(
            residual <= 1e-10 * want.norm($engine, 2.0).unwrap().max(1.0),
            "{}: {residual}",
            $what
        );
    }};
}

/// `svd_compact` and `qr_compact` with rectangular roles that cross the
/// boundary and reorder, `eigh_full` with square roles on a Hermitian
/// source, and a malformed role list.
macro_rules! check {
    ($backend:ty, $device:ident, $engine:ident, $runtime:expr, $d:ty, $v:expr, $w:expr, $label:expr) => {{
        let (v, w) = (&$v, &$w);
        let host: TensorMap<_, $d> =
            TensorMap::rand_with_seed($engine, &$runtime, [v, w], [w, v], 1553).unwrap();
        let source = <$backend>::upload($device, &host);
        for (rows, cols) in [(&[2, 0][..], &[3, 1][..]), (&[3], &[1, 2, 0]), (&[1, 3, 0], &[2])] {
            let permuted = source.permute($device, rows, cols).unwrap();
            let host_permuted = host.permute($engine, rows, cols).unwrap();
            let nout = host_permuted.codomain_rank();
            let (ir, ic): (Vec<usize>, Vec<usize>) = ((0..nout).collect(), (nout..4).collect());
            let down = |device: &mut dyn Engine, t: &TensorMap<_, $d, _>| <$backend>::download(device, t);

            let Svd { u, s, vh } = source.svd_compact($device, rows, cols).unwrap();
            let expected = permuted.svd_compact($device, &ir, &ic).unwrap();
            let (u, s, vh) = (down($device, &u), down($device, &s), down($device, &vh));
            same!($engine, u, down($device, &expected.u), $label);
            same!($engine, s, down($device, &expected.s), $label);
            same!($engine, vh, down($device, &expected.vh), $label);
            let rebuilt = u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap();
            close!($engine, rebuilt, host_permuted, $label);

            let Qr { q, r } = source.qr_compact($device, rows, cols).unwrap();
            let expected = permuted.qr_compact($device, &ir, &ic).unwrap();
            let (q, r) = (down($device, &q), down($device, &r));
            same!($engine, q, down($device, &expected.q), $label);
            same!($engine, r, down($device, &expected.r), $label);
            close!($engine, q.compose($engine, &r).unwrap(), host_permuted, $label);
        }

        // Square roles on a Hermitian source: both sides swap their two legs.
        let square: TensorMap<_, $d> =
            TensorMap::rand_with_seed($engine, &$runtime, [v, w], [v, w], 1554).unwrap();
        let square_adjoint = square.adjoint($engine).unwrap().materialize($engine).unwrap();
        let hermitian = square.axpby($engine, 1.0.into(), &square_adjoint, 1.0.into()).unwrap();
        let source = <$backend>::upload($device, &hermitian);
        let (rows, cols) = ([1, 0], [3, 2]);
        let Eigh { d, v: vectors } =
            source.eigh_full($device, &rows, &cols, HermitianTol::DEFAULT).unwrap();
        let permuted = source.permute($device, &rows, &cols).unwrap();
        let expected = permuted.eigh_full($device, &[0, 1], &[2, 3], HermitianTol::DEFAULT).unwrap();
        let (d, vectors) = (<$backend>::download($device, &d), <$backend>::download($device, &vectors));
        same!($engine, d, <$backend>::download($device, &expected.d), $label);
        same!($engine, vectors, <$backend>::download($device, &expected.v), $label);
        let vectors_adjoint = vectors.adjoint($engine).unwrap().materialize($engine).unwrap();
        let rebuilt = vectors.compose($engine, &d).unwrap().compose($engine, &vectors_adjoint).unwrap();
        close!($engine, rebuilt, hermitian.permute($engine, &rows, &cols).unwrap(), $label);

        assert!(
            matches!(
                source.svd_compact($device, &[0, 0], &[2, 3]),
                Err(symtn::typed::Error::Operation(e))
                    if matches!(*e, OperationError::InvalidPermutation { rank: 4, .. })
            ),
            "{}: malformed roles",
            $label
        );
    }};
}

on_each_engine! {
    /// Leg roles equal the permute composition on U(1), SU(2) and fZ2×U(1),
    /// real and complex, with dual legs on both sides.
    fn leg_roles_equal_the_permute_composition<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let u1 = || (u1_leg(&[(-1, 1), (0, 2), (1, 2)]), u1_leg(&[(0, 1), (1, 2), (2, 1)]).try_dual().unwrap());
        let su2 = || (su2_leg(&[(0, 2), (1, 2), (2, 1)]), su2_leg(&[(1, 1), (2, 2)]).try_dual().unwrap());
        let rule = Arc::new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
                .unwrap(),
        );
        let fz2u1 = |sectors: &[(bool, i32, usize)]| {
            GradedSpace::try_new(
                Arc::clone(&rule),
                sectors.iter().map(|&(odd, q, n)| {
                    let parity = if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN };
                    (product_sector(parity, U1Irrep::new(q)), n)
                }),
            )
            .unwrap()
        };
        let fermionic = || {
            (
                fz2u1(&[(false, 0, 2), (true, 1, 2), (true, -1, 1)]),
                fz2u1(&[(false, 1, 1), (true, 0, 2)]).try_dual().unwrap(),
            )
        };
        let (v, w) = u1();
        check!(B, device, engine, runtime, f64, v, w, "u1 f64");
        check!(B, device, engine, runtime, Complex64, v, w, "u1 c64");
        let (v, w) = su2();
        check!(B, device, engine, runtime, f64, v, w, "su2 f64");
        check!(B, device, engine, runtime, Complex64, v, w, "su2 c64");
        let (v, w) = fermionic();
        check!(B, device, engine, runtime, f64, v, w, "fz2u1 f64");
        check!(B, device, engine, runtime, Complex64, v, w, "fz2u1 c64");
    }
}
