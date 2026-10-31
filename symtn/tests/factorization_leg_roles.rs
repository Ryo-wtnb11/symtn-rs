//! Every factorization and matrix function takes its leg roles
//! `(rows, cols)`, and `t.op(rows, cols)` is defined as
//! `t.permute(rows, cols)?.op(current split)`.
//!
//! The oracle is that composition, spelled with the `permute` primitive and
//! the current-split call, whose factors `named_factorization_reconstruction`
//! checks by reconstruction. Both sides run the same deterministic kernels on the same bytes with a
//! single-threaded dense executor, so the factors are compared bit for bit.
//! Reconstructions against the permuted tensor are checked as well, so a
//! wrong permutation cannot hide behind an equally wrong oracle.

use std::sync::Arc;
use symtn::typed::HermitianTol;

use num_complex::Complex64;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::OperationError;
use symtn::typed::{
    Eig, Eigh, Error, GradedSpace, LeftPolar, Lq, Qr, RightPolar, Runtime, Svd, TensorMap,
};

use crate::report;

use crate::counting_alloc;

/// `(allocations, bytes)` requested on this thread while `f` runs.
use symtn_dense::Engine;

fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let (value, allocs) = counting_alloc::measure(f);
    (value, allocs.calls as usize, allocs.bytes as usize)
}

use crate::fixtures;

use fixtures::host_runtime;

/// Tree transforms executed so far: every transform looks its layout up once.
fn transforms(_: &Runtime) -> usize {
    let info = completed_transformers();
    (info.hits() + info.misses()) as usize
}

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-1, 1), (0, 2), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let w = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(0, 1), (1, 2), (2, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    (v, w.try_dual().unwrap())
}

fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let v = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(0, 2), (1, 2), (2, 1)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
    )
    .unwrap();
    let w = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(1, 1), (2, 2)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
    )
    .unwrap();
    (v, w.try_dual().unwrap())
}

type Fz2U1 = symtn::sector::ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn fz2_u1_legs() -> (GradedSpace<Fz2U1>, GradedSpace<Fz2U1>) {
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
        ],
    )
    .unwrap();
    let w = GradedSpace::try_new(
        rule,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2),
        ],
    )
    .unwrap();
    (v, w.try_dual().unwrap())
}

/// Space, storage and every stored value of a result, bit for bit.
macro_rules! fingerprint {
    ($engine:ident, $t:expr) => {{
        let t = &$t;
        (
            t.codomain(),
            t.domain(),
            t.materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .to_vec(),
        )
    }};
}

macro_rules! assert_same {
    ($engine:ident, $a:expr, $b:expr, $what:expr) => {
        assert!(
            fingerprint!($engine, $a) == fingerprint!($engine, $b),
            "{} differs from the permute composition",
            $what
        )
    };
}

macro_rules! assert_close {
    ($engine:ident, $a:expr, $b:expr, $tol:expr, $what:expr) => {{
        let (a, b) = (&$a, &$b);
        let diff = a
            .axpby($engine, 1.0.into(), b, (-1.0).into())
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        let scale = b.norm($engine, 2.0).unwrap().max(1.0);
        assert!(diff <= $tol * scale, "{}: residual {diff}", $what);
    }};
}

/// Every operation with rectangular roles `rows <- cols` equals the
/// composition, and its factors rebuild the permuted tensor.
macro_rules! check_rectangular {
    ($engine:ident, $runtime:expr, $t:expr, $rows:expr, $cols:expr, $label:expr) => {{
        let (t, rows, cols) = (&$t, &$rows[..], &$cols[..]);
        let p = t.permute($engine, rows, cols).unwrap();
        let nout = p.codomain_rank();
        let (ir, ic): (Vec<usize>, Vec<usize>) = ((0..nout).collect(), (nout..p.rank()).collect());
        let (ir, ic) = (&ir[..], &ic[..]);
        let tol = 1e-10;

        let Svd { u, s, vh } = t.svd_compact($engine, rows, cols).unwrap();
        let expected = p.svd_compact($engine, ir, ic).unwrap();
        assert_same!($engine, u, expected.u, format!("{} svd_compact u", $label));
        assert_same!($engine, s, expected.s, format!("{} svd_compact s", $label));
        assert_same!($engine, vh, expected.vh, format!("{} svd_compact vh", $label));
        assert_close!($engine, u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap(), p, tol, $label);

        let Svd { u, s, vh } = t.svd_full($engine, rows, cols).unwrap();
        let expected = p.svd_full($engine, ir, ic).unwrap();
        assert_same!($engine, u, expected.u, format!("{} svd_full u", $label));
        assert_same!($engine, s, expected.s, format!("{} svd_full s", $label));
        assert_same!($engine, vh, expected.vh, format!("{} svd_full vh", $label));
        assert_close!($engine, u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap(), p, tol, $label);

        assert_eq!(
            t.svd_vals($engine, rows, cols).unwrap(),
            p.svd_vals($engine, ir, ic).unwrap(),
            "{} svd_vals",
            $label
        );

        for full in [false, true] {
            let (Qr { q, r }, expected) = if full {
                (t.qr_full($engine, rows, cols).unwrap(), p.qr_full($engine, ir, ic).unwrap())
            } else {
                (
                    t.qr_compact($engine, rows, cols).unwrap(),
                    p.qr_compact($engine, ir, ic).unwrap(),
                )
            };
            assert_same!($engine, q, expected.q, format!("{} qr q", $label));
            assert_same!($engine, r, expected.r, format!("{} qr r", $label));
            assert_close!($engine, q.compose($engine, &r).unwrap(), p, tol, $label);

            let (Lq { l, q }, expected) = if full {
                (t.lq_full($engine, rows, cols).unwrap(), p.lq_full($engine, ir, ic).unwrap())
            } else {
                (
                    t.lq_compact($engine, rows, cols).unwrap(),
                    p.lq_compact($engine, ir, ic).unwrap(),
                )
            };
            assert_same!($engine, l, expected.l, format!("{} lq l", $label));
            assert_same!($engine, q, expected.q, format!("{} lq q", $label));
            assert_close!($engine, l.compose($engine, &q).unwrap(), p, tol, $label);
        }

        assert_same!($engine,
            t.left_null($engine, rows, cols).unwrap(),
            p.left_null($engine, ir, ic).unwrap(),
            format!("{} left_null", $label)
        );
        assert_same!($engine,
            t.right_null($engine, rows, cols).unwrap(),
            p.right_null($engine, ir, ic).unwrap(),
            format!("{} right_null", $label)
        );
        assert_same!($engine,
            t.pinv($engine, rows, cols, 1e-12).unwrap(),
            p.pinv($engine, ir, ic, 1e-12).unwrap(),
            format!("{} pinv", $label)
        );
        // The polar decompositions need tall / wide blocks: use the adjoint
        // composition for the orientation that fits.
        if let Ok(LeftPolar { w, p: pos }) = p.left_polar($engine, ir, ic) {
            let LeftPolar { w: w2, p: pos2 } = t.left_polar($engine, rows, cols).unwrap();
            assert_same!($engine, w2, w, format!("{} left_polar w", $label));
            assert_same!($engine, pos2, pos, format!("{} left_polar p", $label));
        } else {
            let error = t.left_polar($engine, rows, cols).unwrap_err();
            assert!(matches!(report::cause::<OperationError>(&error), Some(OperationError::InvalidArgument { message }) if message.contains("left_polar requires")));
        }
        if let Ok(RightPolar { p: pos, wh }) = p.right_polar($engine, ir, ic) {
            let RightPolar { p: pos2, wh: wh2 } = t.right_polar($engine, rows, cols).unwrap();
            assert_same!($engine, wh2, wh, format!("{} right_polar wh", $label));
            assert_same!($engine, pos2, pos, format!("{} right_polar p", $label));
        } else {
            let error = t.right_polar($engine, rows, cols).unwrap_err();
            assert!(matches!(report::cause::<OperationError>(&error), Some(OperationError::InvalidArgument { message }) if message.contains("right_polar requires")));
        }
    }};
}

/// Endomorphism roles: `rows` and `cols` name the same spaces after the
/// permute, so `eigh`, `eig`, `exp` and `inv` apply.
macro_rules! check_square {
    ($engine:ident, $t:expr, $rows:expr, $cols:expr, $label:expr) => {{
        let (t, rows, cols) = (&$t, &$rows[..], &$cols[..]);
        let t_adjoint = &t.adjoint($engine).unwrap().materialize($engine).unwrap();
        // A Hermitian input for eigh: h = t + t† in the source split stays
        // Hermitian under the same permutation of both sides.
        let h = t.axpby($engine, 1.0.into(), t_adjoint, 1.0.into()).unwrap();
        let p = t.permute($engine, rows, cols).unwrap();
        let hp = h.permute($engine, rows, cols).unwrap();
        assert_eq!(p.codomain(), p.domain(), "{} roles are not square", $label);
        let nout = p.codomain_rank();
        let (ir, ic): (Vec<usize>, Vec<usize>) = ((0..nout).collect(), (nout..p.rank()).collect());
        let (ir, ic) = (&ir[..], &ic[..]);

        let Eigh { d, v } = h
            .eigh_full($engine, rows, cols, HermitianTol::DEFAULT)
            .unwrap();
        let expected = hp
            .eigh_full($engine, ir, ic, HermitianTol::DEFAULT)
            .unwrap();
        assert_same!($engine, d, expected.d, format!("{} eigh d", $label));
        assert_same!($engine, v, expected.v, format!("{} eigh v", $label));
        let v_adjoint = &v.adjoint($engine).unwrap().materialize($engine).unwrap();
        assert_close!(
            $engine,
            v.compose($engine, &d)
                .unwrap()
                .compose($engine, v_adjoint)
                .unwrap(),
            hp,
            1e-10,
            $label
        );
        assert_eq!(
            h.eigh_vals($engine, rows, cols, HermitianTol::DEFAULT)
                .unwrap(),
            hp.eigh_vals($engine, ir, ic, HermitianTol::DEFAULT)
                .unwrap()
        );

        let Eig { d, v } = t.eig_full($engine, rows, cols).unwrap();
        let expected = p.eig_full($engine, ir, ic).unwrap();
        assert_same!($engine, d, expected.d, format!("{} eig d", $label));
        assert_same!($engine, v, expected.v, format!("{} eig v", $label));
        assert_eq!(
            t.eig_vals($engine, rows, cols).unwrap(),
            p.eig_vals($engine, ir, ic).unwrap()
        );

        let exp = t.exp($engine, rows, cols).unwrap();
        assert_same!(
            $engine,
            exp,
            p.exp($engine, ir, ic).unwrap(),
            format!("{} exp", $label)
        );
        let inv = t.inv($engine, rows, cols).unwrap();
        assert_same!(
            $engine,
            inv,
            p.inv($engine, ir, ic).unwrap(),
            format!("{} inv", $label)
        );
        let id = TensorMap::isomorphism($engine, p.runtime(), &p.codomain(), &p.domain()).unwrap();
        assert_close!(
            $engine,
            p.compose($engine, &inv).unwrap(),
            id,
            1e-8,
            format!("{} inv", $label)
        );
    }};
}

macro_rules! check_fixture {
    ($engine:ident, $runtime:expr, $legs:expr, $d:ty, $seed:expr, $label:expr) => {{
        let (v, w) = $legs;
        // Rectangular roles cross the codomain/domain boundary and reorder.
        let t: TensorMap<_, $d> =
            TensorMap::rand_with_seed($engine, &$runtime, [&v, &w], [&w, &v], $seed).unwrap();
        check_rectangular!($engine, $runtime, t, [2, 0], [3, 1], $label);
        check_rectangular!($engine, $runtime, t, [3], [1, 2, 0], $label);
        check_rectangular!($engine, $runtime, t, [1, 3, 0], [2], $label);
        // Square roles: swap the two codomain legs and the two domain legs.
        let t: TensorMap<_, $d> =
            TensorMap::rand_with_seed($engine, &$runtime, [&v, &w], [&v, &w], $seed + 1).unwrap();
        check_square!($engine, t, [1, 0], [3, 2], $label);
    }};
}

#[test]
fn leg_roles_equal_the_permute_composition() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    check_fixture!(engine, runtime, u1_legs(), f64, 1553, "u1 f64");
    check_fixture!(engine, runtime, u1_legs(), Complex64, 1553, "u1 c64");
    check_fixture!(engine, runtime, su2_legs(), f64, 1553, "su2 f64");
    check_fixture!(engine, runtime, su2_legs(), Complex64, 1553, "su2 c64");
    check_fixture!(engine, runtime, fz2_u1_legs(), f64, 1553, "fz2u1 f64");
    check_fixture!(engine, runtime, fz2_u1_legs(), Complex64, 1553, "fz2u1 c64");
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_leg_roles_equal_the_permute_composition() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;
    let runtime = host_runtime();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let v =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2), (vec![0, 0], 1)]).unwrap();
    let w = GradedSpace::try_new(provider, [(vec![1, 0], 1), (vec![1, 1], 1)])
        .unwrap()
        .try_dual()
        .unwrap();
    check_fixture!(
        engine,
        runtime,
        (v.clone(), w.clone()),
        f64,
        1553,
        "su3 f64"
    );
    check_fixture!(engine, runtime, (v, w), Complex64, 1553, "su3 c64");
}

#[test]
fn lazy_adjoint_leg_roles_equal_the_permute_composition() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    let (v, w) = su2_legs();
    let t: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v, &w], [&w, &v], 7).unwrap();
    let adjoint = t.adjoint(engine).unwrap();
    let explicit = adjoint.permute(engine, &[2, 0], &[3, 1]).unwrap();
    let Svd { u, s, vh } = adjoint.svd_compact(engine, &[2, 0], &[3, 1]).unwrap();
    let expected = explicit.svd_compact(engine, &[0, 1], &[2, 3]).unwrap();
    assert_same!(engine, u, expected.u, "adjoint u");
    assert_same!(engine, s, expected.s, "adjoint s");
    assert_same!(engine, vh, expected.vh, "adjoint vh");
}

/// A permutation that leaves the space unchanged (`[v, v] <- [w, w]` with
/// both sides swapped) makes the operation on the permuted tensor
/// structurally identical to the current-split operation, so their requested
/// allocations and bytes must agree: the leg roles add exactly one permute.
///
/// Each call is measured on its own engine after the same warm-up call.
/// Why not one shared engine: Tenferro's faer SVD checks scratch and factor
/// buffers out of the engine's best-fit buffer pool, so whether a buffer is
/// fresh, and whether symtn's factor publication must grow the first
/// sector's factor to the published length (`append_owned_factor`, 688 B on
/// this fixture), depends on which buffers earlier calls returned to that
/// pool, not on the operation.
#[test]
fn space_preserving_roles_cost_the_current_split_plus_one_permute() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    let (v, w) = su2_legs();
    let t: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v, &v], [&w, &w], 9).unwrap();
    let (rows, cols) = ([1, 0], [3, 2]);
    let current = ([0, 1], [2, 3]);
    let p = t.permute(engine, &rows, &cols).unwrap();
    assert!(p.codomain() == t.codomain() && p.domain() == t.domain());
    let _ = t.svd_compact(engine, &rows, &cols).unwrap();
    let _ = t.svd_compact(engine, &current.0, &current.1).unwrap();
    let _ = p.svd_compact(engine, &current.0, &current.1).unwrap();
    let on_own_engine = |f: &dyn Fn(&mut dyn Engine)| {
        let engine = &mut symtn_dense::CpuEngine::serial();
        f(engine);
        let ((), calls, bytes) = measure(|| f(engine));
        (calls, bytes)
    };

    let before = transforms(&runtime);
    let (current_allocations, current_bytes) =
        on_own_engine(&|engine| drop(t.svd_compact(engine, &current.0, &current.1).unwrap()));
    let (op_allocations, op_bytes) =
        on_own_engine(&|engine| drop(p.svd_compact(engine, &current.0, &current.1).unwrap()));
    assert_eq!(
        transforms(&runtime),
        before,
        "the current split ran a transform"
    );
    assert_eq!(
        (op_allocations, op_bytes),
        (current_allocations, current_bytes)
    );
    let (fused_allocations, fused_bytes) =
        on_own_engine(&|engine| drop(t.svd_compact(engine, &rows, &cols).unwrap()));
    let after_fused = transforms(&runtime);
    let (permute_allocations, permute_bytes) =
        on_own_engine(&|engine| drop(t.permute(engine, &rows, &cols).unwrap()));
    assert_eq!(after_fused - before, transforms(&runtime) - after_fused);
    assert_eq!(
        (fused_allocations, fused_bytes),
        (
            permute_allocations + current_allocations,
            permute_bytes + current_bytes
        )
    );
}

/// The performance contract: the current split runs no transform, and any
/// other split is exactly one transform plus the operation, the same work
/// and requested bytes as the explicit `permute` followed by the operation.
#[test]
fn leg_roles_cost_exactly_the_explicit_composition() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    let (v, w) = su2_legs();
    let t: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v, &w], [&w, &v], 3).unwrap();
    let (rows, cols) = ([2, 0], [3, 1]);
    // Warm the transform and dense caches once so every measured call replays.
    let _ = t.svd_compact(engine, &rows, &cols).unwrap();
    let _ = t.svd_compact(engine, &[0, 1], &[2, 3]).unwrap();
    let _ = t
        .permute(engine, &rows, &cols)
        .unwrap()
        .svd_compact(engine, &[0, 1], &[2, 3])
        .unwrap();

    let before = transforms(&runtime);
    let _ = t.svd_compact(engine, &[0, 1], &[2, 3]).unwrap();
    let after_identity = transforms(&runtime);
    assert_eq!(after_identity, before, "the current split ran a transform");

    let (_, fused_allocations, fused_bytes) =
        measure(|| t.svd_compact(engine, &rows, &cols).unwrap());
    let after_fused = transforms(&runtime);

    let (p, permute_allocations, permute_bytes) =
        measure(|| t.permute(engine, &rows, &cols).unwrap());
    let after_permute = transforms(&runtime);
    let (_, op_allocations, op_bytes) =
        measure(|| p.svd_compact(engine, &[0, 1], &[2, 3]).unwrap());
    assert_eq!(
        transforms(&runtime),
        after_permute,
        "the operation itself ran a transform"
    );
    assert!(after_permute > after_fused, "the permute ran no transform");
    assert_eq!(
        after_fused - after_identity,
        after_permute - after_fused,
        "other roles run exactly the explicit permute's transform work"
    );
    assert_eq!(
        (fused_allocations, fused_bytes),
        (
            permute_allocations + op_allocations,
            permute_bytes + op_bytes
        )
    );
}

#[test]
fn solve_roles_cost_the_two_explicit_permutations() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    let (v, w) = su2_legs();
    let a: TensorMap<_, f64> =
        TensorMap::isomorphism(engine, &runtime, [&v, &w], [&v, &w]).unwrap();
    let b: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v, &w], [&v, &w], 77).unwrap();
    let rows = [1, 0];
    let cols = [3, 2];
    let identity_rows = [0, 1];
    let identity_cols = [2, 3];
    let _ = a.solve(engine, &rows, &cols, &b, &rows, &cols).unwrap();
    let pa = a.permute(engine, &rows, &cols).unwrap();
    let pb = b.permute(engine, &rows, &cols).unwrap();
    let _ = pa
        .solve(
            engine,
            &identity_rows,
            &identity_cols,
            &pb,
            &identity_rows,
            &identity_cols,
        )
        .unwrap();

    let before = transforms(&runtime);
    let _ = a
        .solve(
            engine,
            &identity_rows,
            &identity_cols,
            &b,
            &identity_rows,
            &identity_cols,
        )
        .unwrap();
    assert_eq!(
        transforms(&runtime),
        before,
        "identity roles ran a transform"
    );

    let (_, fused_calls, fused_bytes) =
        measure(|| a.solve(engine, &rows, &cols, &b, &rows, &cols).unwrap());
    let after_fused = transforms(&runtime);
    let (pa, lhs_calls, lhs_bytes) = measure(|| a.permute(engine, &rows, &cols).unwrap());
    let (pb, rhs_calls, rhs_bytes) = measure(|| b.permute(engine, &rows, &cols).unwrap());
    let after_permutations = transforms(&runtime);
    let (_, solve_calls, solve_bytes) = measure(|| {
        pa.solve(
            engine,
            &identity_rows,
            &identity_cols,
            &pb,
            &identity_rows,
            &identity_cols,
        )
        .unwrap()
    });
    assert_eq!(transforms(&runtime), after_permutations);
    assert_eq!(after_fused - before, after_permutations - after_fused);
    assert_eq!(
        (fused_calls, fused_bytes),
        (
            lhs_calls + rhs_calls + solve_calls,
            lhs_bytes + rhs_bytes + solve_bytes
        )
    );
}

#[test]
fn malformed_roles_are_rejected_before_the_operation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    let (v, w) = u1_legs();
    let t: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v, &w], [&w, &v], 1).unwrap();
    let invalid = |error: Error| {
        assert!(
            matches!(
                &error,
                Error::Operation(operation)
                    if matches!(&**operation, OperationError::InvalidPermutation { .. })
            ),
            "{error:?}"
        );
    };
    let before = transforms(&runtime);
    for (rows, cols) in [
        (&[0, 0][..], &[2, 3][..]),
        (&[0, 1][..], &[2][..]),
        (&[0, 1][..], &[2, 4][..]),
    ] {
        invalid(t.svd_compact(engine, rows, cols).err().unwrap());
        invalid(t.qr_compact(engine, rows, cols).err().unwrap());
        invalid(t.exp(engine, rows, cols).err().unwrap());
    }
    assert_eq!(
        transforms(&runtime),
        before,
        "a rejected role ran a transform"
    );
    // Zero-length sides are not malformed: all legs on one side is the
    // row (or column) vector view, TensorKit's `permute(t, ((), (1, 2, 3, 4)))`.
    for (rows, cols) in [(&[][..], &[0, 1, 2, 3][..]), (&[0, 1, 2, 3][..], &[][..])] {
        let p = t.permute(engine, rows, cols).unwrap();
        let nout = p.codomain_rank();
        let (ir, ic): (Vec<usize>, Vec<usize>) = ((0..nout).collect(), (nout..4).collect());
        let Svd { u, s, vh } = t.svd_compact(engine, rows, cols).unwrap();
        let expected = p.svd_compact(engine, &ir, &ic).unwrap();
        assert_same!(engine, u, expected.u, "zero-length u");
        assert_same!(engine, s, expected.s, "zero-length s");
        assert_same!(engine, vh, expected.vh, "zero-length vh");
        assert!(
            matches!(t.exp(engine, rows, cols), Err(Error::Operation(e)) if matches!(*e, OperationError::SpaceMismatch { message } if message.contains("exp requires an endomorphism"))),
            "a vector view is not an endomorphism"
        );
    }
}

/// A fermionic square split that crosses the codomain/domain boundary:
/// `[0, 3] <- [2, 1]` bends one leg each way, so the permute carries the
/// fermionic signs and dual flips into `exp`, `inv` and `eig_full`.
#[test]
fn fermionic_boundary_crossing_square_roles_equal_the_permute_composition() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = host_runtime();
    let (v, w) = fz2_u1_legs();
    let t: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&v, &w], [&v, &w], 1555).unwrap();
    let (rows, cols) = ([0, 3], [2, 1]);
    let p = t.permute(engine, &rows, &cols).unwrap();
    assert_eq!(p.codomain(), p.domain(), "the crossing split is square");
    let (ir, ic) = ([0, 1], [2, 3]);
    assert_same!(
        engine,
        t.exp(engine, &rows, &cols).unwrap(),
        p.exp(engine, &ir, &ic).unwrap(),
        "fz2u1 exp"
    );
    let inv = t.inv(engine, &rows, &cols).unwrap();
    assert_same!(engine, inv, p.inv(engine, &ir, &ic).unwrap(), "fz2u1 inv");
    let id = TensorMap::isomorphism(engine, p.runtime(), &p.codomain(), &p.domain()).unwrap();
    assert_close!(
        engine,
        p.compose(engine, &inv).unwrap(),
        id,
        1e-8,
        "fz2u1 inv"
    );
    let Eig { d, v: vectors } = t.eig_full(engine, &rows, &cols).unwrap();
    let expected = p.eig_full(engine, &ir, &ic).unwrap();
    assert_same!(engine, d, expected.d, "fz2u1 eig d");
    assert_same!(engine, vectors, expected.v, "fz2u1 eig v");
    assert_close!(
        engine,
        p.compose(engine, &vectors).unwrap(),
        vectors.compose(engine, &d).unwrap(),
        1e-8,
        "fz2u1 eig"
    );
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
