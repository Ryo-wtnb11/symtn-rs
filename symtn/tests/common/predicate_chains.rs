// The TensorKit predicates and projections as the public chains symtn spells
// them with (#1557). Included with `include!` so the integration tests and the
// crate's own unit tests read one spelling; `TensorMap` must be in scope at the
// call site, and the includer implements `ChainCoefficient` for its payloads. Each macro evaluates to a `bool` (or the projected tensor) and
// panics on an operation error.

/// The exact `±1` and `±1/2` coefficients the chains need. Integration tests
/// implement it with `common/predicate_chain_coefficients.rs`; the crate's own
/// unit tests implement it over the crate-private scalar trait.
#[allow(dead_code)]
trait ChainCoefficient: Copy {
    fn real(value: f64) -> Self;
}

/// `‖t - t†‖ <= tol · max(‖t‖, 1)`; a non-endomorphism is `false`.
#[allow(unused_macros)]
macro_rules! is_hermitian {
    ($engine:ident, $tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        t.codomain() == t.domain() && {
            let t_adjoint = &t.adjoint($engine).unwrap();
            t.axpby($engine,
                ChainCoefficient::real(1.0),
                t_adjoint,
                ChainCoefficient::real(-1.0),
            )
            .unwrap()
            .norm($engine, 2.0)
            .unwrap()
                <= ($tol) * t.norm($engine, 2.0).unwrap().max(1.0)
        }
    }};
}

/// `‖t + t†‖ <= tol · max(‖t‖, 1)`; a non-endomorphism is `false`.
#[allow(unused_macros)]
macro_rules! is_antihermitian {
    ($engine:ident, $tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        t.codomain() == t.domain() && {
            let adjoint = t.adjoint($engine).unwrap();
            t.axpby($engine,
                ChainCoefficient::real(1.0),
                &adjoint,
                ChainCoefficient::real(1.0),
            )
            .unwrap()
            .norm($engine, 2.0)
            .unwrap()
                <= ($tol) * t.norm($engine, 2.0).unwrap().max(1.0)
        }
    }};
}

/// `‖t†t - id‖ <= tol · max(‖t†t‖, 1)` on the domain.
#[allow(unused_macros)]
macro_rules! is_isometric {
    ($engine:ident, $tensor:expr, $tol:expr) => {{
        let t: &TensorMap<_, _> = &$tensor;
        let gram = t.adjoint($engine).unwrap().compose($engine, t).unwrap();
        let identity = TensorMap::isomorphism($engine, t.runtime(), &t.domain(), &t.domain()).unwrap();
        gram.axpby($engine,
            ChainCoefficient::real(1.0),
            &identity,
            ChainCoefficient::real(-1.0),
        )
        .unwrap()
        .norm($engine, 2.0)
        .unwrap()
            <= ($tol) * gram.norm($engine, 2.0).unwrap().max(1.0)
    }};
}

/// Isometric in both directions.
#[allow(unused_macros)]
macro_rules! is_unitary {
    ($engine:ident, $tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        is_isometric!($engine, t, $tol) && is_isometric!($engine, t.adjoint($engine).unwrap(), $tol)
    }};
}

/// Hermitian, and every `eigh_vals` eigenvalue strictly above
/// `tol · max(‖t‖, 1)`.
#[allow(unused_macros)]
macro_rules! is_posdef {
    ($engine:ident, $tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        is_hermitian!($engine, t, $tol) && {
            let threshold = ($tol) * t.norm($engine, 2.0).unwrap().max(1.0);
            t.eigh_vals($engine, &codomain_axes(&t), &domain_axes(&t), symtn::typed::HermitianTol::DEFAULT)
                .unwrap()
                .iter()
                .flat_map(|spectrum| spectrum.values.iter())
                .all(|&value| value > threshold)
        }
    }};
}

/// [`is_posdef!`] for a compact spectrum factor: its stored values are its
/// Hermitian eigenvalues, so `diagview` replaces `eigh_vals`; `$re` reads the
/// real part of one stored value as `f64`.
#[allow(unused_macros)]
macro_rules! is_posdef_compact {
    ($engine:ident, $tensor:expr, $tol:expr, $re:expr) => {{
        let t = &$tensor;
        is_hermitian!($engine, t, $tol) && {
            let threshold = ($tol) * t.norm($engine, 2.0).unwrap().max(1.0);
            let re = $re;
            t.diagview($engine)
                .unwrap()
                .iter()
                .flat_map(|spectrum| spectrum.values.iter())
                .all(|&value| re(value) > threshold)
        }
    }};
}

/// `(t + t†)/2`.
#[allow(unused_macros)]
macro_rules! project_hermitian {
    ($engine:ident, $tensor:expr) => {{
        let t = &$tensor;
        t.adjoint($engine).and_then(|adjoint| {
            t.axpby($engine,
                ChainCoefficient::real(0.5),
                &adjoint,
                ChainCoefficient::real(0.5),
            )
        })
    }};
}

/// `(t - t†)/2`.
#[allow(unused_macros)]
macro_rules! project_antihermitian {
    ($engine:ident, $tensor:expr) => {{
        let t = &$tensor;
        t.adjoint($engine).and_then(|adjoint| {
            t.axpby($engine,
                ChainCoefficient::real(0.5),
                &adjoint,
                ChainCoefficient::real(-0.5),
            )
        })
    }};
}

/// The receiver's own split as leg roles: `rows = 0..nout`.
#[allow(dead_code)]
fn codomain_axes<R, D, S>(t: &TensorMap<R, D, S>) -> Vec<usize> {
    (0..t.codomain_rank()).collect()
}

/// The receiver's own split as leg roles: `cols = nout..rank`.
#[allow(dead_code)]
fn domain_axes<R, D, S>(t: &TensorMap<R, D, S>) -> Vec<usize> {
    (t.codomain_rank()..t.rank()).collect()
}
