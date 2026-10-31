//! Fibonacci is `TableRule::fibonacci()`, symtn's one Fibonacci provider.
//! These tests hold its values through symtn's traits against
//! TensorKitSectors' closed forms and hand relations, the trait-default
//! A/B symbols, the codec and the domain checks. The tables' own coherence
//! tests (pentagon, hexagon, unitarity, ribbon) live in `fusion-symbols`.

use std::panic::{self, AssertUnwindSafe};

use num_complex::Complex64;
use symtn_sectors::{
    CheckedFusionAlgebra, FusionAlgebraError, FusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorId, TableRule,
};

use crate::numerics;

const SECTORS: [SectorId; 2] = [SectorId::new(0), SectorId::new(1)];

fn table() -> TableRule {
    TableRule::fibonacci()
}

fn bits(z: Complex64) -> String {
    format!("{:016x} {:016x}", z.re.to_bits(), z.im.to_bits())
}

const PHI: f64 = std::f64::consts::GOLDEN_RATIO;

/// TensorKitSectors 0.3.9 `FibonacciAnyon` (`src/anyons.jl`: `Nsymbol`,
/// `Fsymbol`, `Rsymbol`, `dim`) on ids `(1, τ) = (0, 1)`, the same closed
/// forms fusion-symbols' `tests/tables_reference.rs` holds its table to.
/// `N^{ab}_c = 0` exactly when one of `a, b, c` is `τ`.
fn n(a: usize, b: usize, c: usize) -> bool {
    a + b + c != 1
}

fn dim(a: usize) -> f64 {
    if a == 0 {
        1.0
    } else {
        PHI
    }
}

/// `F^{τττ}_τ = [[1/φ, 1/√φ], [1/√φ, -1/φ]]` in the basis `(1, τ)`; every
/// other admissible F is `1`, an inadmissible one `0`.
fn f(a: usize, b: usize, c: usize, d: usize, e: usize, x: usize) -> f64 {
    if !(n(a, b, e) && n(e, c, d) && n(b, c, x) && n(a, x, d)) {
        return 0.0;
    }
    match ([a, b, c, d] == [1; 4], e, x) {
        (true, 0, 0) => 1.0 / PHI,
        (true, 1, 1) => -1.0 / PHI,
        (true, _, _) => 1.0 / PHI.sqrt(),
        _ => 1.0,
    }
}

/// `R^{ττ}_1 = e^{4πi/5}`, `R^{ττ}_τ = e^{-3πi/5}`, `1` with a unit leg.
fn r(a: usize, b: usize, c: usize) -> Complex64 {
    let pi = std::f64::consts::PI;
    match (n(a, b, c), a * b, c) {
        (false, _, _) => Complex64::new(0.0, 0.0),
        (true, 0, _) => Complex64::new(1.0, 0.0),
        (true, _, 0) => Complex64::cis(0.8 * pi),
        (true, _, _) => Complex64::cis(-0.6 * pi),
    }
}

/// The table's symbols, derived values and symtn's trait-default A/B symbols
/// against the TensorKitSectors closed forms above, under the workspace
/// tolerance rule; duals, N and fusion channels exactly. Derived values use
/// TensorKitSectors' generic definitions (every label is self-dual):
/// `frobenius_schur_phase(a) = sign(F^{aaa}_a[1, 1])`,
/// `twist(a) = Σ_b d_b / d_a R^{aa}_b`,
/// `A^{ab}_c = √(d_a d_b / d_c) conj(κ_a F^{a a b}_b[1, c])` and
/// `B^{ab}_c = √(d_a d_b / d_c) F^{a b b}_a[c, 1]`.
#[test]
fn symbols_derived_and_ab_values_match_tensorkitsectors_closed_forms() {
    let fib = table();
    let ids = [0, 1];
    for a in ids {
        let s = SECTORS[a];
        let real = |x: f64| Complex64::new(x, 0.0);
        assert_eq!(fib.dual(s), s, "dual {s:?}");
        numerics::assert_close("dim", fib.dim_scalar(s), real(dim(a)), 4);
        // Hand relations tie the derived dimensions to `dim`.
        let one = real(1.0);
        let sqrt = fib.sqrt_dim_scalar(s);
        numerics::assert_close("sqrt_dim^2", sqrt * sqrt, fib.dim_scalar(s), 4);
        numerics::assert_close(
            "dim * inv_dim",
            fib.dim_scalar(s) * fib.inv_dim_scalar(s),
            one,
            4,
        );
        numerics::assert_close(
            "sqrt_dim * inv_sqrt_dim",
            sqrt * fib.inv_sqrt_dim_scalar(s),
            one,
            4,
        );
        let fs = f(a, a, a, a, 0, 0).signum();
        numerics::assert_close(
            "frobenius-schur",
            fib.frobenius_schur_phase_scalar(s),
            real(fs),
            4,
        );
        let twist: Complex64 = ids.iter().map(|&b| r(a, a, b) * (dim(b) / dim(a))).sum();
        numerics::assert_close("twist", fib.twist_scalar(s), twist, 4);
    }
    // The twist formula evaluated by hand: `θ_τ = e^{4πi/5}/φ + e^{-3πi/5}
    // = e^{-4πi/5}` (checked against TensorKitSectors 0.3.9 `twist`).
    numerics::assert_close(
        "twist(τ)",
        fib.twist_scalar(SECTORS[1]),
        Complex64::cis(-0.8 * std::f64::consts::PI),
        4,
    );
    for a in ids {
        for b in ids {
            let channels: Vec<SectorId> = fib
                .fusion_channels(SECTORS[a], SECTORS[b])
                .iter()
                .copied()
                .collect();
            for c in ids {
                let [sa, sb, sc] = [a, b, c].map(|id| SECTORS[id]);
                let labels = [a, b, c];
                assert_eq!(
                    fib.nsymbol(sa, sb, sc),
                    usize::from(n(a, b, c)),
                    "N{labels:?}"
                );
                assert_eq!(channels.contains(&sc), n(a, b, c), "channels{labels:?}");
                numerics::assert_close("R", fib.r_symbol_scalar(sa, sb, sc), r(a, b, c), 4);
                let scale = (dim(a) * dim(b) / dim(c)).sqrt();
                let fs = f(a, a, a, a, 0, 0).signum();
                let want_a = scale * fs * f(a, a, b, b, 0, c);
                let want_b = scale * f(a, b, b, a, c, 0);
                numerics::assert_close(
                    "A",
                    fib.a_symbol_scalar(sa, sb, sc),
                    Complex64::new(want_a, 0.0),
                    4,
                );
                numerics::assert_close(
                    "B",
                    fib.b_symbol_scalar(sa, sb, sc),
                    Complex64::new(want_b, 0.0),
                    4,
                );
            }
        }
    }
    for labels in (0..64usize).map(|index| [5, 4, 3, 2, 1, 0].map(|axis| (index >> axis) & 1)) {
        let [a, b, c, d, e, x] = labels;
        let [sa, sb, sc, sd, se, sx] = labels.map(SectorId::new);
        numerics::assert_close(
            "F",
            fib.f_symbol_scalar(sa, sb, sc, sd, se, sx),
            Complex64::new(f(a, b, c, d, e, x), 0.0),
            4,
        );
    }
}

/// `CanonicalUnitFusionRule` promises exact identity unitors, so the
/// unit-associated F symbols are asserted bitwise `1 + 0i`.
#[test]
fn unit_associators_are_bitwise_one() {
    fn check<R: MultiplicityFreeFusionSymbols<Scalar = Complex64>>(fib: &R) {
        let unit = fib.vacuum();
        for a in SECTORS {
            for b in SECTORS {
                for c in fib.fusion_channels(a, b) {
                    for value in [
                        fib.f_symbol_scalar(unit, a, b, c, a, c),
                        fib.f_symbol_scalar(a, unit, b, c, a, b),
                        fib.f_symbol_scalar(a, b, unit, c, c, b),
                    ] {
                        assert_eq!(bits(value), bits(Complex64::new(1.0, 0.0)));
                    }
                }
            }
        }
    }
    check(&table());
}

/// Issue #1737: the infallible symbols have no checked twin, so an
/// out-of-domain id panics with the typed domain error,
/// never aliasing another entry.
#[test]
fn infallible_symbols_panic_with_the_domain_error_on_out_of_domain_ids() {
    fn assert_domain_panic(what: &str, invalid: SectorId, call: impl FnOnce() -> Complex64) {
        let payload = panic::catch_unwind(AssertUnwindSafe(call))
            .expect_err(&format!("{what} returned a value for {invalid:?}"));
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        let expected = FusionAlgebraError::InvalidSector { sector: invalid }.to_string();
        assert!(
            message.contains(what) && message.contains(&expected),
            "{what} on {invalid:?} panicked with {message:?}"
        );
    }

    fn check<R>(fib: &R)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra,
    {
        let ok = SECTORS[1];
        for invalid in [SectorId::new(2), SectorId::new(usize::MAX)] {
            assert_eq!(
                fib.try_nsymbol(ok, ok, invalid),
                Err(FusionAlgebraError::InvalidSector { sector: invalid })
            );
            for slot in 0..6 {
                let mut ids = [ok; 6];
                ids[slot] = invalid;
                let [a, b, c, d, e, f] = ids;
                assert_domain_panic("f_symbol_scalar", invalid, || {
                    fib.f_symbol_scalar(a, b, c, d, e, f)
                });
            }
            for slot in 0..3 {
                let mut ids = [SECTORS[0]; 3];
                ids[slot] = invalid;
                let [a, b, c] = ids;
                assert_domain_panic("r_symbol_scalar", invalid, || fib.r_symbol_scalar(a, b, c));
            }
            assert_domain_panic("dim_scalar", invalid, || fib.dim_scalar(invalid));
            assert_domain_panic("inv_dim_scalar", invalid, || fib.inv_dim_scalar(invalid));
            assert_domain_panic("sqrt_dim_scalar", invalid, || fib.sqrt_dim_scalar(invalid));
            assert_domain_panic("inv_sqrt_dim_scalar", invalid, || {
                fib.inv_sqrt_dim_scalar(invalid)
            });
            assert_domain_panic("twist_scalar", invalid, || fib.twist_scalar(invalid));
            assert_domain_panic("frobenius_schur_phase_scalar", invalid, || {
                fib.frobenius_schur_phase_scalar(invalid)
            });
        }
    }
    check(&table());
}
