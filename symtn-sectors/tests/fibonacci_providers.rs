//! Fibonacci is `TableRule::fibonacci()`, symtn's one Fibonacci provider.
//! These tests hold its values through symtn's traits against
//! CategoryData.jl / TensorKitSectors and hand relations, the trait-default
//! A/B symbols, the codec and the domain checks. The tables' own coherence
//! tests (pentagon, hexagon, unitarity, ribbon) live in `fusion-symbols`.

use std::panic::{self, AssertUnwindSafe};

use num_complex::{Complex32, Complex64};
use symtn_sectors::{
    CheckedFusionAlgebra, FusionAlgebraError, FusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorCodec, SectorId, TableRule, TableSector,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

const SECTORS: [SectorId; 2] = [SectorId::new(0), SectorId::new(1)];

/// CategoryData.jl 0.3.6 / TensorKitSectors 0.3.9 values for the table
/// (`fixtures/generate_fib_categorydata_fixture.jl`).
const CATEGORYDATA: &str = include_str!("fixtures/fib-categorydata-v0.1.3.txt");

fn table() -> TableRule {
    TableRule::fibonacci()
}

fn bits(z: Complex64) -> String {
    format!("{:016x} {:016x}", z.re.to_bits(), z.im.to_bits())
}

/// `tag` and one-based labels -> value, as emitted by the Julia generator.
fn categorydata_value(tag: &str, labels: &[usize]) -> Complex64 {
    let wanted: Vec<String> = labels.iter().map(|id| (id + 1).to_string()).collect();
    CATEGORYDATA
        .lines()
        .filter(|line| !line.trim_start().starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|fields| fields[0] == tag && fields[1..=labels.len()] == wanted[..])
        .map(|fields| {
            let value = &fields[labels.len() + 1..];
            Complex64::new(
                value[0].parse().unwrap(),
                value.get(1).map_or(0.0, |im| im.parse().unwrap()),
            )
        })
        .unwrap_or_else(|| panic!("fixture is missing {tag}{labels:?}"))
}

/// The table's symbols, derived values and symtn's trait-default A/B symbols
/// against CategoryData.jl / TensorKitSectors, under the workspace tolerance
/// rule; duals, N and fusion channels exactly.
#[test]
fn symbols_derived_and_ab_values_match_categorydata_and_tensorkitsectors() {
    let fib = table();
    for a in SECTORS {
        let id = [a.id()];
        let close = |what, actual, tag| {
            numerics::assert_close(what, actual, categorydata_value(tag, &id), 4)
        };
        assert_eq!(
            fib.dual(a).id() as f64 + 1.0,
            categorydata_value("dual", &id).re,
            "dual {a:?}"
        );
        close("dim", fib.dim_scalar(a), "dim");
        // Hand relations tie the derived dimensions to `dim`.
        let one = Complex64::new(1.0, 0.0);
        let sqrt = fib.sqrt_dim_scalar(a);
        numerics::assert_close("sqrt_dim^2", sqrt * sqrt, fib.dim_scalar(a), 4);
        numerics::assert_close(
            "dim * inv_dim",
            fib.dim_scalar(a) * fib.inv_dim_scalar(a),
            one,
            4,
        );
        numerics::assert_close(
            "sqrt_dim * inv_sqrt_dim",
            sqrt * fib.inv_sqrt_dim_scalar(a),
            one,
            4,
        );
        close(
            "frobenius-schur",
            fib.frobenius_schur_phase_scalar(a),
            "frobeniusschur",
        );
        close("twist", fib.twist_scalar(a), "twist");
    }
    for a in SECTORS {
        for b in SECTORS {
            let channels: Vec<SectorId> = fib.fusion_channels(a, b).iter().copied().collect();
            for c in SECTORS {
                let labels = [a.id(), b.id(), c.id()];
                let n = categorydata_value("N", &labels).re as usize;
                assert_eq!(fib.nsymbol(a, b, c), n, "N{labels:?}");
                assert_eq!(channels.contains(&c), n > 0, "channels{labels:?}");
                numerics::assert_close(
                    "R",
                    fib.r_symbol_scalar(a, b, c),
                    categorydata_value("R", &labels),
                    4,
                );
                numerics::assert_close(
                    "A",
                    fib.a_symbol_scalar(a, b, c),
                    categorydata_value("A", &labels),
                    4,
                );
                numerics::assert_close(
                    "B",
                    fib.b_symbol_scalar(a, b, c),
                    categorydata_value("B", &labels),
                    4,
                );
            }
        }
    }
    for labels in (0..64usize).map(|index| [5, 4, 3, 2, 1, 0].map(|axis| (index >> axis) & 1)) {
        let [a, b, c, d, e, f] = labels.map(SectorId::new);
        numerics::assert_close(
            "F",
            fib.f_symbol_scalar(a, b, c, d, e, f),
            categorydata_value("F", &labels),
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

#[test]
fn table_codec_round_trips_every_object_with_the_recorded_ids() {
    let fib = table();
    for id in [0, 1] {
        let label = TableSector::new(id as u16 + 1);
        assert_eq!(fib.encode_sector(&label), Ok(SectorId::new(id)));
        assert_eq!(fib.decode_sector(SectorId::new(id)), Ok(label));
    }
    for a in SECTORS {
        for b in SECTORS {
            for channel in fib.fusion_channels(a, b) {
                let object = fib.decode_sector(channel).unwrap();
                assert_eq!(fib.encode_sector(&object), Ok(channel));
            }
        }
    }
    assert_eq!(
        fib.decode_sector(SectorId::new(2)),
        Err(FusionAlgebraError::InvalidSector {
            sector: SectorId::new(2)
        })
    );
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
            assert!(fib.try_nsymbol(ok, ok, invalid).is_err());
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

#[test]
fn table_checked_and_infallible_fusion_queries_agree() {
    let fib = table();
    for a in SECTORS {
        assert_eq!(fib.try_dual_sector(a), Ok(fib.dual(a)));
        for b in SECTORS {
            assert_eq!(
                fib.try_fusion_channels(a, b).unwrap().as_slice(),
                fib.fusion_channels(a, b).as_slice()
            );
            for c in SECTORS {
                assert_eq!(fib.try_nsymbol(a, b, c), Ok(fib.nsymbol(a, b, c)));
            }
        }
    }
}
