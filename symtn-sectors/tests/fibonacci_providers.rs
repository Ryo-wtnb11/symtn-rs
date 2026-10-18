//! Fibonacci is `TableRule::fibonacci()`, symtn's one Fibonacci provider.
//! These tests hold its values through symtn's traits against the values
//! symtn exposed before (the removed closed form and table adapters), the
//! trait-default A/B symbols, the codec and the domain checks. The tables'
//! own coherence tests (pentagon, hexagon, unitarity, ribbon) live in
//! `fusion-symbols`.

use std::panic::{self, AssertUnwindSafe};

use num_complex::{Complex32, Complex64};
use symtn_sectors::{
    CheckedFusionAlgebra, FusionAlgebraError, FusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorCodec, SectorId, TableRule, TableSector,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

const SECTORS: [SectorId; 2] = [SectorId::new(0), SectorId::new(1)];

/// Every value both providers exposed at symtn 957ccc05 (then TeNeT, with the
/// table provider in `symtn-category-data`), as f64 bits. Printed by
/// `fusion-symbols` 0.3.0 `tests/fixtures/extract_tenet_values.rs` on macOS
/// aarch64; symtn 50a8d4f6 prints the same 196 rows.
const VALUES_957CCC05: &str = include_str!("fixtures/symtn-957ccc05-fibonacci-values.txt");

/// CategoryData.jl 0.3.6 / TensorKitSectors 0.3.9 values for the table
/// (`fixtures/generate_fib_categorydata_fixture.jl`).
const CATEGORYDATA: &str = include_str!("fixtures/fib-categorydata-v0.1.3.txt");

fn table() -> TableRule {
    TableRule::fibonacci()
}

fn bits(z: Complex64) -> String {
    format!("{:016x} {:016x}", z.re.to_bits(), z.im.to_bits())
}

/// The rows of `VALUES_957CCC05` for one provider, in the same order.
fn dump<R>(tag: &str, rule: &R) -> Vec<String>
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64>,
{
    let mut rows = Vec::new();
    for a in SECTORS {
        let i = a.id();
        rows.push(format!("{tag} dual {i} {}", rule.dual(a).id()));
        rows.push(format!("{tag} dim {i} {}", bits(rule.dim_scalar(a))));
        rows.push(format!(
            "{tag} inv_dim {i} {}",
            bits(rule.inv_dim_scalar(a))
        ));
        rows.push(format!(
            "{tag} sqrt_dim {i} {}",
            bits(rule.sqrt_dim_scalar(a))
        ));
        rows.push(format!(
            "{tag} inv_sqrt_dim {i} {}",
            bits(rule.inv_sqrt_dim_scalar(a))
        ));
        rows.push(format!(
            "{tag} fs {i} {}",
            bits(rule.frobenius_schur_phase_scalar(a))
        ));
        rows.push(format!("{tag} twist {i} {}", bits(rule.twist_scalar(a))));
    }
    for a in SECTORS {
        for b in SECTORS {
            let channels: Vec<String> = rule
                .fusion_channels(a, b)
                .iter()
                .map(|c| c.id().to_string())
                .collect();
            rows.push(format!(
                "{tag} fusion {} {} {}",
                a.id(),
                b.id(),
                channels.join(",")
            ));
            for c in SECTORS {
                let (i, j, k) = (a.id(), b.id(), c.id());
                rows.push(format!("{tag} N {i} {j} {k} {}", rule.nsymbol(a, b, c)));
                rows.push(format!(
                    "{tag} R {i} {j} {k} {}",
                    bits(rule.r_symbol_scalar(a, b, c))
                ));
            }
        }
    }
    for a in SECTORS {
        for b in SECTORS {
            for c in SECTORS {
                for d in SECTORS {
                    for e in SECTORS {
                        for f in SECTORS {
                            let ids = [a, b, c, d, e, f].map(|s| s.id().to_string());
                            rows.push(format!(
                                "{tag} F {} {}",
                                ids.join(" "),
                                bits(rule.f_symbol_scalar(a, b, c, d, e, f))
                            ));
                        }
                    }
                }
            }
        }
    }
    rows
}

/// Distance in representable f64s between two hex bit patterns.
fn ulps(actual: &str, recorded: &str) -> u64 {
    let key = |hex: &str| {
        let bits = u64::from_str_radix(hex, 16).unwrap() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    key(actual).abs_diff(key(recorded))
}

/// The Fibonacci golden. symtn's previous table adapter is the same tables
/// (byte-identical F/R text in fusion-symbols 0.3.0 and 0.4.0) and the same
/// derivations, so `TableRule::fibonacci()` reproduces its 98 rows of
/// `VALUES_957CCC05` bit for bit.
#[test]
fn every_value_matches_the_recorded_table_golden_bit_for_bit() {
    let actual = dump("table", &table());
    let recorded: Vec<&str> = VALUES_957CCC05
        .lines()
        .filter(|line| line.starts_with("table "))
        .collect();
    assert_eq!(recorded.len(), 98);
    assert_eq!(actual, recorded);
}

/// The removed closed form's 98 rows remain an oracle for the table: every
/// row is equal, except the entries pinned here at their exact (re, im) ulp
/// distance from the recorded closed-form bits. The closed form evaluated
/// `1/φ`, `1/√φ`, `φ` and `cispi` in f64; the table is correctly rounded.
#[test]
fn every_value_matches_the_removed_closed_form_up_to_pinned_ulps() {
    const MOVED: [(&str, (u64, u64)); 7] = [
        ("closed dim 1", (1, 0)),
        ("closed inv_dim 1", (1, 0)),
        ("closed twist 1", (2, 2)),
        ("closed R 1 1 0", (1, 1)),
        ("closed R 1 1 1", (2, 1)),
        ("closed F 1 1 1 1 0 0", (1, 0)),
        ("closed F 1 1 1 1 1 1", (1, 0)),
    ];
    let actual = dump("closed", &table());
    let recorded: Vec<&str> = VALUES_957CCC05
        .lines()
        .filter(|line| line.starts_with("closed "))
        .collect();
    assert_eq!(actual.len(), 98);
    assert_eq!(actual.len(), recorded.len());

    let mut moved = Vec::new();
    for (actual, recorded) in actual.iter().zip(&recorded) {
        if actual == recorded {
            continue;
        }
        let fields =
            |row: &str| -> Vec<String> { row.split_whitespace().map(str::to_owned).collect() };
        let (a, r) = (fields(actual), fields(recorded));
        let n = a.len();
        assert_eq!(a[..n - 2], r[..n - 2], "{actual} vs recorded {recorded}");
        let prefix = r[..n - 2].join(" ");
        moved.push((
            prefix,
            (ulps(&a[n - 2], &r[n - 2]), ulps(&a[n - 1], &r[n - 1])),
        ));
    }
    let expected: Vec<(String, (u64, u64))> = MOVED
        .iter()
        .map(|(prefix, distance)| (prefix.to_string(), *distance))
        .collect();
    assert_eq!(moved, expected);
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

/// The table's derived values and symtn's trait-default A/B symbols against
/// CategoryData.jl / TensorKitSectors, under the workspace tolerance rule.
#[test]
fn derived_and_ab_symbols_match_categorydata_and_tensorkitsectors() {
    let fib = table();
    for a in SECTORS {
        let id = [a.id()];
        let close = |what, actual, tag| {
            numerics::assert_close(what, actual, categorydata_value(tag, &id), 4)
        };
        close("dim", fib.dim_scalar(a), "dim");
        close(
            "frobenius-schur",
            fib.frobenius_schur_phase_scalar(a),
            "frobeniusschur",
        );
        close("twist", fib.twist_scalar(a), "twist");
    }
    for a in SECTORS {
        for b in SECTORS {
            for c in SECTORS {
                let labels = [a.id(), b.id(), c.id()];
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
