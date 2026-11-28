//! What `CategoryRule<TableCategory, EnumeratedCodec<_>>` owns over
//! `fusion-symbols`' tables: admission of every shipped entry, the braiding
//! and fusion styles it reports, the codec, the domain checks, the per-entry
//! identity and the fusion-only R boundary. The tables' own
//! coherence tests (pentagon, hexagon, unitarity) live in `fusion-symbols`.

use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};

use fusion_symbols::{
    category::{BraidingStyle, FusionCategory},
    table::TableCategory,
};
use symtn_sectors::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorCodec, SectorId,
};

use crate::tables::{ising_object, table_by_uuid, table_ising, table_object, TableCategoryRule};

/// Rep(S₃), `anyonwiki_code [3,1,0,2,2,0,1]`: fusion only, non-Abelian.
const REP_S3: &str = "37fa4e9c-97c6-11f1-8ce9-19309dfc597c";
/// Semion, `[2,1,0,1,1,1,1]`: pointed, `F^{sss}_s = −1`, `R^{ss}_𝟙 = i`.
const SEMION: &str = "37fa4d16-97c6-11f1-9c13-c70bad3f9e94";

fn ids(rule: &TableCategoryRule) -> impl Iterator<Item = SectorId> {
    let rank = rule.category().object_count().unwrap().unwrap();
    (0..usize::try_from(rank).unwrap()).map(SectorId::new)
}

#[test]
fn styles_follow_the_braiding_and_the_exact_fusion_table() {
    let rep_s3 = table_by_uuid(REP_S3);
    assert_eq!(rep_s3.braiding_style(), BraidingStyleKind::NoBraiding);
    assert_eq!(rep_s3.fusion_style(), FusionStyleKind::Simple);

    for rule in [crate::tables::table_fibonacci(), table_ising()] {
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Anyonic);
        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
    }

    let semion = table_by_uuid(SEMION);
    assert_eq!(semion.fusion_style(), FusionStyleKind::Unique);
    assert_eq!(semion.braiding_style(), BraidingStyleKind::Anyonic);
    // Semion's nontrivial associator prevents the direct Unique braid.
    assert!(!semion.has_trivial_associator_gauge());
}

#[test]
fn every_shipped_entry_constructs_with_its_own_identity() {
    let mut identities = HashSet::new();
    for uuid in TableCategory::shipped_uuids() {
        let rule = table_by_uuid(uuid);
        assert_eq!(rule.category().uuid(), uuid);
        assert_eq!(
            rule.braiding_style().has_braiding(),
            TableCategory::by_uuid(uuid).unwrap().braiding_style() != BraidingStyle::None
        );
        assert_eq!(rule.rule_identity(), table_by_uuid(uuid).rule_identity());
        assert!(identities.insert(rule.rule_identity()), "{uuid}");
    }
    assert_eq!(identities.len(), 74);
    for closed in [
        symtn_sectors::ZNFusionRule::new(2).unwrap().rule_identity(),
        symtn_sectors::ZNFusionRule::new(3).unwrap().rule_identity(),
    ] {
        assert!(!identities.contains(&closed));
    }
}

#[test]
fn every_shipped_entry_reports_the_category_style_and_certificates() {
    let mut trivial = 0;
    let mut nontrivial = 0;
    let mut symmetric = 0;
    for uuid in TableCategory::shipped_uuids() {
        let category = TableCategory::by_uuid(uuid).unwrap();
        let rule = table_by_uuid(uuid);
        assert!(category.has_canonical_unit(), "{uuid}");
        assert_eq!(
            rule.has_trivial_associator_gauge(),
            category.has_trivial_associator_gauge(),
            "{uuid}"
        );
        let expected = match category.braiding_style() {
            BraidingStyle::None => BraidingStyleKind::NoBraiding,
            BraidingStyle::Bosonic => BraidingStyleKind::Bosonic,
            BraidingStyle::Fermionic => BraidingStyleKind::Fermionic,
            BraidingStyle::Anyonic => BraidingStyleKind::Anyonic,
            other => panic!("{uuid}: unexpected braiding style {other:?}"),
        };
        assert_eq!(rule.braiding_style(), expected, "{uuid}");
        trivial += usize::from(category.has_trivial_associator_gauge());
        nontrivial += usize::from(!category.has_trivial_associator_gauge());
        symmetric += usize::from(expected.is_symmetric());
    }
    // Both certificate outcomes and a symmetric category distinguish forwarding
    // from the previous unconditional false / Anyonic overrides.
    assert!(trivial > 0 && nontrivial > 0 && symmetric > 0);
}

#[test]
fn codec_round_trips_and_rejects_objects_of_another_table() {
    let rep_s3 = table_by_uuid(REP_S3);
    assert_eq!(rep_s3.vacuum(), SectorId::new(0));
    assert_eq!(
        rep_s3.decode_sector(rep_s3.vacuum()),
        Ok(table_object(&rep_s3, 1))
    );
    for sector in ids(&rep_s3) {
        let label = rep_s3.decode_sector(sector).unwrap();
        assert_eq!(usize::from(label.index()), sector.id() + 1);
        assert_eq!(rep_s3.encode_sector(&label), Ok(sector));
        for right in ids(&rep_s3) {
            for coupled in rep_s3.fusion_channels(sector, right) {
                let label = rep_s3.decode_sector(coupled).unwrap();
                assert_eq!(rep_s3.encode_sector(&label), Ok(coupled));
            }
        }
    }
    for label in [1, 3] {
        assert!(matches!(
            rep_s3.encode_sector(&ising_object(label)),
            Err(FusionAlgebraError::UnrepresentableSectorLabel { rule, .. })
                if rule == rep_s3.rule_identity()
        ));
    }
    let invalid = SectorId::new(3);
    assert_eq!(
        rep_s3.decode_sector(invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
    assert_eq!(
        rep_s3.try_fusion_channels(invalid, SectorId::new(0)),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
}

#[test]
fn fusion_only_entry_has_f_data_and_panics_on_r_and_twist() {
    let rep_s3 = table_by_uuid(REP_S3);
    // The two-dimensional object fuses with itself into all three objects.
    let two = SectorId::new(2);
    assert_eq!(rep_s3.dim_scalar(two).re, 2.0);
    assert_eq!(rep_s3.fusion_channels(two, two).len(), 3);
    let f = rep_s3.f_symbol_scalar(two, two, two, two, SectorId::new(0), SectorId::new(0));
    assert!(
        (f.re - 0.5).abs() <= 4.0 * f64::EPSILON && f.im == 0.0,
        "{f}"
    );

    let panics = |what: &str, call: &dyn Fn()| {
        let message = panic::catch_unwind(AssertUnwindSafe(call)).unwrap_err();
        let message = message
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert_eq!(
            message,
            format!("{what} precondition violated: the category has no braiding")
        );
    };
    let unit = SectorId::new(0);
    panics("r_symbol", &|| {
        rep_s3.r_symbol_scalar(unit, unit, unit);
    });
    panics("twist", &|| {
        rep_s3.twist_scalar(unit);
    });
}

/// Ising's derived quantities are gauge invariant, so they equal
/// TensorKitSectors 0.3.9 `IsingAnyon` (`src/anyons.jl`: `dim`, the generic
/// `frobenius_schur_phase` and `twist`) under the object map
/// TensorKit `(I, σ, ψ)` → AnyonWiki `(1, 3, 2)`.
#[test]
fn ising_dims_phases_and_twists_match_tensorkit() {
    use num_complex::Complex64;
    let ising = table_ising();
    let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);
    let one = Complex64::new(1.0, 0.0);
    for (label, dim, twist) in [
        (1, one, one),
        (2, one, -one),
        (3, Complex64::new(2f64.sqrt(), 0.0), cispi(1.0 / 8.0)),
    ] {
        let sector = ising.encode_sector(&ising_object(label)).unwrap();
        let close = |what: &str, actual: Complex64, expected: Complex64| {
            assert!(
                (actual - expected).norm() <= 32.0 * f64::EPSILON,
                "{what}({label}) {actual} != {expected}"
            );
        };
        close("dim", ising.dim_scalar(sector), dim);
        close("fs", ising.frobenius_schur_phase_scalar(sector), one);
        close("twist", ising.twist_scalar(sector), twist);
    }
}
