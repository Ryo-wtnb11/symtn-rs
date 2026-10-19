//! A fusion category without braiding through the public typed API:
//! Rep(S₃) from `TableRule` reports `NoBraiding`. The crossing-free
//! operations (transpose, repartition, compose, otimes, and a braid whose
//! only crossings involve the unit) work; every operation that needs an
//! R-symbol is rejected with its typed error before any provider R call and
//! before the destination is touched, as TensorKit throws `SectorMismatch`
//! for `NoBraiding` in `artin_braid` and `permute`.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::typed::{
    ContractSpec, CoreError, Error, GradedSpace, OperationError, Runtime, TensorMap,
};
use symtn_sectors::{
    BraidingStyleKind, FusionRule, ProductFusionRuleExt, SectorCodec, SectorId, TableRule,
    TableSector, Z2FusionRule,
};

/// Rep(S₃), `anyonwiki_code [3,1,0,2,2,0,1]`: objects `1` (trivial), `2`
/// (sign), `3` (two-dimensional).
const REP_S3: &str = "37fa4e9c-97c6-11f1-8ce9-19309dfc597c";

const BOUND: f64 = 256.0 * f64::EPSILON;

/// TensorKitSectors 0.3.9 `D3Irrep` (Rep(S₃), `src/irreps/dnirrep.jl`):
/// the crossing-free transpose-then-compose matrix on `x ⊗ x ⊗ x ← x` in the
/// channel order `(xx) → A₁, A₂, E`; it equals TensorKit's `F^{xxx}_x` up to
/// the sign of the `A₂` column (checked against TensorKit 0.17.1). Only the
/// moduli `1/2`, `1/√2`,
/// `0` and the cross ratio are vertex-gauge invariant, and only those are
/// compared.
fn tensorkit_rep_s3_f(row: usize, column: usize) -> Complex64 {
    use std::f64::consts::FRAC_1_SQRT_2 as S;
    let f = [[0.5, 0.5, S], [0.5, 0.5, -S], [S, -S, 0.0]];
    Complex64::new(f[row][column], 0.0)
}

fn rep_s3() -> Arc<TableRule> {
    Arc::new(TableRule::by_uuid(REP_S3).unwrap())
}

fn leg(rule: &Arc<TableRule>, sectors: &[(u16, usize)]) -> GradedSpace<TableRule> {
    GradedSpace::try_new(
        rule.clone(),
        sectors
            .iter()
            .map(|&(label, d)| (TableSector::new(label), d)),
    )
    .unwrap()
}

fn assert_close(
    actual: &TensorMap<TableRule, Complex64>,
    expected: &TensorMap<TableRule, Complex64>,
) {
    let (a, e) = (actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(a.len(), e.len());
    for (x, y) in a.iter().zip(e.iter()) {
        assert!((x - y).norm() <= BOUND, "{x} != {y}");
    }
}

fn seeded(
    runtime: &Runtime,
    codomain: &[&GradedSpace<TableRule>],
    domain: &[&GradedSpace<TableRule>],
    seed: u64,
) -> TensorMap<TableRule, Complex64> {
    TensorMap::rand_with_seed(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        seed,
    )
    .unwrap()
}

#[test]
fn rep_s3_reports_no_braiding_and_products_inherit_it() {
    let rule = rep_s3();
    assert_eq!(rule.braiding_style(), BraidingStyleKind::NoBraiding);
    let product = TableRule::by_uuid(REP_S3).unwrap().product(Z2FusionRule);
    assert_eq!(product.braiding_style(), BraidingStyleKind::NoBraiding);
}

#[test]
fn crossing_free_operations_work_without_braiding() {
    let runtime = Runtime::builder().build().unwrap();
    let rule = rep_s3();
    let x = leg(&rule, &[(3, 1)]);
    let mixed = leg(&rule, &[(1, 2), (3, 1)]);
    let dual_x = x.try_dual().unwrap();

    // transpose and repartition round trips.
    let t = seeded(&runtime, &[&x, &x, &x], &[&x], 7);
    let back = t
        .transpose(&[3], &[2, 1, 0])
        .unwrap()
        .transpose(&[3, 2, 1], &[0])
        .unwrap();
    assert_close(&back, &t);
    let m = seeded(&runtime, &[&mixed, &x], &[&mixed], 8);
    assert_close(&m.repartition(1).unwrap().repartition(2).unwrap(), &m);

    // The crossing-free transpose of the three states of x³ ← x, composed
    // with the dual basis: a unitary 3x3 basis change whose entries agree in
    // modulus with TensorKit's `D3Irrep` (Rep(S₃) in another gauge; a vertex
    // gauge moves only phases of these entries).
    let channels: Vec<SectorId> = rule
        .fusion_channels(SectorId::new(2), SectorId::new(2))
        .to_vec();
    assert_eq!(channels.len(), 3);
    let basis =
        |space: &GradedSpace<TableRule>, channel: SectorId| -> TensorMap<TableRule, Complex64> {
            let label = rule.decode_sector(channel).unwrap();
            TensorMap::from_subblock_fn(&runtime, [space, space, space], [space], |trees, _| {
                Complex64::new((trees.codomain_innerlines() == [label]) as u8 as f64, 0.0)
            })
            .unwrap()
        };
    let mut f = [[Complex64::new(0.0, 0.0); 3]; 3];
    for (column, &c) in channels.iter().enumerate() {
        let transformed = basis(&x, c).transpose(&[3], &[2, 1, 0]).unwrap();
        for (row, &r) in channels.iter().enumerate() {
            f[row][column] = transformed
                .compose(&basis(&dual_x, r))
                .unwrap()
                .dense_data()
                .unwrap()[0];
        }
    }
    // Moduli plus unitarity fix the matrix up to row and column phases, and a
    // vertex gauge acts by exactly such diagonal phases; the cross ratio
    // M00·M12 / (M02·M10) is invariant under them and pins the one remaining
    // relative phase (−1 in TensorKit's real gauge).
    let cross = |m: &dyn Fn(usize, usize) -> Complex64| m(0, 0) * m(1, 2) / (m(0, 2) * m(1, 0));
    let (ours, tensorkit) = (cross(&|r, c| f[r][c]), cross(&tensorkit_rep_s3_f));
    assert!(
        (tensorkit + 1.0).norm() <= BOUND,
        "TensorKit cross ratio {tensorkit}"
    );
    assert!(
        (ours - tensorkit).norm() <= BOUND,
        "cross ratio {ours} != TensorKit {tensorkit}"
    );
    for row in 0..3 {
        for column in 0..3 {
            let tensorkit = tensorkit_rep_s3_f(row, column);
            assert!(
                (f[row][column].norm() - tensorkit.norm()).abs() <= BOUND,
                "|f[{row}][{column}]| {} != TensorKit {}",
                f[row][column],
                tensorkit
            );
            let gram: Complex64 = (0..3).map(|k| f[k][row].conj() * f[k][column]).sum();
            let identity = Complex64::new((row == column) as u8 as f64, 0.0);
            assert!((gram - identity).norm() <= BOUND, "f is not unitary");
        }
    }

    // compose and otimes.
    let a = seeded(&runtime, &[&mixed], &[&mixed], 9);
    let b = seeded(&runtime, &[&mixed], &[&mixed], 10);
    assert!(a.compose(&b).is_ok());
    assert_eq!(a.otimes(&b).unwrap().rank(), 4);

    // A braid whose only crossing involves the unit needs no R-symbol
    // (TensorKit `artin_braid` returns coefficient one); the leg reorder
    // leaves the one stored coefficient unchanged up to rounding.
    let unit = leg(&rule, &[(1, 1)]);
    let u = seeded(&runtime, &[&unit, &x], &[&x], 11);
    let swapped = u.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap();
    let (a, e) = (swapped.dense_data().unwrap(), u.dense_data().unwrap());
    assert_eq!(a.len(), e.len());
    assert!(a.iter().zip(e.iter()).all(|(x, y)| (x - y).norm() <= BOUND));
}

/// A braid crossing two non-unit sectors of a `NoBraiding` rule.
fn is_no_braiding_sector_braid(error: &Error) -> bool {
    matches!(
        error,
        Error::Operation(operation) if matches!(
            **operation,
            OperationError::Core(CoreError::UnsupportedSectorBraid {
                style: BraidingStyleKind::NoBraiding,
                ..
            })
        )
    )
}

#[test]
fn braiding_operations_are_rejected_before_any_work() {
    let runtime = Runtime::builder().build().unwrap();
    let rule = rep_s3();
    // Mixed blocks: the unit-unit and unit-x crossings are structural, the
    // x-x crossing needs R. The whole operation fails.
    let mixed = leg(&rule, &[(1, 1), (3, 1)]);
    let source = seeded(&runtime, &[&mixed, &mixed], &[&mixed], 12);

    let error = source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap_err();
    assert!(is_no_braiding_sector_braid(&error), "{error:?}");

    let mut destination = seeded(&runtime, &[&mixed, &mixed], &[&mixed], 13);
    let before = destination.dense_data().unwrap().to_vec();
    let error = source
        .braid_into(
            &[1, 0],
            &[2],
            &[0, 1, 2],
            &mut destination,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap_err();
    assert!(is_no_braiding_sector_braid(&error), "{error:?}");
    assert_eq!(destination.dense_data().unwrap(), &before[..]);

    let error = source.permute(&[1, 0], &[2]).unwrap_err();
    assert!(
        matches!(
            &error,
            Error::Operation(operation) if matches!(
                **operation,
                OperationError::UnsupportedBraidingStyle {
                    style: BraidingStyleKind::NoBraiding,
                    ..
                }
            )
        ),
        "{error:?}"
    );

    let other = seeded(&runtime, &[&mixed], &[&mixed], 14);
    let error = source
        .contract(
            &other,
            &ContractSpec {
                lhs: &[0],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap_err();
    assert!(
        matches!(
            &error,
            Error::Operation(operation)
                if matches!(**operation, OperationError::UnsupportedTensorContractScope { .. })
        ),
        "{error:?}"
    );
}

#[test]
fn product_with_a_fusion_only_factor_rejects_braid_into_untouched() {
    use symtn_sectors::{product_sector, Z2Irrep};
    let runtime = Runtime::builder().build().unwrap();
    let product = Arc::new(TableRule::by_uuid(REP_S3).unwrap().product(Z2FusionRule));
    assert_eq!(product.braiding_style(), BraidingStyleKind::NoBraiding);
    let leg = GradedSpace::try_new(
        product,
        [
            (product_sector(TableSector::new(1), Z2Irrep::EVEN), 1),
            (product_sector(TableSector::new(3), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 15).unwrap();
    let mut destination: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 16).unwrap();
    let before = destination.dense_data().unwrap().to_vec();
    let error = source
        .braid_into(
            &[1, 0],
            &[2],
            &[0, 1, 2],
            &mut destination,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap_err();
    assert!(is_no_braiding_sector_braid(&error), "{error:?}");
    assert_eq!(destination.dense_data().unwrap(), &before[..]);
}
