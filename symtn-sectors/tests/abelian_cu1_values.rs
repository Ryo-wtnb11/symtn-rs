//! The abelian and CU(1) adapters expose TensorKitSectors' values through
//! symtn's `SectorId` codec: every expected value below is a closed form over
//! semantic labels, so these tests check each adapter's codec and argument
//! order as well as the symbols behind it. An inadmissible F or R is `0.0`,
//! as in TensorKitSectors.
//!
//! The checked fusion algebra keeps symtn's error contract: U(1) or CU(1)
//! fusion leaving the label range is an `Err`, never a wrapped label.

use num_complex::{Complex32, Complex64};
use symtn_sectors::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule, FusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorId, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep, ZNFusionRule, CU1_MAX_TWICE_CHARGE,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

trait Rule: FusionRule + CheckedFusionAlgebra + MultiplicityFreeRigidSymbols<Scalar = f64> {}
impl<R: FusionRule + CheckedFusionAlgebra + MultiplicityFreeRigidSymbols<Scalar = f64>> Rule for R {}

/// `N`, with symtn's checked fusion overflow (U(1)/CU(1) sums leaving the
/// label range) read as the exact `0`.
fn n<R: Rule>(rule: &R, a: SectorId, b: SectorId, c: SectorId) -> bool {
    rule.try_nsymbol(a, b, c) == Ok(1)
}

fn admissible_f<R: Rule>(rule: &R, [a, b, c, d, e, f]: [SectorId; 6]) -> bool {
    n(rule, a, b, e) && n(rule, e, c, d) && n(rule, b, c, f) && n(rule, a, f, d)
}

/// An abelian group `a ⊗ b = fuse(a, b)` on the semantic labels `labels`
/// (TensorKitSectors `ZNIrrep`, `U1Irrep`, `FermionParity`): dimensions and
/// FS phase `1`; `N^{ab}_c = [c = a + b]`; `F` is the product of its four
/// vertices' `N`; `R = N`, except `R^{11}_0 = -1` and `θ_1 = -1` for
/// fermion parity.
fn check_abelian<R: Rule>(
    rule: &R,
    labels: &[i64],
    id: impl Fn(i64) -> SectorId,
    fuse: impl Fn(i64, i64) -> i64,
    dual: impl Fn(i64) -> i64,
    fermionic: bool,
) {
    let odd = |a: i64| fermionic && a == 1;
    let nn = |a, b, c| c == fuse(a, b);
    for &a in labels {
        let sa = id(a);
        assert_eq!(rule.try_dual_sector(sa), Ok(id(dual(a))), "dual {a}");
        for value in [
            rule.dim_scalar(sa),
            rule.inv_dim_scalar(sa),
            rule.sqrt_dim_scalar(sa),
            rule.inv_sqrt_dim_scalar(sa),
            rule.frobenius_schur_phase_scalar(sa),
        ] {
            assert_eq!(value, 1.0, "dimensions and FS phase of {a}");
        }
        let twist = if odd(a) { -1.0 } else { 1.0 };
        assert_eq!(rule.twist_scalar(sa), twist, "twist {a}");
        for &b in labels {
            let sb = id(b);
            assert_eq!(
                rule.try_fusion_channels(sa, sb).unwrap().as_slice(),
                &[id(fuse(a, b))],
                "{a} ⊗ {b}"
            );
            for &c in labels {
                let sc = id(c);
                assert_eq!(rule.try_nsymbol(sa, sb, sc), Ok(usize::from(nn(a, b, c))));
                let r = match (nn(a, b, c), odd(a) && odd(b)) {
                    (false, _) => 0.0,
                    (true, true) => -1.0,
                    (true, false) => 1.0,
                };
                assert_eq!(rule.r_symbol_scalar(sa, sb, sc), r, "R {a} {b} {c}");
                for &d in labels {
                    for &e in labels {
                        for &f in labels {
                            let admissible =
                                nn(a, b, e) && nn(e, c, d) && nn(b, c, f) && nn(a, f, d);
                            assert_eq!(
                                rule.f_symbol_scalar(sa, sb, sc, id(d), id(e), id(f)),
                                if admissible { 1.0 } else { 0.0 },
                                "F {a} {b} {c} {d} {e} {f}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn abelian_values_are_the_tensorkitsectors_closed_forms() {
    let parity = |p: i64| Z2Irrep::new(p as u8).sector_id();
    let xor = |a: i64, b: i64| a ^ b;
    check_abelian(&Z2FusionRule, &[0, 1], parity, xor, |a| a, false);
    check_abelian(&FermionParityFusionRule, &[0, 1], parity, xor, |a| a, true);
    for modulus in 1..=4 {
        let rule = ZNFusionRule::new(modulus).unwrap();
        let m = i64::from(modulus);
        let labels: Vec<i64> = (0..m).collect();
        check_abelian(
            &rule,
            &labels,
            |q| rule.irrep(q).into(),
            |a, b| (a + b) % m,
            |a| (m - a) % m,
            false,
        );
    }
    let u1 = |q: i64| U1Irrep::new(i32::try_from(q).unwrap()).sector_id();
    // `c` ranges over the window too, so `N = 0` on sums outside it is
    // exercised without leaving the label range.
    check_abelian(
        &U1FusionRule,
        &[-2, -1, 0, 1, 2],
        u1,
        |a, b| a + b,
        |a| -a,
        false,
    );
}

#[test]
fn u1_fusion_leaving_the_charge_range_is_an_error() {
    let rule = U1FusionRule;
    let u1 = |q: i32| U1Irrep::new(q).sector_id();
    let (min, max) = (i32::MIN + 1, i32::MAX);
    assert_eq!(rule.try_dual_sector(u1(max)), Ok(u1(min)));
    for (a, b) in [(max, 1), (1, max), (min, -1), (-1, min)] {
        assert!(rule.try_fusion_channels(u1(a), u1(b)).is_err(), "{a} ⊗ {b}");
        for c in [min, -1, 0, 1, max] {
            assert!(
                rule.try_nsymbol(u1(a), u1(b), u1(c)).is_err(),
                "N {a} {b} {c}"
            );
        }
    }
    for (a, b, c) in [(max, -1, max - 1), (min, max, 0), (min, 1, min + 1)] {
        assert_eq!(
            rule.try_fusion_channels(u1(a), u1(b)).unwrap().as_slice(),
            &[u1(c)]
        );
    }
}

const OP: CU1Irrep = CU1Irrep::VACUUM;
const OM: CU1Irrep = CU1Irrep::PSEUDOSCALAR;

fn j(twice_charge: u32) -> CU1Irrep {
    CU1Irrep::from_twice_charge(twice_charge)
}

/// `0+`, `0-`, `2j = 1..=5`.
fn cu1_labels() -> Vec<CU1Irrep> {
    [OP, OM].into_iter().chain((1..=5).map(j)).collect()
}

fn cu1_dim(a: CU1Irrep) -> usize {
    if a.twice_charge() == 0 {
        1
    } else {
        2
    }
}

/// TensorKitSectors `cu1irrep.jl:fusiontensor(a, b, c)` as `C[i][k][m]`
/// (basis `+j, -j` of a charged irrep), all zero when `c ∉ a ⊗ b`; the same
/// tensors fusion-symbols' `tests/cu1_values.rs` contracts.
fn fusion_tensor(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> [[[f64; 2]; 2]; 2] {
    let mut t = [[[0.0; 2]; 2]; 2];
    let (qa, qb, qc) = (
        i64::from(a.twice_charge()),
        i64::from(b.twice_charge()),
        i64::from(c.twice_charge()),
    );
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let sign = |x: CU1Irrep| {
        if x.conjugation_sector() == 1 {
            -1.0
        } else {
            1.0
        }
    };
    if qc == 0 {
        if qa == 0 && qb == 0 {
            if a.conjugation_sector() ^ b.conjugation_sector() == c.conjugation_sector() {
                t[0][0][0] = 1.0;
            }
        } else if qa == qb {
            t[0][1][0] = r;
            t[1][0][0] = r * sign(c);
        }
    } else if qa == 0 {
        if qb == qc {
            t[0][0][0] = 1.0;
            t[0][1][1] = sign(a);
        }
    } else if qb == 0 {
        if qa == qc {
            t[0][0][0] = 1.0;
            t[1][0][1] = sign(b);
        }
    } else if qc == qa + qb {
        t[0][0][0] = 1.0;
        t[1][1][1] = 1.0;
    } else if qc == qa - qb {
        t[0][1][0] = 1.0;
        t[1][0][1] = 1.0;
    } else if qc == qb - qa {
        t[1][0][0] = 1.0;
        t[0][1][1] = 1.0;
    }
    t
}

/// `F^{abc}_d[e, f] = (1/d_d) Σ C^{ab}_e[i,k,μ] C^{ec}_d[μ,l,m] C^{bc}_f[k,l,ν] C^{af}_d[i,ν,m]`.
fn dense_f([a, b, c, d, e, f]: [CU1Irrep; 6]) -> f64 {
    let (x1, x2, y1, y2) = (
        fusion_tensor(a, b, e),
        fusion_tensor(e, c, d),
        fusion_tensor(b, c, f),
        fusion_tensor(a, f, d),
    );
    let mut sum = 0.0;
    for i in 0..cu1_dim(a) {
        for k in 0..cu1_dim(b) {
            for l in 0..cu1_dim(c) {
                for m in 0..cu1_dim(d) {
                    for mu in 0..cu1_dim(e) {
                        for nu in 0..cu1_dim(f) {
                            sum += x1[i][k][mu] * x2[mu][l][m] * y1[k][l][nu] * y2[i][nu][m];
                        }
                    }
                }
            }
        }
    }
    sum / cu1_dim(d) as f64
}

/// `C^{ba}_c[k,i,m] = R^{ab}_c C^{ab}_c[i,k,m]`: the ratio on any nonzero entry.
fn dense_r(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> f64 {
    let (ab, ba) = (fusion_tensor(a, b, c), fusion_tensor(b, a, c));
    for i in 0..2 {
        for k in 0..2 {
            for m in 0..2 {
                if ab[i][k][m] != 0.0 {
                    return ba[k][i][m] / ab[i][k][m];
                }
            }
        }
    }
    0.0
}

#[test]
fn cu1_f_and_r_are_the_fusion_tensor_recouplings() {
    let rule = CU1FusionRule;
    let all = cu1_labels();
    let id = |x: CU1Irrep| x.sector_id();
    let mut nonzero = (0, 0);
    for &a in &all {
        for &b in &all {
            for &c in &all {
                let r = rule.r_symbol_scalar(id(a), id(b), id(c));
                assert_eq!(r, dense_r(a, b, c), "R {a:?} {b:?} {c:?}");
                nonzero.0 += usize::from(r != 0.0);
                for &d in &all {
                    for &e in &all {
                        for &f in &all {
                            let l = [a, b, c, d, e, f];
                            let got =
                                rule.f_symbol_scalar(id(a), id(b), id(c), id(d), id(e), id(f));
                            numerics::assert_close(&format!("F {l:?}"), got, dense_f(l), 16);
                            nonzero.1 += usize::from(got != 0.0);
                        }
                    }
                }
            }
        }
    }
    // Nonzero counts over the seven labels (as in fusion-symbols and
    // TensorKitSectors 0.3.9): a vacuous sweep would not reach them.
    assert_eq!(nonzero, (64, 628));
}

#[test]
fn cu1_unary_data_and_fusion_order_follow_tensorkitsectors() {
    let rule = CU1FusionRule;
    let id = |x: CU1Irrep| x.sector_id();
    // `dim = 1` for `0±`, `2` when charged; self-dual; FS phase and twist 1,
    // up to the largest representable charge.
    let wide = [j(1 << 31), j(CU1_MAX_TWICE_CHARGE)];
    for a in cu1_labels().into_iter().chain(wide) {
        let dim = cu1_dim(a) as f64;
        assert_eq!(rule.vacuum(), id(OP));
        assert_eq!(rule.try_dual_sector(id(a)), Ok(id(a)));
        assert_eq!(rule.dim_scalar(id(a)), dim, "{a:?}");
        assert_eq!(rule.inv_dim_scalar(id(a)), 1.0 / dim, "{a:?}");
        assert_eq!(rule.sqrt_dim_scalar(id(a)), dim.sqrt(), "{a:?}");
        assert_eq!(rule.inv_sqrt_dim_scalar(id(a)), 1.0 / dim.sqrt(), "{a:?}");
        assert_eq!(rule.frobenius_schur_phase_scalar(id(a)), 1.0, "{a:?}");
        assert_eq!(rule.twist_scalar(id(a)), 1.0, "{a:?}");
    }
    // `SectorProductIterator{CU1Irrep}` order: `0± ⊗ 0± = 0^{s xor s'}`,
    // `0± ⊗ j = j`, `j ⊗ j = [0+, 0-, 2j]`, `j ⊗ k = [|j-k|, j+k]`.
    let all = cu1_labels();
    for &a in &all {
        for &b in &all {
            let (qa, qb) = (a.twice_charge(), b.twice_charge());
            let want = match (qa, qb) {
                (0, 0) if a.conjugation_sector() == b.conjugation_sector() => vec![OP],
                (0, 0) => vec![OM],
                (0, _) => vec![b],
                (_, 0) => vec![a],
                _ if qa == qb => vec![OP, OM, j(2 * qa)],
                _ => vec![j(qa.abs_diff(qb)), j(qa + qb)],
            };
            let want: Vec<SectorId> = want.into_iter().map(id).collect();
            let got = rule.try_fusion_channels(id(a), id(b)).unwrap();
            assert_eq!(got.as_slice(), want.as_slice(), "{a:?} ⊗ {b:?}");
            for &c in &all {
                assert_eq!(
                    rule.try_nsymbol(id(a), id(b), id(c)),
                    Ok(usize::from(want.contains(&id(c))))
                );
            }
        }
    }
}

#[test]
fn cu1_fusion_leaving_the_charge_range_is_an_error() {
    let rule = CU1FusionRule;
    let id = |x: CU1Irrep| x.sector_id();
    let max = CU1_MAX_TWICE_CHARGE;
    let half = max / 2;
    assert_eq!(
        rule.try_fusion_channels(id(j(half)), id(j(half)))
            .unwrap()
            .as_slice(),
        &[id(OP), id(OM), id(j(max))]
    );
    for (a, b) in [(1, max), (max, 1), (half + 1, half + 1), (max, max)] {
        assert!(
            rule.try_fusion_channels(id(j(a)), id(j(b))).is_err(),
            "{a} ⊗ {b}"
        );
        for c in [OP, OM, j(1), j(max)] {
            assert!(
                rule.try_nsymbol(id(j(a)), id(j(b)), id(c)).is_err(),
                "N {a} {b} {c:?}"
            );
        }
    }
}

/// Every F and R tuple over `ids`: `0.0` exactly when a vertex is
/// inadmissible (TensorKitSectors' value), and `±1` otherwise.
fn assert_off_shell_zero<R: Rule>(rule: &R, ids: &[SectorId]) {
    for &a in ids {
        for &b in ids {
            for &c in ids {
                let r = rule.r_symbol_scalar(a, b, c);
                assert_eq!(r == 0.0, !n(rule, a, b, c), "R {a:?} {b:?} {c:?}");
                for &d in ids {
                    for &e in ids {
                        for &f in ids {
                            let labels = [a, b, c, d, e, f];
                            let value = rule.f_symbol_scalar(a, b, c, d, e, f);
                            assert_eq!(value == 0.0, !admissible_f(rule, labels), "F {labels:?}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn inadmissible_f_and_r_are_zero_at_the_label_boundaries() {
    // What: off-shell F/R are 0, including at the Z_N modulus and U(1)
    // charge extremes the closed-form windows do not reach.
    for modulus in [3, u32::MAX] {
        let rule = ZNFusionRule::new(modulus).unwrap();
        let ids: Vec<SectorId> = [0, 1, i64::from(u32::MAX) - 1]
            .into_iter()
            .map(|q| rule.irrep(q).into())
            .collect();
        assert_off_shell_zero(&rule, &ids);
    }
    let u1: Vec<SectorId> = [i32::MIN + 1, -1, 0, 1, i32::MAX]
        .into_iter()
        .map(|q| U1Irrep::new(q).sector_id())
        .collect();
    assert_off_shell_zero(&U1FusionRule, &u1);
    let parity = [Z2Irrep::EVEN.sector_id(), Z2Irrep::ODD.sector_id()];
    assert_off_shell_zero(&Z2FusionRule, &parity);
    assert_off_shell_zero(&FermionParityFusionRule, &parity);
}
