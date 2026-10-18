//! Systematic TensorKit-semantics axiom checks for the multiplicity-free
//! symbol layer (issue #9, part 1).
//!
//! Conventions implemented here are copied verbatim from TensorKitSectors
//! v0.3.6 (`~/.julia/packages/TensorKitSectors/LXgpR/src/sectors.jl`), which
//! is the version loaded by TensorKit v0.16.2 — NOT textbook variants:
//!
//! * `Fsymbol(a, b, c, d, e, f)` is the recoupling from `((a ⊗ b) ⊗ c → d`
//!   via `e = a ⊗ b)` to `(a ⊗ (b ⊗ c) → d` via `f = b ⊗ c)`. symtn's
//!   `f_symbol_scalar(left, middle, right, coupled, left_coupled,
//!   right_coupled)` uses the same argument order.
//! * Pentagon (`pentagon_equation`, multiplicity-free scalar branch):
//!   for `f ∈ a⊗b`, `h ∈ c⊗d`, `g ∈ f⊗c`, `i ∈ b⊗h`, `e ∈ (g⊗d) ∩ (a⊗i)`:
//!   `F(f,c,d,e,g,h)·F(a,b,h,e,f,i)
//!      = Σ_{j ∈ b⊗c} F(a,b,c,g,f,j)·F(a,j,d,e,g,i)·F(b,c,d,i,j,h)`.
//! * Hexagon (`hexagon_equation`; R chirality: `c` is braided *over* the
//!   fusion product of `a` and `b`): for `e ∈ c⊗a`, `f ∈ c⊗b`,
//!   `d ∈ (e⊗b) ∩ (a⊗f)`:
//!   `R(c,a,e)·F(a,c,b,d,e,f)·R(c,b,f)
//!      = Σ_{g ∈ a⊗b} F(c,a,b,d,e,g)·R(c,g,d)·F(a,b,c,d,g,f)`.
//!   The second ("inverse") hexagon is the same equation with every
//!   R-symbol replaced by its inverse (all rules here have real unitary
//!   R ∈ {±1}, so R⁻¹ = 1/R = R).
//! * Twist (`twist_from_Rsymbol`):
//!   `θ_a = Σ_{b ∈ a⊗a} (dim b / dim a) · R(a,a,b)`.
//! * Frobenius-Schur phase (`frobenius_schur_phase_from_Fsymbol`):
//!   `fs(a) = sign(F(a, ā, a, a, 1, 1))`.
//! * Triangle (`triangle_equation`): `F(1,a,b,c,a,c) = F(a,1,b,c,a,b)
//!   = F(a,b,1,c,c,b) = 1` for every `c ∈ a⊗b`.
//!
//! Hardcoded cross-check values are TensorKit `Fsymbol`/`Rsymbol`/`twist`/
//! `frobenius_schur_phase` outputs, checked against TensorKit 0.17.1
//! (TensorKitSectors 0.3.10, Julia 1.11.6).

use symtn_core::{
    CU1FusionRule, CU1Irrep, FermionParityFusionRule, FusionRule, Fz2SectorLayout,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, PackedProductCodec,
    ProductFusionRule, ProductSectorCodec, ProductSectorLayout, SU2FusionRule, SU2Irrep, SectorId,
    Su2SectorLayout, TensorKitProductCodec, U1FusionRule, U1Irrep, U1SectorLayout, Z2FusionRule,
    Z2Irrep,
};

const TOL: f64 = 1e-12;

fn close(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= TOL * (1.0 + lhs.abs().max(rhs.abs()))
}

/// Every symbol trait in this file resolves to `Scalar = f64`.
trait Rule: MultiplicityFreeRigidSymbols<Scalar = f64> {}
impl<R: MultiplicityFreeRigidSymbols<Scalar = f64>> Rule for R {}

/// TensorKitSectors `pentagon_equation` (multiplicity-free scalar branch)
/// over every sector quadruple; returns the number of scalar equations
/// checked. Inadmissible summands are safe to include: unique-fusion
/// components force admissibility and the SU(2) 6j returns 0.
fn check_pentagon<R: Rule>(rule: &R, sectors: &[SectorId]) -> u64 {
    let mut checked = 0u64;
    for &a in sectors {
        for &b in sectors {
            for &c in sectors {
                for &d in sectors {
                    for &f in &rule.fusion_channels(a, b) {
                        for &h in &rule.fusion_channels(c, d) {
                            for &g in &rule.fusion_channels(f, c) {
                                for &i in &rule.fusion_channels(b, h) {
                                    for &e in &rule.fusion_channels(g, d) {
                                        if rule.nsymbol(a, i, e) == 0 {
                                            continue;
                                        }
                                        let lhs = rule.f_symbol_scalar(f, c, d, e, g, h)
                                            * rule.f_symbol_scalar(a, b, h, e, f, i);
                                        let rhs: f64 = rule
                                            .fusion_channels(b, c)
                                            .iter()
                                            .map(|&j| {
                                                rule.f_symbol_scalar(a, b, c, g, f, j)
                                                    * rule.f_symbol_scalar(a, j, d, e, g, i)
                                                    * rule.f_symbol_scalar(b, c, d, i, j, h)
                                            })
                                            .sum();
                                        assert!(
                                            close(lhs, rhs),
                                            "pentagon failed at \
                                             (a,b,c,d)=({a:?},{b:?},{c:?},{d:?}) \
                                             (e,f,g,h,i)=({e:?},{f:?},{g:?},{h:?},{i:?}): \
                                             {lhs} vs {rhs}"
                                        );
                                        checked += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    checked
}

/// TensorKitSectors `hexagon_equation` over every sector triple; with
/// `invert` every R-symbol is replaced by its inverse (second hexagon).
/// Summands with `N(c,g,d) == 0` vanish identically (their F factor is 0)
/// and are skipped so `1/R` never divides by zero.
fn check_hexagon<R: Rule>(rule: &R, sectors: &[SectorId], invert: bool) -> u64 {
    let r = |rule: &R, a, b, c| {
        let value = rule.r_symbol_scalar(a, b, c);
        if invert {
            1.0 / value
        } else {
            value
        }
    };
    let mut checked = 0u64;
    for &a in sectors {
        for &b in sectors {
            for &c in sectors {
                for &e in &rule.fusion_channels(c, a) {
                    for &f in &rule.fusion_channels(c, b) {
                        for &d in &rule.fusion_channels(e, b) {
                            if rule.nsymbol(a, f, d) == 0 {
                                continue;
                            }
                            let lhs = r(rule, c, a, e)
                                * rule.f_symbol_scalar(a, c, b, d, e, f)
                                * r(rule, c, b, f);
                            let rhs: f64 = rule
                                .fusion_channels(a, b)
                                .iter()
                                .filter(|&&g| rule.nsymbol(c, g, d) > 0)
                                .map(|&g| {
                                    rule.f_symbol_scalar(c, a, b, d, e, g)
                                        * r(rule, c, g, d)
                                        * rule.f_symbol_scalar(a, b, c, d, g, f)
                                })
                                .sum();
                            assert!(
                                close(lhs, rhs),
                                "hexagon(invert={invert}) failed at \
                                 (a,b,c)=({a:?},{b:?},{c:?}) (d,e,f)=({d:?},{e:?},{f:?}): \
                                 {lhs} vs {rhs}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    checked
}

/// TensorKitSectors `triangle_equation`: unit F-moves are trivial.
fn check_triangle<R: Rule>(rule: &R, sectors: &[SectorId]) -> u64 {
    let one = rule.vacuum();
    let mut checked = 0u64;
    for &a in sectors {
        for &b in sectors {
            for &c in &rule.fusion_channels(a, b) {
                for f in [
                    rule.f_symbol_scalar(one, a, b, c, a, c),
                    rule.f_symbol_scalar(a, one, b, c, a, b),
                    rule.f_symbol_scalar(a, b, one, c, c, b),
                ] {
                    assert!(close(f, 1.0), "triangle failed at ({a:?},{b:?}): {f}");
                    checked += 1;
                }
            }
        }
    }
    checked
}

/// Twist/R consistency (`twist_from_Rsymbol`), dual coherence
/// `θ_ā = θ_a`, and Frobenius-Schur coherence
/// (`frobenius_schur_phase_from_Fsymbol`, `fs(ā) = conj(fs(a)) = fs(a)`).
fn check_twist_and_frobenius<R: Rule>(rule: &R, sectors: &[SectorId]) -> u64 {
    let mut checked = 0u64;
    for &a in sectors {
        let dual = rule.dual(a);
        let theta = rule.twist_scalar(a);
        let derived: f64 = rule
            .fusion_channels(a, a)
            .iter()
            .map(|&b| rule.dim_scalar(b) / rule.dim_scalar(a) * rule.r_symbol_scalar(a, a, b))
            .sum();
        assert!(
            close(theta, derived),
            "twist_scalar({a:?}) = {theta} but Σ (d_b/d_a) R(a,a,b) = {derived}"
        );
        assert!(
            close(rule.twist_scalar(dual), theta),
            "θ(dual({a:?})) != θ({a:?})"
        );

        let fs = rule.frobenius_schur_phase_scalar(a);
        let f_loop = rule.f_symbol_scalar(a, dual, a, a, rule.vacuum(), rule.vacuum());
        assert!(fs.abs() == 1.0, "fs({a:?}) = {fs} is not a phase");
        assert!(
            close(fs, f_loop.signum()),
            "fs({a:?}) = {fs} but sign F(a,ā,a,a,1,1) = {}",
            f_loop.signum()
        );
        assert!(
            close(rule.frobenius_schur_phase_scalar(dual), fs),
            "fs(dual({a:?})) != fs({a:?})"
        );
        checked += 5;
    }
    checked
}

fn check_all<R: Rule>(name: &str, rule: &R, sectors: &[SectorId]) {
    let pentagon = check_pentagon(rule, sectors);
    let hexagon = check_hexagon(rule, sectors, false);
    let hexagon_inv = check_hexagon(rule, sectors, true);
    let triangle = check_triangle(rule, sectors);
    let twist = check_twist_and_frobenius(rule, sectors);
    // Every window must actually exercise the identities.
    assert!(pentagon > 0 && hexagon > 0 && triangle > 0 && twist > 0);
    println!(
        "{name}: pentagon {pentagon}, hexagon {hexagon}+{hexagon_inv}, \
         triangle {triangle}, twist/frobenius {twist}"
    );
}

// ---------------------------------------------------------------------------
// Sector windows
// ---------------------------------------------------------------------------

fn z2_sectors() -> Vec<SectorId> {
    vec![Z2Irrep::EVEN.into(), Z2Irrep::ODD.into()]
}

fn u1_sectors(window: i32) -> Vec<SectorId> {
    (-window..=window).map(|q| U1Irrep::new(q).into()).collect()
}

fn su2_sectors(max_twice_spin: usize) -> Vec<SectorId> {
    (0..=max_twice_spin)
        .map(|j| SU2Irrep::from_twice_spin(j).into())
        .collect()
}

fn cu1_sectors() -> Vec<SectorId> {
    [
        CU1Irrep::VACUUM,
        CU1Irrep::PSEUDOSCALAR,
        CU1Irrep::from_twice_charge(1),
        CU1Irrep::from_twice_charge(2),
        CU1Irrep::from_twice_charge(3),
    ]
    .map(Into::into)
    .into()
}

/// `U1 ⊠ fZ2`, with U1 as the left factor and fZ2 as the right factor.
type U1Fz2Codec = PackedProductCodec<U1SectorLayout, Fz2SectorLayout>;
type U1Fz2Rule = ProductFusionRule<U1FusionRule, FermionParityFusionRule, U1Fz2Codec>;

fn u1_fz2_sectors(window: i32) -> Vec<SectorId> {
    let mut sectors = Vec::new();
    for q in -window..=window {
        for p in 0..2usize {
            sectors.push(U1Fz2Codec::encode(
                U1Irrep::new(q).sector_id(),
                SectorId::new(p),
            ));
        }
    }
    sectors
}

/// `fZ2 ⊠ U1 ⊠ SU2`, left-associated as `(fZ2 ⊠ U1) ⊠ SU2`.
type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

fn triple_rule() -> TripleRule {
    ProductFusionRule::new(
        ProductFusionRule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

fn triple_sector(parity: u8, charge: i32, twice_spin: usize) -> SectorId {
    let inner = Fz2U1Codec::encode(
        SectorId::new(usize::from(parity & 1)),
        U1Irrep::new(charge).sector_id(),
    );
    TripleCodec::encode(inner, SU2Irrep::from_twice_spin(twice_spin).sector_id())
}

fn triple_sectors(charge_window: i32, max_twice_spin: usize) -> Vec<SectorId> {
    let mut sectors = Vec::new();
    for p in 0..2u8 {
        for q in -charge_window..=charge_window {
            for j in 0..=max_twice_spin {
                sectors.push(triple_sector(p, q, j));
            }
        }
    }
    sectors
}

type CantorU1Fz2Rule = ProductFusionRule<U1FusionRule, FermionParityFusionRule>;
type CantorFz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type CantorTripleRule = ProductFusionRule<CantorFz2U1Rule, SU2FusionRule>;

fn cantor_u1_fz2_sectors(window: i32) -> Vec<SectorId> {
    let mut sectors = Vec::new();
    for q in -window..=window {
        for p in 0..2usize {
            sectors.push(TensorKitProductCodec::encode(
                U1Irrep::new(q).sector_id(),
                SectorId::new(p),
            ));
        }
    }
    sectors
}

fn cantor_triple_rule() -> CantorTripleRule {
    ProductFusionRule::new(
        ProductFusionRule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

fn cantor_triple_sector(parity: u8, charge: i32, twice_spin: usize) -> SectorId {
    let inner = TensorKitProductCodec::encode(
        SectorId::new(usize::from(parity & 1)),
        U1Irrep::new(charge).sector_id(),
    );
    TensorKitProductCodec::encode(inner, SU2Irrep::from_twice_spin(twice_spin).sector_id())
}

fn cantor_triple_sectors(charge_window: i32, max_twice_spin: usize) -> Vec<SectorId> {
    let mut sectors = Vec::new();
    for p in 0..2u8 {
        for q in -charge_window..=charge_window {
            for j in 0..=max_twice_spin {
                sectors.push(cantor_triple_sector(p, q, j));
            }
        }
    }
    sectors
}

// ---------------------------------------------------------------------------
// Axiom sweeps
// ---------------------------------------------------------------------------

#[test]
fn axioms_z2() {
    check_all("Z2", &Z2FusionRule, &z2_sectors());
}

#[test]
fn axioms_fz2() {
    check_all("fZ2", &FermionParityFusionRule, &z2_sectors());
}

#[test]
fn axioms_u1() {
    check_all("U1", &U1FusionRule, &u1_sectors(4));
}

#[test]
fn axioms_cu1() {
    // TensorKitSectors 0.3.4 `CU1Irrep`, including both charge-conjugation
    // singlets and charges 1/2, 1, 3/2.
    check_all("CU1", &CU1FusionRule, &cu1_sectors());
}

#[test]
fn axioms_su2() {
    // Hexagon over the full 2j <= 6 window (cheap: triples only).
    let full = su2_sectors(6);
    assert!(check_hexagon(&SU2FusionRule, &full, false) > 0);
    assert!(check_hexagon(&SU2FusionRule, &full, true) > 0);
    // Pentagon over 2j <= 3 stays fast in debug builds; the exhaustive
    // 2j <= 6 sweep is `axioms_su2_exhaustive` below.
    check_all("SU2", &SU2FusionRule, &su2_sectors(3));
}

/// Exhaustive SU(2) pentagon sweep over 2j <= 6.
/// Run with: `cargo test -p symtn-core --release -- --ignored su2_exhaustive`
#[test]
#[ignore = "exhaustive sweep, run explicitly in release"]
fn axioms_su2_exhaustive() {
    check_all("SU2 (2j<=6)", &SU2FusionRule, &su2_sectors(6));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn axioms_u1_fz2() {
    let rule = U1Fz2Rule::new(U1FusionRule, FermionParityFusionRule);
    check_all("U1xfZ2", &rule, &u1_fz2_sectors(2));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn axioms_fz2_u1_su2() {
    check_all("fZ2xU1xSU2", &triple_rule(), &triple_sectors(1, 1));
}

/// Product rules with the default `TensorKitProductCodec` (Cantor pairing)
/// satisfy the axioms.
#[test]
fn axioms_cantor_product_codec() {
    let pair = CantorU1Fz2Rule::new(U1FusionRule, FermionParityFusionRule);
    check_all("Cantor U1xfZ2", &pair, &cantor_u1_fz2_sectors(1));
    check_all(
        "Cantor fZ2xU1xSU2",
        &cantor_triple_rule(),
        &cantor_triple_sectors(1, 1),
    );
}

/// Exhaustive packed triple-product sweep (|q| <= 2, 2j <= 2).
/// Run with: `cargo test -p symtn-core --release -- --ignored triple_exhaustive`
#[cfg(target_pointer_width = "64")]
#[test]
#[ignore = "exhaustive sweep, run explicitly in release"]
fn axioms_fz2_u1_su2_triple_exhaustive() {
    check_all(
        "fZ2xU1xSU2 (|q|<=2, 2j<=2)",
        &triple_rule(),
        &triple_sectors(2, 2),
    );
}

// ---------------------------------------------------------------------------
// Direct symbol-value cross-checks against TensorKit.
// Checked against TensorKit 0.17.1 (TensorKitSectors 0.3.10).
// ---------------------------------------------------------------------------

#[test]
fn tensorkit_symbol_values_fz2() {
    let rule = FermionParityFusionRule;
    let even: SectorId = Z2Irrep::EVEN.into();
    let odd: SectorId = Z2Irrep::ODD.into();
    // R fZ2 1 1 = -1; all other pairs +1. twist fZ2 1 = -1, fs = +1.
    assert_eq!(rule.r_symbol_scalar(odd, odd, even), -1.0);
    assert_eq!(rule.r_symbol_scalar(odd, even, odd), 1.0);
    assert_eq!(rule.r_symbol_scalar(even, odd, odd), 1.0);
    assert_eq!(rule.r_symbol_scalar(even, even, even), 1.0);
    assert_eq!(rule.twist_scalar(odd), -1.0);
    assert_eq!(rule.twist_scalar(even), 1.0);
    assert_eq!(rule.frobenius_schur_phase_scalar(odd), 1.0);
}

#[test]
fn tensorkit_symbol_values_su2() {
    let rule = SU2FusionRule;
    let s = |twice_spin: usize| -> SectorId { SU2Irrep::from_twice_spin(twice_spin).into() };
    let f = |labels: [usize; 6]| {
        rule.f_symbol_scalar(
            s(labels[0]),
            s(labels[1]),
            s(labels[2]),
            s(labels[3]),
            s(labels[4]),
            s(labels[5]),
        )
    };
    // F SU2 <2j labels> from the oracle dump.
    let expected = [
        ([1, 1, 1, 1, 0, 0], -0.5),
        ([1, 1, 1, 1, 0, 2], 0.866_025_403_784_438_6),
        ([1, 1, 1, 1, 2, 0], 0.866_025_403_784_438_6),
        ([1, 1, 1, 1, 2, 2], 0.499_999_999_999_999_9),
        ([1, 1, 2, 2, 0, 1], -0.577_350_269_189_625_8),
        ([1, 2, 1, 2, 1, 3], 0.942_809_041_582_063_4),
        ([2, 2, 2, 2, 0, 0], 0.333_333_333_333_333_3),
        ([2, 2, 2, 2, 2, 2], 0.499_999_999_999_999_9),
        ([2, 3, 2, 3, 1, 1], 0.166_666_666_666_666_7),
        ([2, 3, 2, 3, 3, 3], 0.733_333_333_333_333_3),
        ([3, 3, 3, 3, 0, 0], -0.25),
        ([3, 3, 3, 3, 2, 2], -0.549_999_999_999_999_9),
    ];
    for (labels, value) in expected {
        assert!(
            close(f(labels), value),
            "F SU2 {labels:?}: {} vs TensorKit {value}",
            f(labels)
        );
    }
    // R SU2 <2j labels>.
    let expected_r = [
        ([1, 1, 0], -1.0),
        ([1, 1, 2], 1.0),
        ([2, 2, 0], 1.0),
        ([2, 2, 2], -1.0),
        ([2, 2, 4], 1.0),
        ([3, 3, 0], -1.0),
        ([3, 3, 2], 1.0),
        ([2, 3, 1], 1.0),
        ([2, 3, 3], -1.0),
    ];
    for ([a, b, c], value) in expected_r {
        assert_eq!(rule.r_symbol_scalar(s(a), s(b), s(c)), value);
    }
    // dim / twist / fs table.
    for twice_spin in 0..=6usize {
        assert_eq!(rule.dim_scalar(s(twice_spin)), (twice_spin + 1) as f64);
        assert_eq!(rule.twist_scalar(s(twice_spin)), 1.0);
        let fs = if twice_spin % 2 == 0 { 1.0 } else { -1.0 };
        assert_eq!(rule.frobenius_schur_phase_scalar(s(twice_spin)), fs);
    }
}

#[test]
fn tensorkit_symbol_values_cu1() {
    // Independent Julia oracle: TensorKitSectors 0.3.4 (`Project.toml`),
    // source tree `~/.julia/packages/TensorKitSectors/JS0fj`, Julia 1.11.6,
    // `cu1irrep.jl` SHA-256 `aead568bf0e1954dd33893e081d0cbb6c5b7f3218ec4c9695f8898014737fd10`,
    // evaluating `src/irreps/cu1irrep.jl:123-216` with zero-based IDs.
    // The values below are not generated by symtn's provider.
    let rule = CU1FusionRule;
    let op: SectorId = CU1Irrep::VACUUM.into();
    let om: SectorId = CU1Irrep::PSEUDOSCALAR.into();
    let q: SectorId = CU1Irrep::from_twice_charge(1).into();
    let r: SectorId = CU1Irrep::from_twice_charge(2).into();
    let t: SectorId = CU1Irrep::from_twice_charge(3).into();
    assert_eq!(rule.fusion_channels(op, q).as_slice(), &[q]);
    assert_eq!(rule.fusion_channels(om, om).as_slice(), &[op]);
    assert_eq!(rule.fusion_channels(om, q).as_slice(), &[q]);
    assert_eq!(rule.fusion_channels(q, q).as_slice(), &[op, om, r]);
    assert_eq!(rule.fusion_channels(q, r).as_slice(), &[q, t]);
    let sqrt_half = 2.0_f64.sqrt() / 2.0;
    for ([a, b, c, d, e, f], value) in [
        ([q, q, q, q, op, op], 0.5),
        ([q, q, q, q, op, r], 2.0_f64.sqrt() / 2.0),
        ([q, q, q, q, om, r], -2.0_f64.sqrt() / 2.0),
        ([q, q, q, q, r, op], 2.0_f64.sqrt() / 2.0),
        ([q, q, q, q, r, r], 0.0),
        ([q, q, r, r, op, q], 2.0_f64.sqrt() / 2.0),
        ([q, r, q, r, q, t], 1.0),
        ([q, r, q, r, q, q], 0.0),
        // One representative of every remaining non-trivial sign family. IDs are
        // deliberately literal so the fixture also catches a codec swap.
        (
            [
                SectorId::new(1),
                SectorId::new(2),
                SectorId::new(3),
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(2),
            ],
            -1.0,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(1),
                SectorId::new(2),
                SectorId::new(0),
                SectorId::new(2),
                SectorId::new(2),
            ],
            -1.0,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(0),
                SectorId::new(2),
            ],
            -1.0,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(0),
                SectorId::new(1),
            ],
            -0.5,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(3),
                SectorId::new(3),
                SectorId::new(1),
                SectorId::new(4),
            ],
            -sqrt_half,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(3),
                SectorId::new(3),
                SectorId::new(2),
                SectorId::new(2),
                SectorId::new(1),
            ],
            -sqrt_half,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(3),
                SectorId::new(2),
                SectorId::new(1),
                SectorId::new(2),
                SectorId::new(2),
            ],
            -1.0,
        ),
        (
            [
                SectorId::new(2),
                SectorId::new(4),
                SectorId::new(3),
                SectorId::new(1),
                SectorId::new(3),
                SectorId::new(2),
            ],
            -1.0,
        ),
    ] {
        let got = rule.f_symbol_scalar(a, b, c, d, e, f);
        assert!(
            close(got, value),
            "CU1 F ({a:?},{b:?},{c:?},{d:?},{e:?},{f:?}): {got} vs {value}"
        );
    }
    assert_eq!(rule.r_symbol_scalar(q, q, op), 1.0);
    assert_eq!(rule.r_symbol_scalar(q, q, om), -1.0);
    assert_eq!(rule.r_symbol_scalar(q, q, r), 1.0);
    assert_eq!(rule.r_symbol_scalar(q, r, q), 1.0);
    assert_eq!(rule.r_symbol_scalar(q, r, t), 1.0);
    for sector in [op, om, q, r, t] {
        assert_eq!(rule.twist_scalar(sector), 1.0);
        assert_eq!(rule.frobenius_schur_phase_scalar(sector), 1.0);
    }
    assert_eq!(rule.dim_scalar(op), 1.0);
    assert_eq!(rule.dim_scalar(om), 1.0);
    assert_eq!(rule.dim_scalar(q), 2.0);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn tensorkit_symbol_values_fz2_u1_su2() {
    let rule = triple_rule();
    // R I3 (1,1,1) (1,1,1) -> (0,2,0) = 1, -> (0,2,2) = -1
    let odd = triple_sector(1, 1, 1);
    assert_eq!(rule.r_symbol_scalar(odd, odd, triple_sector(0, 2, 0)), 1.0);
    assert_eq!(rule.r_symbol_scalar(odd, odd, triple_sector(0, 2, 2)), -1.0);
    // R I3 (1,-1,1) (1,1,1) -> (0,0,0) = 1, -> (0,0,2) = -1
    let anti = triple_sector(1, -1, 1);
    assert_eq!(rule.r_symbol_scalar(anti, odd, triple_sector(0, 0, 0)), 1.0);
    assert_eq!(
        rule.r_symbol_scalar(anti, odd, triple_sector(0, 0, 2)),
        -1.0
    );
    // twist / fs / dim table.
    assert_eq!(rule.twist_scalar(triple_sector(0, 0, 0)), 1.0);
    assert_eq!(rule.twist_scalar(odd), -1.0);
    assert_eq!(rule.twist_scalar(triple_sector(0, 2, 0)), 1.0);
    assert_eq!(rule.twist_scalar(triple_sector(1, -1, 3)), -1.0);
    assert_eq!(rule.frobenius_schur_phase_scalar(odd), -1.0);
    assert_eq!(
        rule.frobenius_schur_phase_scalar(triple_sector(1, -1, 3)),
        -1.0
    );
    assert_eq!(rule.dim_scalar(triple_sector(1, -1, 3)), 4.0);
    assert_eq!(rule.dim_scalar(odd), 2.0);
}
