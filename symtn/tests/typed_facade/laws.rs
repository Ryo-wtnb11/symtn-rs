use super::*;

/// The U(1) oracle. `p` carries three charges with unequal degeneracies
/// and `q` two, so the two legs are distinguishable by shape alone; `q` is
/// dual, so the charge balance of a block is not symmetric under swapping the
/// legs either.
fn u1_oracle(runtime: &Runtime, first_value: f64) -> TensorMap<symtn::sector::U1FusionRule, f64> {
    let rule = Arc::new(symtn::sector::U1FusionRule);
    let typed_p = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (symtn::sector::U1Irrep::new(-1), 1),
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    // Built through `try_dual` because the dual flips the sector labels too.
    let typed_q = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (symtn::sector::U1Irrep::new(0), 1),
            (symtn::sector::U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    counter_oracle(runtime, (&typed_p, &typed_q), first_value)
}

/// The U(1) x fZ2 oracle. Both legs mix the two fermion parities with
/// nonzero U(1) charge, so neither the parity factor nor the charge factor is
/// constant on a leg — a product route that dropped either component would
/// still produce a nonempty layout, but not this one.
fn u1_fz2_oracle(runtime: &Runtime, first_value: f64) -> TensorMap<U1Fz2Rule, f64> {
    let rule = Arc::new(
        U1Fz2Rule::try_new(
            symtn::sector::U1FusionRule,
            symtn::sector::FermionParityFusionRule,
        )
        .unwrap(),
    );
    let label = |charge: i32, parity: u8| {
        symtn::sector::ProductSector::new(
            symtn::sector::U1Irrep::new(charge),
            if parity == 0 {
                symtn::sector::Z2Irrep::EVEN
            } else {
                symtn::sector::Z2Irrep::ODD
            },
        )
    };
    let typed_p =
        GradedSpace::try_new(Arc::clone(&rule), [(label(0, 0), 1), (label(1, 1), 2)]).unwrap();
    let typed_q = GradedSpace::try_new(
        Arc::clone(&rule),
        [(label(-1, 1), 1), (label(0, 0), 2), (label(1, 1), 1)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    counter_oracle(runtime, (&typed_p, &typed_q), first_value)
}

/// Asserts a nonzero result: a law on an empty or all-zero buffer is vacuous,
/// and the product routes are exactly where an
/// over-restrictive fusion could silently produce one.
fn assert_nonzero(what: &str, data: &[f64]) {
    assert!(
        data.iter().any(|value| *value != 0.0),
        "{what}: result is empty or all zero, so the law proves nothing"
    );
}

/// Upper bound on the floating terms reaching one entry of a contraction,
/// recoupled fusion trees included: an entry is bilinear in the operands, so
/// it has at most `len(lhs) * len(rhs)` distinct products.
fn bilinear_terms<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>) -> usize {
    lhs.dense_data().unwrap().len() * rhs.dense_data().unwrap().len()
}

/// Contract, compose, and compact-SVD laws shared by all three provider families.
fn assert_contract_compose_compact_laws_hold<R>(
    what: &str,
    nontrivial_twist: bool,
    typed: (&TensorMap<R, f64>, &TensorMap<R, f64>),
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let shapes = typed_leg_shapes(typed.0);
    assert_ne!(shapes[0], shapes[1], "{what}: p and q must differ");
    assert!(shapes[1].0, "{what}: q must be dual");
    assert_ne!(
        typed.0.dense_data().unwrap(),
        typed.1.dense_data().unwrap(),
        "{what}: operands must differ"
    );

    // Contracting the full domain against the other codomain leaves four open
    // legs. An explicit output order must equal the default typed contraction
    // followed by the corresponding public permutation.
    let (lhs_axes, rhs_axes) = ([2, 3], [0, 1]);
    let output_axes = [1, 3, 0, 2];
    let default_contract = typed
        .0
        .contract(
            typed.1,
            &ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap();
    let ordered_contract = typed
        .0
        .contract(
            typed.1,
            &ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &output_axes[..2],
                domain: &output_axes[2..],
            },
        )
        .unwrap();
    let permuted_default = default_contract.permute(&[1, 3], &[0, 2]).unwrap();
    assert_same_legs(&ordered_contract.codomain(), &permuted_default.codomain());
    assert_same_legs(&ordered_contract.domain(), &permuted_default.domain());
    // Routes that may recouple and accumulate in different orders: agreement
    // within the tolerance rule is the law, with the bilinear `terms` bound.
    let terms = bilinear_terms(typed.0, typed.1);
    numerics::assert_slices_close(
        &format!("{what}: order"),
        ordered_contract.dense_data().unwrap(),
        permuted_default.dense_data().unwrap(),
        terms,
    );
    assert!(
        std::ptr::eq(ordered_contract.provider(), typed.0.provider()),
        "{what}: contract lost left provider authority"
    );
    assert_nonzero(what, ordered_contract.dense_data().unwrap());
    assert_ne!(
        typed_leg_shapes(&default_contract),
        typed_leg_shapes(&ordered_contract),
        "{what}: the nonidentity output order did not move a leg"
    );
    assert_ne!(
        default_contract.dense_data().unwrap(),
        ordered_contract.dense_data().unwrap(),
        "{what}: the nonidentity output order left the buffer unchanged"
    );

    // Composition omits the supertrace twist of ordinary contraction. Twisting
    // exactly the right operand's dual codomain legs therefore relates the two.
    let dual_codomain_legs: Vec<usize> = typed
        .1
        .codomain()
        .iter()
        .enumerate()
        .filter_map(|(index, leg)| leg.is_dual().then_some(index))
        .collect();
    let twisted_right = typed
        .1
        .twist(&dual_codomain_legs, Direction::Forward)
        .unwrap();
    let composed = typed.0.compose(typed.1).unwrap();
    let twisted_contract = typed
        .0
        .contract(
            &twisted_right,
            &ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap();
    assert_same_legs(&composed.codomain(), &typed.0.codomain());
    assert_same_legs(&composed.domain(), &typed.1.domain());
    numerics::assert_slices_close(
        &format!("{what}: compose"),
        composed.dense_data().unwrap(),
        twisted_contract.dense_data().unwrap(),
        terms,
    );
    assert!(
        std::ptr::eq(composed.provider(), typed.0.provider()),
        "{what}: compose lost left provider authority"
    );
    let bound = numerics::tolerance::<f64>(
        terms,
        default_contract
            .dense_data()
            .unwrap()
            .iter()
            .fold(0.0, |max, v| v.abs().max(max)),
    );
    let differs = composed
        .dense_data()
        .unwrap()
        .iter()
        .zip(default_contract.dense_data().unwrap())
        .any(|(a, b)| (a - b).abs() > bound);
    assert_eq!(differs, nontrivial_twist, "{what}: compose twist control");
    assert_nonzero(what, composed.dense_data().unwrap());

    // Absorb the compact spectrum into U, then use the remaining Vh factor to
    // reconstruct the source. This tests the compact arm without comparing an
    // SVD gauge that is not public API.
    let Svd {
        u: typed_u,
        s: typed_s,
        vh: typed_vh,
    } = typed
        .0
        .svd_compact(&[0, 1], &[2, 3])
        .unwrap_or_else(|error| panic!("{what}: svd_compact failed: {error}"));
    let typed_absorbed = typed_u
        .contract(
            &typed_s,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    assert_same_legs(&typed_absorbed.codomain(), &typed_u.codomain());
    assert_same_legs(&typed_absorbed.domain(), &typed_s.domain());
    assert!(
        std::ptr::eq(typed_absorbed.provider(), typed.0.provider()),
        "{what}: absorption lost left provider authority"
    );
    assert_nonzero(what, typed_absorbed.dense_data().unwrap());
    assert_ne!(
        typed_absorbed.dense_data().unwrap(),
        typed_u.dense_data().unwrap(),
        "{what}: the diagonal factor was not applied"
    );
    let reconstructed = typed_absorbed
        .contract(
            &typed_vh,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap();
    assert_same_legs(&reconstructed.codomain(), &typed.0.codomain());
    assert_same_legs(&reconstructed.domain(), &typed.0.domain());
    assert_data_close_f64(
        reconstructed.dense_data().unwrap(),
        typed.0.dense_data().unwrap(),
    );
    assert!(std::ptr::eq(reconstructed.provider(), typed.0.provider()));
}

#[test]
fn contract_compose_and_compact_laws_hold_on_u1() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed_a = u1_oracle(&runtime, 1.0);
    let typed_b = u1_oracle(&runtime, 100.0);
    assert_contract_compose_compact_laws_hold(
        "U1, [p, q] <- [p, q] with q dual",
        false,
        (&typed_a, &typed_b),
    );
}

#[test]
fn contract_compose_and_compact_laws_hold_on_u1_fz2() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed_a = u1_fz2_oracle(&runtime, 1.0);
    let typed_b = u1_fz2_oracle(&runtime, 100.0);
    assert_contract_compose_compact_laws_hold(
        "U1 x fZ2, [p, q] <- [p, q] with q dual",
        true,
        (&typed_a, &typed_b),
    );
}

#[test]
fn contract_compose_and_compact_laws_hold_on_fz2_u1_su2() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed_a = fz2_u1_su2_oracle(&runtime, 1.0);
    let typed_b = fz2_u1_su2_oracle(&runtime, 100.0);
    assert_contract_compose_compact_laws_hold(
        "fZ2 x U1 x SU2, [p, q] <- [p, q] with q dual",
        true,
        (&typed_a, &typed_b),
    );
}

#[test]
fn the_fermionic_product_compose_is_contract_against_a_twisted_right_operand() {
    let _serial = crate::counting_alloc::serial();
    // What: the exact relation on `fZ2 x U(1) x SU(2)`,
    // `compose(a, b) == contract(a, twist(b, b's dual codomain legs))`. The
    // weaker `contract != compose` would pin only that *a* twist exists, not
    // which legs it acts on nor with which sign; this form pins all three, and
    // it is stated inside the product family so the suite does not lean on the
    // plain-fZ2 test above for it. A byte-parity oracle cannot do this
    // job at all: a twist deleted from a shared kernel moves both buffers
    // together. The bosonic family is the control — theta is one there, so the
    // twisted operand is the operand and the two contractions agree.
    let runtime = runtime();

    let fermionic_a = fz2_u1_su2_oracle(&runtime, 1.0);
    let fermionic_b = fz2_u1_su2_oracle(&runtime, 100.0);
    // `b`'s codomain is `[p, q]` and only `q` is dual, so theta acts on
    // codomain leg 1 alone — a "twist every contracted leg" reading would
    // multiply leg 0 too and fail here. Theta comes from the rule rather than
    // from an assumed `(-1)^F`: the product's twist is the product of its
    // factors', and hard-coding the parity sign here would quietly assume the
    // U(1) and SU(2) factors contribute one.
    let (leg_p, leg_q) = fz2_u1_su2_typed_legs();
    let rule = leg_p.provider().clone();
    let mut next = 99.0;
    let twisted_b = TensorMap::from_subblock_fn(
        &runtime,
        [&leg_p, &leg_q],
        [&leg_p, &leg_q],
        |sectors, _| {
            next += 1.0;
            let dual_leg = SectorCodec::encode_sector(&rule, &sectors.codomain_uncoupled()[1])
                .expect("the fixture's own labels encode");
            next * MultiplicityFreeRigidSymbols::twist_scalar(&rule, dual_leg)
        },
    )
    .unwrap();

    // The two routes may accumulate in different orders; within the tolerance
    // rule, with the bilinear `terms` bound.
    let terms = bilinear_terms(&fermionic_a, &fermionic_b);
    numerics::assert_slices_close(
        "fZ2 x U1 x SU2: compose is contract against the twisted right operand",
        fermionic_a
            .compose(&fermionic_b)
            .unwrap()
            .dense_data()
            .unwrap(),
        fermionic_a
            .contract(
                &twisted_b,
                &ContractSpec {
                    lhs: &[2, 3],
                    rhs: &[0, 1],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
            )
            .unwrap()
            .dense_data()
            .unwrap(),
        terms,
    );
    // And the twist is not vacuous: without it the two contractions differ.
    assert_ne!(
        fermionic_a
            .contract(
                &fermionic_b,
                &ContractSpec {
                    lhs: &[2, 3],
                    rhs: &[0, 1],
                    codomain: &[0, 1],
                    domain: &[2, 3]
                }
            )
            .unwrap()
            .dense_data()
            .unwrap(),
        fermionic_a
            .compose(&fermionic_b)
            .unwrap()
            .dense_data()
            .unwrap(),
        "fZ2 x U1 x SU2: contract and compose must differ on a dual contracted leg"
    );

    let bosonic_a = u1_oracle(&runtime, 1.0);
    let bosonic_b = u1_oracle(&runtime, 100.0);
    numerics::assert_slices_close(
        "U1: a bosonic rule has no twist, so the two agree",
        bosonic_a
            .contract(
                &bosonic_b,
                &ContractSpec {
                    lhs: &[2, 3],
                    rhs: &[0, 1],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
            )
            .unwrap()
            .dense_data()
            .unwrap(),
        bosonic_a.compose(&bosonic_b).unwrap().dense_data().unwrap(),
        bilinear_terms(&bosonic_a, &bosonic_b),
    );
}

/// The reduction and factorization half of the oracle matrix.
///
/// `dimension_weighted` says whether the family has a coupled sector with
/// `dim(c) != 1`: `inner` is TensorKit's quantum-dimension-weighted `dot`, so
/// only there does it differ from a plain sum of squares. Stating it per
/// family rather than deriving it keeps the fixture honest — if a fixture were
/// weakened to spin 0 everywhere, this flag would start lying and the test
/// would fail rather than quietly stop covering the weighted branch.
fn assert_reductions_and_factorizations_hold<R>(
    what: &str,
    dimension_weighted: bool,
    typed: (&TensorMap<R, f64>, &TensorMap<R, f64>),
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    // Neither coefficient is 1 and they differ in sign, so dropping or
    // swapping one moves the buffer.
    let typed_sum = typed.0.axpby(2.0, typed.1, -3.0).unwrap();
    for ((&actual, &left), &right) in typed_sum
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.0.dense_data().unwrap())
        .zip(typed.1.dense_data().unwrap())
    {
        assert_eq!(actual, 2.0 * left - 3.0 * right, "{what}: add");
    }
    assert_nonzero(what, typed_sum.dense_data().unwrap());
    assert_ne!(
        typed
            .0
            .axpby(-3.0, typed.1, 2.0)
            .unwrap()
            .dense_data()
            .unwrap(),
        typed_sum.dense_data().unwrap(),
        "{what}: add is not symmetric in its coefficients"
    );

    let typed_scaled = typed.0.scale(-2.5).unwrap();
    for (&actual, &source) in typed_scaled
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.0.dense_data().unwrap())
    {
        assert_eq!(actual, -2.5 * source, "{what}: scale");
    }
    assert_nonzero(what, typed_scaled.dense_data().unwrap());

    let typed_inner = typed.0.inner(typed.1).unwrap();
    let reverse_inner = typed.1.inner(typed.0).unwrap();
    assert!(
        (typed_inner - reverse_inner).abs() < 1e-12 * typed_inner.abs().max(1.0),
        "{what}: inner is not conjugate symmetric"
    );
    assert_ne!(typed_inner, 0.0, "{what}: inner is vacuously zero");
    // `<t, t>` is the squared weighted norm, which is the identity that pins
    // this weighting to `norm`'s.
    let norm = typed.0.norm(2.0).unwrap();
    let self_inner = typed.0.inner(typed.0).unwrap();
    assert!(
        (self_inner - norm * norm).abs() < 1e-9 * norm * norm,
        "{what}: <t, t> != norm^2"
    );
    // And the weighting is (or is not) visible against the unweighted sum of
    // squares, per the family.
    let unweighted: f64 = typed
        .0
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value * value)
        .sum();
    assert_eq!(
        (self_inner - unweighted).abs() > 1e-9 * unweighted,
        dimension_weighted,
        "{what}: dimension weighting expected {dimension_weighted}, \
         <t, t> = {self_inner}, sum of squares = {unweighted}"
    );

    let Qr {
        q: typed_q,
        r: typed_r,
    } = typed.0.qr_compact(&[0, 1], &[2, 3]).unwrap();
    assert_data_close_f64(
        typed_q.compose(&typed_r).unwrap().dense_data().unwrap(),
        typed.0.dense_data().unwrap(),
    );
    assert!(is_isometric!(typed_q, 1e-12), "{what}: qr q");
    assert_same_legs(&typed_q.codomain(), &typed.0.codomain());
    assert_same_legs(&typed_r.domain(), &typed.0.domain());
    assert_same_legs(&typed_q.domain(), &typed_r.codomain());
    assert_nonzero(what, typed_q.dense_data().unwrap());
    assert_nonzero(what, typed_r.dense_data().unwrap());

    let Lq {
        l: typed_c,
        q: typed_vh,
    } = typed.0.lq_compact(&[0, 1], &[2, 3]).unwrap();
    assert_data_close_f64(
        typed_c.compose(&typed_vh).unwrap().dense_data().unwrap(),
        typed.0.dense_data().unwrap(),
    );
    assert!(
        is_isometric!(typed_vh.adjoint().unwrap(), 1e-12),
        "{what}: lq q"
    );
    assert_same_legs(&typed_c.codomain(), &typed.0.codomain());
    assert_same_legs(&typed_vh.domain(), &typed.0.domain());
    assert_same_legs(&typed_c.domain(), &typed_vh.codomain());
    assert_nonzero(what, typed_c.dense_data().unwrap());
    assert_nonzero(what, typed_vh.dense_data().unwrap());

    let typed_spectrum = typed.0.svd_vals(&[0, 1], &[2, 3]).unwrap();
    assert!(
        typed_spectrum
            .windows(2)
            .all(|pair| pair[0].sector < pair[1].sector),
        "{what}: svd_vals labels are not strictly sorted"
    );
    assert!(
        typed_spectrum.iter().all(|entry| {
            entry.values.iter().all(|value| *value >= 0.0)
                && entry.values.windows(2).all(|pair| pair[0] >= pair[1])
        }),
        "{what}: svd_vals are not nonnegative and descending"
    );
    assert!(
        typed_spectrum
            .iter()
            .any(|entry| entry.values.iter().any(|value| *value != 0.0)),
        "{what}: svd_vals is vacuously zero"
    );
    let codomain = typed.0.codomain();
    let provider = codomain[0].provider();
    let spectral_weight: f64 = typed_spectrum
        .iter()
        .map(|entry| {
            let sector = SectorCodec::encode_sector(provider, &entry.sector).unwrap();
            provider.dim_scalar(sector)
                * entry.values.iter().map(|value| value * value).sum::<f64>()
        })
        .sum();
    assert!(
        (spectral_weight - self_inner).abs() < 1e-9 * self_inner.abs().max(1.0),
        "{what}: weighted sum(sigma^2) = {spectral_weight}, <t, t> = {self_inner}"
    );
}

#[test]
fn reductions_and_factorizations_hold_on_u1() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed_a = u1_oracle(&runtime, 1.0);
    let typed_b = u1_oracle(&runtime, 100.0);
    assert_reductions_and_factorizations_hold(
        "U1, [p, q] <- [p, q] with q dual",
        false,
        (&typed_a, &typed_b),
    );
}

#[test]
fn reductions_and_factorizations_hold_on_u1_fz2() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed_a = u1_fz2_oracle(&runtime, 1.0);
    let typed_b = u1_fz2_oracle(&runtime, 100.0);
    assert_reductions_and_factorizations_hold(
        "U1 x fZ2, [p, q] <- [p, q] with q dual",
        false,
        (&typed_a, &typed_b),
    );
}

#[test]
fn reductions_and_factorizations_hold_on_fz2_u1_su2() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed_a = fz2_u1_su2_oracle(&runtime, 1.0);
    let typed_b = fz2_u1_su2_oracle(&runtime, 100.0);
    // The only weighted family here: its SU(2) factor carries spin 1/2 and
    // spin 1, so `dim(c)` is not identically one.
    assert_reductions_and_factorizations_hold(
        "fZ2 x U1 x SU2, [p, q] <- [p, q] with q dual",
        true,
        (&typed_a, &typed_b),
    );
}

// ---------------------------------------------------------------------------
// Issue #610: the generic product provider as the canonical public route.
//
// Everything below is written the way a downstream user would have to write
// it: `ProductFusionRule::try_new` for the provider, `product_sector` for
// the label, `symtn::typed` for the space and the tensor. No
// fixed product-space constructor appears, so the tests fail — they do not
// silently reroute — if a group-specific constructor is the only working path.
// ---------------------------------------------------------------------------

/// `fZ2 ⊠ U(1)` built from the generic rule product, then run through a typed
/// operation chain: this is a public-path claim, not a sign or permutation
/// detector. Fermionic sign detection is owned by
/// `the_fermionic_product_compose_is_contract_against_a_twisted_right_operand`
/// earlier in this file.
#[test]
fn generic_product_provider_drives_the_typed_facade_without_a_fixed_constructor() {
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();

    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn::sector::FermionParityFusionRule,
            symtn::sector::U1FusionRule,
        )
        .unwrap(),
    );
    let label = |parity: symtn::sector::Z2Irrep, charge: i32| {
        symtn::sector::product_sector(parity, symtn::sector::U1Irrep::new(charge))
    };

    let p = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (label(symtn::sector::Z2Irrep::EVEN, 0), 2),
            (label(symtn::sector::Z2Irrep::ODD, 1), 1),
        ],
    )
    .unwrap();
    // Dual, so the dual-flag path — admission, dual sector resolution, block
    // layout, composition — runs on a product provider. The identities below
    // are sign-consistent and hold for either flag, so this leg is coverage of
    // that path, not a probe that makes them sign-sensitive.
    let q = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (label(symtn::sector::Z2Irrep::ODD, -1), 1),
            (label(symtn::sector::Z2Irrep::EVEN, 0), 2),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();

    // A distinct value per element, so multiple blocks are exercised
    // nontrivially.
    let mut next = 0.0_f64;
    let t: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&p, &q], [&p, &q], |_, _| {
        next += 1.0;
        next
    })
    .unwrap();

    assert!(
        t.subblock_count() > 1,
        "a single block would make the identities below near-vacuous"
    );
    let norm = t.norm(2.0).unwrap();
    assert!(norm > 0.0, "zero tensor: the assertions below are vacuous");

    // <t, t> = |t|^2 and tr(t† ∘ t) = |t|^2: three independent code paths
    // (weighted inner, reduction norm, composition + trace) over the product
    // provider's own fusion and dual data.
    assert!((t.inner(&t).unwrap() - norm * norm).abs() < 1e-9 * norm * norm);
    let gram_trace = t.adjoint().unwrap().compose(&t).unwrap().tr().unwrap();
    assert!((gram_trace - norm * norm).abs() < 1e-9 * norm * norm);
}

/// The recursive three-factor spelling: `(fZ2 ⊠ U(1)) ⊠ SU(2)` needs no new
/// core type, and its factor order and association are structure — of the Rust
/// type, of the label, and of the [`symtn::sector::RuleIdentity`] — never an
/// automatic equivalence.
#[test]
fn nested_three_factor_product_keeps_its_declared_factor_order() {
    let _serial = crate::counting_alloc::serial();
    let left_assoc = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                symtn::sector::FermionParityFusionRule,
                symtn::sector::U1FusionRule,
            )
            .unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let label = |parity: symtn::sector::Z2Irrep, charge: i32, twice_spin: usize| {
        symtn::sector::product_sector(
            symtn::sector::product_sector(parity, symtn::sector::U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };

    let declared = [
        (label(symtn::sector::Z2Irrep::ODD, 1, 1), 1),
        (label(symtn::sector::Z2Irrep::EVEN, 0, 2), 2),
    ];
    let v = GradedSpace::try_new(Arc::clone(&left_assoc), declared).unwrap();

    // Decoded labels come back nested exactly as declared: parity outermost
    // left, then charge, with the spin as the outer right factor. Compared as
    // `(label, degeneracy)` pairs — labels alone would not catch a label ↔
    // degeneracy mispairing — and as a set, because a leg is stored in
    // `SectorId` order, not declaration order.
    let mut decoded: Vec<_> = v
        .sectors()
        .unwrap()
        .into_iter()
        .zip(v.degeneracies().iter().copied())
        .collect();
    decoded.sort_unstable();
    let mut expected = declared.to_vec();
    expected.sort_unstable();
    assert_eq!(decoded, expected);
    let odd = decoded
        .iter()
        .map(|(label, _)| label)
        .find(|label| *label.left().left() == symtn::sector::Z2Irrep::ODD)
        .expect("the odd-parity label survives the round trip");
    assert_eq!(*odd.left().right(), symtn::sector::U1Irrep::new(1));
    assert_eq!(*odd.right(), SU2Irrep::from_twice_spin(1));

    // Association is not an equivalence: `fZ2 ⊠ (U(1) ⊠ SU(2))` is a different
    // provider with a different label type, and the identities do not compare
    // equal. The label type difference is what makes the line below compile
    // only when written for the right association.
    let right_assoc = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn::sector::FermionParityFusionRule,
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn::sector::U1FusionRule,
            SU2FusionRule,
        )
        .unwrap(),
    )
    .unwrap();
    let right_label = symtn::sector::product_sector(
        symtn::sector::Z2Irrep::ODD,
        symtn::sector::product_sector(symtn::sector::U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
    );
    assert_ne!(
        left_assoc.rule_identity(),
        right_assoc.rule_identity(),
        "left- and right-associated products must stay distinct providers"
    );
    assert_eq!(*right_label.left(), symtn::sector::Z2Irrep::ODD);

    // Factor order is not an equivalence either: swapping the two factors of
    // the inner product gives a third distinct provider.
    let swapped = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn::sector::U1FusionRule,
            symtn::sector::FermionParityFusionRule,
        )
        .unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    assert_ne!(left_assoc.rule_identity(), swapped.rule_identity());
}
