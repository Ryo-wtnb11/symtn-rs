//! Public, by-value conformance checks for the built-in multiplicity-free providers.
//!
//! This deliberately covers ordinary structural operations only.  Factorizations
//! and matrix functions have their own #1002 gates.

use std::sync::Arc;
use symtn::typed::{ContractSpec, Direction, Duality, Side};

use symtn::sector::{
    product_sector, CU1FusionRule, CU1Irrep, FermionParityFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
};
use symtn::typed::{Complex64, GradedSpace, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

#[test]
fn zn3_index_flip_and_units_keep_the_original_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge = |value| provider.irrep(value);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(charge(0), 1), (charge(1), 1), (charge(2), 1)],
    )
    .unwrap();
    let source: TensorMap<ZNFusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |sectors, _| {
            10.0 + sectors.coupled().charge() as f64
        })
        .unwrap();

    // Independent label/value oracle: the three diagonal charge blocks are
    // ordered by their Z3 charge, not merely restored by an inverse operation.
    assert_eq!(source.dense_data().unwrap(), &[10.0, 11.0, 12.0]);
    assert_eq!(
        (0..source.subblock_count())
            .map(|i| *source.subblock_fusion_trees(i).unwrap().coupled())
            .collect::<Vec<_>>(),
        vec![charge(0), charge(1), charge(2)]
    );

    let flipped = source.flip(engine, &[0], Direction::Forward).unwrap();
    assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        flipped.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert!(std::ptr::eq(
        flipped.domain()[0].provider(),
        provider.as_ref()
    ));
    assert!(flipped.codomain()[0].is_dual());
    assert_eq!(flipped.dense_data().unwrap(), source.dense_data().unwrap());
    assert_eq!(
        flipped
            .flip(engine, &[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        source.dense_data().unwrap()
    );

    let inserted = source
        .insert_unit(engine, 1, Side::Domain, Duality::Dual)
        .unwrap();
    assert!(std::ptr::eq(inserted.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        inserted.domain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(inserted.domain()[0].sectors().unwrap(), vec![charge(0)]);
    let restored = inserted.remove_unit(engine, 1).unwrap();
    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert_eq!(restored.dense_data().unwrap(), source.dense_data().unwrap());
}

#[test]
fn z2_cat_and_absorb_have_hand_computed_slabs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(Z2FusionRule);
    let codomain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap();
    let lhs_domain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 1)]).unwrap();
    let rhs_domain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap();
    let lhs: TensorMap<Z2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&lhs_domain], |_, i| {
            (i[0] + 1) as f64
        })
        .unwrap();
    let rhs: TensorMap<Z2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&rhs_domain], |_, i| {
            (3 + i[0] + 2 * i[1]) as f64
        })
        .unwrap();
    let joined = lhs.cat(engine, &rhs, Side::Domain).unwrap();
    assert!(std::ptr::eq(joined.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        joined.domain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(
        joined.dense_data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(joined.domain()[0].degeneracies(), &[3]);

    let destination: TensorMap<Z2FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap()],
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 3)]).unwrap()],
        |_, i| (10 * (i[0] + 1) + i[1] + 1) as f64,
    )
    .unwrap();
    let source: TensorMap<Z2FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 3)]).unwrap()],
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap()],
        |_, i| -((10 * (i[0] + 1) + i[1] + 1) as f64),
    )
    .unwrap();
    let absorbed = destination.absorb(engine, &source).unwrap();
    assert!(std::ptr::eq(absorbed.provider(), provider.as_ref()));
    assert_eq!(
        absorbed.dense_data().unwrap(),
        &[-11.0, -21.0, -12.0, -22.0, 13.0, 23.0]
    );
}

#[test]
fn fermionic_product_contract_otimes_and_reductions_keep_provider_and_signs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(odd, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 3.0).unwrap();

    let contracted = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert!(std::ptr::eq(contracted.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        contracted.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(contracted.dense_data().unwrap(), &[6.0]);
    let product = lhs.otimes(engine, &rhs).unwrap();
    assert!(std::ptr::eq(product.provider(), provider.as_ref()));
    assert_eq!(product.dense_data().unwrap(), &[6.0]);
    assert_eq!(lhs.inner(engine, &rhs).unwrap(), 6.0);
    assert_eq!(lhs.norm(engine, 2.0).unwrap(), 2.0);
    assert_eq!(lhs.tr(engine).unwrap(), 2.0);
}

#[test]
fn nested_fermionic_su2_product_and_complex_adjoint_are_publicly_conformant() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule,
                U1FusionRule,
            )
            .unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let spin_half = product_sector(
        product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
        SU2Irrep::from_twice_spin(1),
    );
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(spin_half, 1)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex64::new(2.0, 3.0)
        })
        .unwrap();
    let adjoint = source.adjoint(engine).unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        adjoint.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert!(std::ptr::eq(
        adjoint.domain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(
        source
            .twist(engine, &[0, 1], Direction::Forward)
            .unwrap()
            .dense_data()
            .unwrap(),
        source.dense_data().unwrap()
    );
    assert_eq!(
        adjoint.materialize(engine).unwrap().dense_data().unwrap(),
        &[Complex64::new(2.0, -3.0)]
    );
    assert_eq!(
        adjoint.codomain()[0].sectors().unwrap(),
        vec![spin_half],
        "the nonzero SU(2) label survives the public product route"
    );
}

#[test]
fn zn3_extended_structural_paths_execute_on_the_original_arc() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge = |value| provider.irrep(value);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(charge(0), 1), (charge(1), 2), (charge(2), 1)],
    )
    .unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, index| {
            Complex64::new(
                100.0 * trees.coupled().charge() as f64 + 10.0 * index[0] as f64 + index[1] as f64,
                index[2] as f64,
            )
        })
        .unwrap();
    let permuted = source.permute(engine, &[1, 0], &[2]).unwrap();
    let braided = source.braid(engine, &[1, 0], &[2], &[1, 0, 2]).unwrap();
    assert_eq!(
        braided.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "Z3 is bosonic"
    );
    assert_ne!(
        permuted.dense_data().unwrap(),
        source.dense_data().unwrap(),
        "asymmetric raw payload moved"
    );
    let restored = permuted.permute(engine, &[1, 0], &[2]).unwrap();
    assert_eq!(restored.dense_data().unwrap(), source.dense_data().unwrap());
    for index in 0..source.subblock_count() {
        assert_eq!(
            restored.subblock_fusion_trees(index).unwrap(),
            source.subblock_fusion_trees(index).unwrap()
        );
    }
    let twisted = source
        .twist(engine, &[0, 1, 2], Direction::Forward)
        .unwrap();
    assert_eq!(
        twisted.dense_data().unwrap(),
        source.dense_data().unwrap(),
        "ZN(3) has trivial twist"
    );
    assert_eq!(
        twisted.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
    for output in [
        source.adjoint(engine).unwrap(),
        source.transpose(engine, &[2], &[1, 0]).unwrap(),
        source.repartition(engine, 1).unwrap(),
        braided,
        source
            .twist(engine, &[0, 1, 2], Direction::Forward)
            .unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    }
    let adjoint = source.adjoint(engine).unwrap();
    assert!(std::ptr::eq(
        adjoint.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert!(std::ptr::eq(
        adjoint.domain()[0].provider(),
        provider.as_ref()
    ));
}

#[test]
fn cu1_charged_structural_paths_keep_the_original_arc() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(CU1FusionRule);
    let charged = CU1Irrep::from_twice_charge(1);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(charged, 1)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex64::new(2.0, 3.0)
        })
        .unwrap();
    for output in [
        source.adjoint(engine).unwrap(),
        source.twist(engine, &[0, 1], Direction::Forward).unwrap(),
        source.flip(engine, &[0], Direction::Forward).unwrap(),
        source
            .insert_unit(engine, 0, Side::Codomain, Duality::Plain)
            .unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
        assert!(std::ptr::eq(
            output.codomain()[0].provider(),
            provider.as_ref()
        ));
    }
    assert_eq!(
        source
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[Complex64::new(2.0, -3.0)]
    );
    assert_eq!(
        source
            .twist(engine, &[0, 1], Direction::Forward)
            .unwrap()
            .dense_data()
            .unwrap(),
        source.dense_data().unwrap()
    );
    assert_eq!(
        source
            .flip(engine, &[0], Direction::Forward)
            .unwrap()
            .dense_data()
            .unwrap(),
        source.dense_data().unwrap()
    );
    let pseudo =
        GradedSpace::try_new(Arc::clone(&provider), [(CU1Irrep::PSEUDOSCALAR, 1)]).unwrap();
    let braid_source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&pseudo], |_, _| 1.0).unwrap();
    let permuted = braid_source.permute(engine, &[1, 0], &[2]).unwrap();
    let braided = braid_source
        .braid(engine, &[1, 0], &[2], &[0, 1, 2])
        .unwrap();
    // `permute` is the symmetric-braiding permutation (not a raw ndarray
    // transpose), so it and `braid` coincide for CU1.  Both differ from the
    // raw source by the charged exchange coefficient R(q,q;pseudo) = -1.
    assert_eq!(permuted.dense_data().unwrap(), &[-1.0]);
    assert_eq!(braided.dense_data().unwrap(), &[-1.0]);
    assert!(std::ptr::eq(braided.provider(), provider.as_ref()));
    let inserted = source
        .insert_unit(engine, 0, Side::Codomain, Duality::Plain)
        .unwrap();
    assert!(inserted
        .codomain()
        .into_iter()
        .chain(inserted.domain())
        .any(|leg| leg.sectors().unwrap() == vec![CU1Irrep::VACUUM]));
}

#[test]
fn su2_and_exact_products_keep_their_provider_through_flip_and_units() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    macro_rules! check {
        ($engine:ident, $provider:expr, $label:expr) => {{
            let provider = Arc::new($provider);
            let leg = GradedSpace::try_new(Arc::clone(&provider), [($label, 1)]).unwrap();
            let source: TensorMap<_, f64> =
                TensorMap::from_subblock_fn($engine, &runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
            let flipped = source.flip($engine, &[1], Direction::Forward).unwrap();
            let inserted = source
                .insert_unit($engine, 1, Side::Domain, Duality::Dual)
                .unwrap();
            let restored = inserted.remove_unit($engine, 1).unwrap();
            for output in [&flipped, &inserted, &restored] {
                assert!(std::ptr::eq(output.provider(), provider.as_ref()));
                assert!(std::ptr::eq(
                    output.codomain()[0].provider(),
                    provider.as_ref()
                ));
            }
            assert_eq!(restored.dense_data().unwrap(), source.dense_data().unwrap());
        }};
    }
    check!(engine, SU2FusionRule, SU2Irrep::from_twice_spin(1));
    check!(
        engine,
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
        product_sector(Z2Irrep::ODD, U1Irrep::new(1))
    );
    check!(
        engine,
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule,
                U1FusionRule
            )
            .unwrap(),
            SU2FusionRule
        )
        .unwrap(),
        product_sector(
            product_sector(Z2Irrep::ODD, U1Irrep::new(0)),
            SU2Irrep::from_twice_spin(1)
        )
    );
}

#[test]
fn dual_nonabelian_flip_pins_the_pivotal_phase() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    macro_rules! check {
        ($engine:ident, $provider:expr, $label:expr, $axis:expr, $codomain_dual:expr, $domain_dual:expr) => {{
            let provider = Arc::new($provider);
            let plain = GradedSpace::try_new(Arc::clone(&provider), [($label, 1)]).unwrap();
            let dual = plain.try_dual().unwrap();
            let source: TensorMap<_, f64> =
                TensorMap::from_subblock_fn($engine, &runtime, [&dual], [&plain], |_, _| 1.0)
                    .unwrap();
            let flipped = source.flip($engine, &[$axis], Direction::Forward).unwrap();
            assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
            assert!(std::ptr::eq(
                flipped.codomain()[0].provider(),
                provider.as_ref()
            ));
            assert!(std::ptr::eq(
                flipped.domain()[0].provider(),
                provider.as_ref()
            ));
            assert_eq!(flipped.codomain()[0].is_dual(), $codomain_dual);
            assert_eq!(flipped.domain()[0].is_dual(), $domain_dual);
            assert_eq!(flipped.dense_data().unwrap(), &[-1.0]);
        }};
    }
    // Dual SU2 codomain: the forward factor is χθ=-1.
    check!(
        engine,
        SU2FusionRule,
        SU2Irrep::from_twice_spin(1),
        0,
        false,
        false
    );
    check!(
        engine,
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule,
                U1FusionRule
            )
            .unwrap(),
            SU2FusionRule
        )
        .unwrap(),
        product_sector(
            product_sector(Z2Irrep::ODD, U1Irrep::new(0)),
            SU2Irrep::from_twice_spin(1)
        ),
        // Plain nested domain: its forward factor is θ=-1.
        1,
        true,
        true
    );
}

#[test]
fn covered_builtin_multiplicity_free_providers_have_cat_and_absorb_execution() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    macro_rules! check {
        ($engine:ident, $provider:expr, $label:expr) => {{
            let provider = Arc::new($provider);
            let codomain = GradedSpace::try_new(Arc::clone(&provider), [($label, 1)]).unwrap();
            let left = GradedSpace::try_new(Arc::clone(&provider), [($label, 1)]).unwrap();
            let right = GradedSpace::try_new(Arc::clone(&provider), [($label, 1)]).unwrap();
            let a: TensorMap<_, f64> =
                TensorMap::from_subblock_fn($engine, &runtime, [&codomain], [&left], |_, _| 1.0)
                    .unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::from_subblock_fn($engine, &runtime, [&codomain], [&right], |_, _| 2.0)
                    .unwrap();
            let domain = a.cat($engine, &b, Side::Domain).unwrap();
            let b_adjoint = &b.adjoint($engine).unwrap();
            let codomain_join = a
                .adjoint($engine)
                .unwrap()
                .cat($engine, b_adjoint, Side::Codomain)
                .unwrap();
            let absorbed = a.absorb($engine, &b).unwrap();
            for output in [&domain, &codomain_join, &absorbed] {
                assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            }
            // Single-sector column and row slabs, plus overwrite, are raw
            // payload oracles independent of a round trip.
            assert_eq!(domain.dense_data().unwrap(), &[1.0, 2.0]);
            assert_eq!(codomain_join.dense_data().unwrap(), &[1.0, 2.0]);
            assert_eq!(absorbed.dense_data().unwrap(), &[2.0]);
        }};
    }
    check!(engine, Z2FusionRule, Z2Irrep::EVEN);
    check!(
        engine,
        ZNFusionRule::new(3).unwrap(),
        ZNFusionRule::new(3).unwrap().irrep(1)
    );
    check!(engine, CU1FusionRule, CU1Irrep::from_twice_charge(1));
    check!(engine, FermionParityFusionRule, Z2Irrep::ODD);
    check!(
        engine,
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
        product_sector(Z2Irrep::ODD, U1Irrep::new(1))
    );
    check!(engine, SU2FusionRule, SU2Irrep::from_twice_spin(1));
    check!(
        engine,
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule,
                U1FusionRule
            )
            .unwrap(),
            SU2FusionRule
        )
        .unwrap(),
        product_sector(
            product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
            SU2Irrep::from_twice_spin(1)
        )
    );
}

#[test]
fn zn3_and_cu1_arithmetic_contraction_and_reductions_have_scalar_oracles() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    macro_rules! check {
        ($engine:ident, $provider:expr, $label:expr, $qdim:expr) => {{
            let provider = Arc::new($provider);
            let leg = GradedSpace::try_new(Arc::clone(&provider), [($label, 1)]).unwrap();
            let a: TensorMap<_, f64> =
                TensorMap::from_subblock_fn($engine, &runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::from_subblock_fn($engine, &runtime, [&leg], [&leg], |_, _| 3.0).unwrap();
            let ordered = a
                .contract(
                    $engine,
                    &b,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[1],
                        domain: &[0],
                    },
                )
                .unwrap();
            let composed = a.compose($engine, &b).unwrap();
            let tensor_product = a.otimes($engine, &b).unwrap();
            let sum = a.axpby($engine, 1.0, &b, -1.0).unwrap();
            let scaled = a.scale($engine, 4.0).unwrap();
            for output in [&ordered, &composed, &tensor_product, &sum, &scaled] {
                assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            }
            for values in [
                ordered.dense_data().unwrap(),
                composed.dense_data().unwrap(),
                tensor_product.dense_data().unwrap(),
            ] {
                assert!((values[0] - 6.0).abs() < 1e-12);
            }
            assert_eq!(sum.dense_data().unwrap(), &[-1.0]);
            assert_eq!(scaled.dense_data().unwrap(), &[8.0]);
            assert!((a.norm($engine, 2.0).unwrap() - 2.0 * ($qdim as f64).sqrt()).abs() < 1e-12);
            assert!((a.inner($engine, &b).unwrap() - 6.0 * $qdim).abs() < 1e-12);
            assert!((a.tr($engine).unwrap() - 2.0 * $qdim).abs() < 1e-12);
            assert!(
                (a.trace_pairs($engine, &[(0, 1)])
                    .unwrap()
                    .scalar($engine)
                    .unwrap()
                    - 2.0 * $qdim)
                    .abs()
                    < 1e-12
            );
        }};
    }
    check!(
        engine,
        ZNFusionRule::new(3).unwrap(),
        ZNFusionRule::new(3).unwrap().irrep(1),
        1.0
    );
    check!(engine, CU1FusionRule, CU1Irrep::from_twice_charge(1), 2.0);

    let provider = Arc::new(FermionParityFusionRule);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let a: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let b: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 3.0).unwrap();
    assert_eq!(a.inner(engine, &b).unwrap(), 6.0);
}
