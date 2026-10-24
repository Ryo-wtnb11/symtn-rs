//! Lie global forms and low-rank representation oracles, independent of admission code.
//!
//! Reference production paths: TensorKitSectors 0.3.10 tree
//! b8912f258633c7651e0bf1d2a2bab12fc7eab412, su2irrep.jl (dim/N/F/R),
//! SUNRepresentations 0.4.0 tree 1969df89e1621be7e20775cdd988ec39c2f6c0dc,
//! sector.jl (CGC contractions), and QSpace
//! d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371, Source/clebsch_aux.cc,
//! DY::wdim_{A,B,C,D}. The literals below are standard defining/spinor products;
//! braid traces are permutation characters, so do not depend on CGC gauge.

#![cfg(feature = "cgc-gen")]

use std::collections::BTreeMap;

use fusion_symbols::category::FusionCategory;
use fusion_symbols::lie::{GlobalForm, GroupId, LieError, RootSystem};
use num_complex::{Complex32, Complex64};
use symtn_sectors::{
    BraidingStyleKind, CategoryProvider, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, LieGroupFusionRule, LieGroupFusionRuleError,
    SectorId,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn rule(group: GroupId) -> LieGroupFusionRule {
    LieGroupFusionRule::new(group).unwrap()
}

#[test]
fn constructor_global_form_matrix_and_all_codec_routes() {
    // Hard-coded center characters: these expectations deliberately do not call admits.
    let cases: Vec<(GroupId, Vec<i64>, Vec<Vec<i64>>)> = vec![
        (GroupId::su(3).unwrap(), vec![1, 0], vec![]),
        (
            GroupId::su_quotient(4, 2).unwrap(),
            vec![0, 1, 0],
            vec![vec![1, 0, 0]],
        ),
        (GroupId::psu(3).unwrap(), vec![1, 1], vec![vec![1, 0]]),
        (GroupId::sp(4).unwrap(), vec![1, 0], vec![]),
        (GroupId::psp(4).unwrap(), vec![0, 1], vec![vec![1, 0]]),
        (GroupId::spin(5).unwrap(), vec![0, 1], vec![]),
        (GroupId::so(5).unwrap(), vec![1, 0], vec![vec![0, 1]]),
        (GroupId::pso(5).unwrap(), vec![1, 0], vec![vec![0, 1]]),
        (GroupId::so(3).unwrap(), vec![2], vec![vec![1]]),
        (GroupId::psp(2).unwrap(), vec![2], vec![vec![1]]),
        (GroupId::psu(2).unwrap(), vec![2], vec![vec![1]]),
        (
            GroupId::so(6).unwrap(),
            vec![1, 0, 0],
            vec![vec![0, 1, 0], vec![0, 0, 1]],
        ),
        (
            GroupId::pso(6).unwrap(),
            vec![0, 1, 1],
            vec![vec![1, 0, 0], vec![0, 1, 0]],
        ),
        (
            GroupId::so(8).unwrap(),
            vec![1, 0, 0, 0],
            vec![vec![0, 0, 1, 0], vec![0, 0, 0, 1]],
        ),
        (
            GroupId::pso(8).unwrap(),
            vec![0, 1, 0, 0],
            vec![vec![1, 0, 0, 0], vec![0, 0, 1, 0], vec![0, 0, 0, 1]],
        ),
        (
            GroupId::half_spin_plus(8).unwrap(),
            vec![0, 0, 0, 1],
            vec![vec![0, 0, 1, 0], vec![1, 0, 0, 0]],
        ),
        (
            GroupId::half_spin_minus(8).unwrap(),
            vec![0, 0, 1, 0],
            vec![vec![0, 0, 0, 1], vec![1, 0, 0, 0]],
        ),
        (
            GroupId::so(12).unwrap(),
            vec![1, 0, 0, 0, 0, 0],
            vec![vec![0, 0, 0, 0, 1, 0], vec![0, 0, 0, 0, 0, 1]],
        ),
        (
            GroupId::pso(12).unwrap(),
            vec![0, 1, 0, 0, 0, 0],
            vec![
                vec![1, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 1, 0],
                vec![0, 0, 0, 0, 0, 1],
            ],
        ),
        (
            GroupId::half_spin_plus(12).unwrap(),
            vec![0, 0, 0, 0, 0, 1],
            vec![vec![0, 0, 0, 0, 1, 0], vec![1, 0, 0, 0, 0, 0]],
        ),
        (
            GroupId::half_spin_minus(12).unwrap(),
            vec![0, 0, 0, 0, 1, 0],
            vec![vec![0, 0, 0, 0, 0, 1], vec![1, 0, 0, 0, 0, 0]],
        ),
    ];
    for (group, accepted, rejected) in cases {
        let r = rule(group);
        assert_eq!(r.group_id(), group);
        assert_eq!(r.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(r.braiding_style(), BraidingStyleKind::Bosonic);
        let id = r.encode_dynkin(&accepted).unwrap();
        assert_eq!(r.decode_dynkin(id).unwrap(), accepted);
        assert_eq!(r.sector_id(&r.object(id).unwrap()).unwrap(), id);
        assert!(matches!(
            r.encode_dynkin(&[]),
            Err(LieGroupFusionRuleError::LabelLength { .. })
        ));
        let mut negative = accepted.clone();
        negative[0] = -1;
        assert!(matches!(
            r.encode_dynkin(&negative),
            Err(LieGroupFusionRuleError::NegativeLabel {
                index: 0,
                value: -1
            })
        ));
        let cover = rule(GroupId {
            form: GlobalForm::SimplyConnected,
            ..group
        });
        for label in rejected {
            let raw_id = cover.encode_dynkin(&label).unwrap();
            assert!(
                matches!(
                    r.encode_dynkin(&label),
                    Err(LieGroupFusionRuleError::Provider(LieError::NotAdmissible))
                ),
                "{group:?}: {label:?}"
            );
            assert!(
                matches!(
                    r.decode_dynkin(raw_id),
                    Err(LieGroupFusionRuleError::Provider(LieError::NotAdmissible))
                ),
                "{group:?}: raw {label:?}"
            );
            assert!(matches!(
                r.object(raw_id),
                Err(LieGroupFusionRuleError::Provider(LieError::NotAdmissible))
            ));
            // A cover object belongs to a different full GroupId: reject it before
            // looking up labels, even if those labels were interned previously.
            assert!(matches!(
                r.sector_id(&cover.object(raw_id).unwrap()),
                Err(LieGroupFusionRuleError::Provider(LieError::InvalidLabel))
            ));
        }
    }
    assert_eq!(GroupId::so(5).unwrap(), GroupId::pso(5).unwrap());
    for group in [
        GroupId::spin(4).unwrap(),
        GroupId::so(4).unwrap(),
        GroupId::pso(4).unwrap(),
    ] {
        assert!(matches!(
            LieGroupFusionRule::new(group),
            Err(LieGroupFusionRuleError::Provider(
                LieError::Unsupported { .. }
            ))
        ));
    }
    for root_system in [
        RootSystem::G2,
        RootSystem::F4,
        RootSystem::E6,
        RootSystem::E7,
        RootSystem::E8,
    ] {
        assert!(matches!(
            LieGroupFusionRule::new(GroupId {
                root_system,
                form: GlobalForm::SimplyConnected
            }),
            Err(LieGroupFusionRuleError::Provider(
                LieError::Unsupported { .. }
            ))
        ));
    }
}

fn fusion(r: &LieGroupFusionRule, a: &[i64], b: &[i64]) -> BTreeMap<Vec<i64>, usize> {
    let (a, b) = (r.encode_dynkin(a).unwrap(), r.encode_dynkin(b).unwrap());
    r.try_fusion_channels(a, b)
        .unwrap()
        .into_iter()
        .map(|c| (r.decode_dynkin(c).unwrap(), r.try_nsymbol(a, b, c).unwrap()))
        .collect()
}

fn exact_dimension(r: &LieGroupFusionRule, label: &[i64], dimension: u64) {
    let id = r.encode_dynkin(label).unwrap();
    assert_eq!(
        r.object(id).unwrap().dim_exact().to_string(),
        dimension.to_string()
    );
    assert_eq!(r.try_dim_scalar(id).unwrap(), dimension as f64);
}

#[test]
fn rank_one_isomorphisms_match_su2_triangle_oracle() {
    // TensorKitSectors su2irrep.jl: dim = 2j+1; triangle/parity N; self-dual.
    for group in [
        GroupId::su(2).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
    ] {
        let r = rule(group);
        for a in 0..=3 {
            exact_dimension(&r, &[a], a as u64 + 1);
            let id = r.encode_dynkin(&[a]).unwrap();
            assert_eq!(r.try_dual(id).unwrap(), id);
            for b in 0..=3 {
                let expected = ((a - b).abs()..=a + b)
                    .step_by(2)
                    .map(|c| (vec![c], 1))
                    .collect();
                assert_eq!(fusion(&r, &[a], &[b]), expected, "{group:?}: {a} x {b}");
            }
        }
    }
}

#[test]
fn b2_c2_isomorphism_has_fixed_swapped_labels_and_complete_products() {
    let b = rule(GroupId::spin(5).unwrap());
    let c = rule(GroupId::sp(4).unwrap());
    for (label, dim) in [
        ([0, 0], 1),
        ([0, 1], 4),
        ([1, 0], 5),
        ([0, 2], 10),
        ([2, 0], 14),
        ([1, 1], 16),
    ] {
        exact_dimension(&b, &label, dim);
        exact_dimension(&c, &[label[1], label[0]], dim);
        for (r, l) in [(&b, label), (&c, [label[1], label[0]])] {
            let id = r.encode_dynkin(&l).unwrap();
            assert_eq!(r.try_dual(id).unwrap(), id);
        }
    }
    for (a, d, outcomes) in [
        ([0, 1], [0, 1], vec![[0, 0], [1, 0], [0, 2]]),
        ([1, 0], [1, 0], vec![[0, 0], [0, 2], [2, 0]]),
        ([0, 1], [1, 0], vec![[0, 1], [1, 1]]),
    ] {
        let expected: BTreeMap<_, _> = outcomes.iter().map(|x| (x.to_vec(), 1)).collect();
        assert_eq!(fusion(&b, &a, &d), expected);
        let expected = outcomes.iter().map(|x| (vec![x[1], x[0]], 1)).collect();
        assert_eq!(fusion(&c, &[a[1], a[0]], &[d[1], d[0]]), expected);
    }
}

#[test]
fn d3_a3_isomorphism_has_fixed_cyclic_labels_products_and_duals() {
    let d = rule(GroupId::spin(6).unwrap());
    let a = rule(GroupId::su(4).unwrap());
    let map = |x: [i64; 3]| [x[2], x[0], x[1]];
    for (label, dim) in [
        ([0, 0, 0], 1),
        ([1, 0, 0], 6),
        ([0, 0, 1], 4),
        ([0, 1, 0], 4),
        ([0, 1, 1], 15),
        ([2, 0, 0], 20),
        ([0, 0, 2], 10),
        ([1, 0, 1], 20),
    ] {
        exact_dimension(&d, &label, dim);
        exact_dimension(&a, &map(label), dim);
        let dual = [label[0], label[2], label[1]];
        assert_eq!(
            d.decode_dynkin(d.try_dual(d.encode_dynkin(&label).unwrap()).unwrap())
                .unwrap(),
            dual
        );
        assert_eq!(
            a.decode_dynkin(a.try_dual(a.encode_dynkin(&map(label)).unwrap()).unwrap())
                .unwrap(),
            map(dual)
        );
    }
    for (x, y, outcomes) in [
        ([1, 0, 0], [1, 0, 0], vec![[0, 0, 0], [0, 1, 1], [2, 0, 0]]),
        ([0, 0, 1], [0, 0, 1], vec![[1, 0, 0], [0, 0, 2]]),
        ([0, 0, 1], [0, 1, 0], vec![[0, 0, 0], [0, 1, 1]]),
        ([1, 0, 0], [0, 0, 1], vec![[0, 1, 0], [1, 0, 1]]),
    ] {
        assert_eq!(
            fusion(&d, &x, &y),
            outcomes.iter().map(|x| (x.to_vec(), 1)).collect()
        );
        assert_eq!(
            fusion(&a, &map(x), &map(y)),
            outcomes.iter().map(|&x| (map(x).to_vec(), 1)).collect()
        );
    }
}

#[test]
fn codec_holes_width_foreign_objects_and_full_group_identity() {
    let su2 = rule(GroupId::su(2).unwrap());
    assert_eq!(
        su2.encode_dynkin(&[u32::MAX as i64]).unwrap().id(),
        u32::MAX as usize
    );
    assert!(matches!(
        su2.encode_dynkin(&[u32::MAX as i64 + 1]),
        Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
    ));
    if usize::BITS > 32 {
        assert!(matches!(
            su2.decode_dynkin(SectorId::new(u32::MAX as usize + 1)),
            Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
        ));
    }
    for group in [
        GroupId::su(3).unwrap(),
        GroupId::sp(4).unwrap(),
        GroupId::spin(6).unwrap(),
    ] {
        let r = rule(group);
        for raw in 0..100 {
            let id = SectorId::new(raw);
            assert_eq!(r.encode_dynkin(&r.decode_dynkin(id).unwrap()).unwrap(), id);
        }
        assert!(matches!(
            r.encode_dynkin(&vec![i64::MAX; group.root_system.rank()]),
            Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
        ));
    }
    let quotient = rule(GroupId::psu(2).unwrap());
    assert_eq!(quotient.encode_dynkin(&[2]).unwrap(), SectorId::new(2));
    assert!(matches!(
        quotient.decode_dynkin(SectorId::new(1)),
        Err(LieGroupFusionRuleError::Provider(LieError::NotAdmissible))
    ));
    let warm = quotient.encode_dynkin(&[2]).unwrap();
    let member = quotient.object(warm).unwrap();
    assert_eq!(quotient.sector_id(&member).unwrap(), warm);
    assert!(matches!(
        quotient.sector_id(&su2.object(warm).unwrap()),
        Err(LieGroupFusionRuleError::Provider(LieError::InvalidLabel))
    ));
    let ids = [
        GroupId::su(2).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
    ];
    let rules = ids.map(rule);
    for i in 0..rules.len() {
        assert_eq!(rules[i].rule_identity(), rule(ids[i]).rule_identity());
        for j in 0..i {
            assert_eq!(
                rules[i].category().authority_fingerprint(),
                rules[j].category().authority_fingerprint()
            );
            assert_ne!(rules[i], rules[j]);
            assert_ne!(rules[i].rule_identity(), rules[j].rule_identity());
            let id = rules[i].encode_dynkin(&[1]).unwrap();
            assert!(matches!(
                rules[i].sector_id(&rules[j].object(id).unwrap()),
                Err(LieGroupFusionRuleError::Provider(LieError::InvalidLabel))
            ));
        }
    }
    // Equal quotient subcategories still have distinct full GroupId metadata.
    assert_ne!(
        rule(GroupId::su_quotient(2, 1).unwrap()).rule_identity(),
        su2.rule_identity()
    );
}

fn orthogonal(what: &str, matrix: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let dot = (0..n).map(|k| matrix[i * n + k] * matrix[j * n + k]).sum();
            numerics::assert_close(what, dot, f64::from(i == j), n);
        }
    }
}

#[test]
fn gauge_invariant_closed_braid_traces_fs_and_twists() {
    // Analytic S3 characters on Hom(a,a^3): (dimension, transposition, 3-cycle).
    // This checks closed compositions, not a gauge-dependent trace of F alone.
    let cases = [
        (GroupId::su(2).unwrap(), vec![1], 2, 0.0, -1.0, -1.0),
        (GroupId::spin(3).unwrap(), vec![1], 2, 0.0, -1.0, -1.0),
        (GroupId::sp(2).unwrap(), vec![1], 2, 0.0, -1.0, -1.0),
        (GroupId::spin(5).unwrap(), vec![0, 1], 3, -1.0, 0.0, -1.0),
        (GroupId::sp(4).unwrap(), vec![1, 0], 3, -1.0, 0.0, -1.0),
        (GroupId::spin(6).unwrap(), vec![1, 0, 0], 3, 1.0, 0.0, 1.0),
        (GroupId::su(4).unwrap(), vec![0, 1, 0], 3, 1.0, 0.0, 1.0),
    ];
    for (group, label, n, trace, cycle, fs) in cases {
        let r = rule(group);
        let a = r.encode_dynkin(&label).unwrap();
        assert_eq!(
            r.category()
                .frobenius_schur_indicator(&r.object(a).unwrap()),
            fs
        );
        numerics::assert_close(
            "FS phase",
            r.try_frobenius_schur_phase_scalar(a).unwrap(),
            fs,
            1,
        );
        numerics::assert_close("bosonic twist", r.try_twist_scalar(a).unwrap(), 1.0, 1);
        let channels = r.try_fusion_channels(a, a).unwrap();
        assert_eq!(channels.len(), n);
        let mut f = vec![0.0; n * n];
        let mut exchanges = Vec::new();
        let mut weighted_trace = 0.0;
        for (i, &e) in channels.iter().enumerate() {
            assert_eq!(r.try_nsymbol(a, a, e).unwrap(), 1);
            let exchange = r.try_r_symbol_generic(a, a, e).unwrap();
            assert_eq!(exchange.shape(), (1, 1));
            exchanges.push(*exchange.get(0, 0));
            weighted_trace += r.try_dim_scalar(e).unwrap() * exchange.get(0, 0);
            for (j, &g) in channels.iter().enumerate() {
                let block = r.try_f_symbol_generic(a, a, a, a, e, g).unwrap();
                assert_eq!(block.shape(), (1, 1, 1, 1));
                f[i * n + j] = *block.get(0, 0, 0, 0);
            }
        }
        orthogonal("F F^T", &f, n);
        for &exchange in &exchanges {
            numerics::assert_close("inverse braid", exchange * exchange, 1.0, 1);
        }
        numerics::assert_close("trace sigma1", exchanges.iter().sum(), trace, n);
        // sigma2 = F diag(R) F^T; tr(sigma1 sigma2) has n^2 terms.
        let cycle_trace = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| exchanges[i] * f[i * n + j].powi(2) * exchanges[j])
            .sum();
        numerics::assert_close("trace sigma1 sigma2", cycle_trace, cycle, n * n);
        numerics::assert_close(
            "dimension-weighted R trace = twist",
            weighted_trace / r.try_dim_scalar(a).unwrap(),
            r.try_twist_scalar(a).unwrap(),
            n,
        );
    }
    let d = rule(GroupId::spin(6).unwrap());
    let spinor = d.encode_dynkin(&[0, 0, 1]).unwrap();
    assert_ne!(d.try_dual(spinor).unwrap(), spinor);
    assert_eq!(
        d.category()
            .frobenius_schur_indicator(&d.object(spinor).unwrap()),
        0.0
    );
    numerics::assert_close(
        "complex irrep twist",
        d.try_twist_scalar(spinor).unwrap(),
        1.0,
        1,
    );
}

#[test]
fn multiplicity_two_sp4_and_so6_have_nonscalar_f_r_a_b() {
    // Independent C2 character derivation: chi16 = chi4*chi5-chi4, with
    // weights(4)=(+/-1,0),(0,+/-1), weights(5)=(+/-1,+/-1),(0,0).
    // Multiply chi16^2 by prod(1-exp(-alpha)) for positive roots
    // (1,-1),(0,2),(1,1),(2,0); dominant exponents x>=y>=0 give [x-y,y].
    // Dimensions use (a+1)(b+1)(a+b+2)(a+2b+3)/6 (QSpace's Weyl product).
    // SO6 15x15: 1+2*15+20+45+45bar+84 (SU4 adjoint product).
    for (group, label, dimension, outcomes, channels) in [
        (
            GroupId::sp(4).unwrap(),
            vec![1, 1],
            16,
            vec![
                (vec![0, 0], 1, 1),
                (vec![0, 1], 1, 5),
                (vec![0, 2], 1, 14),
                (vec![0, 3], 1, 30),
                (vec![2, 0], 2, 10),
                (vec![2, 1], 2, 35),
                (vec![2, 2], 1, 81),
                (vec![4, 0], 1, 35),
            ],
            vec![vec![2, 0], vec![2, 1]],
        ),
        (
            GroupId::so(6).unwrap(),
            vec![0, 1, 1],
            15,
            vec![
                (vec![0, 0, 0], 1, 1),
                (vec![0, 1, 1], 2, 15),
                (vec![2, 0, 0], 1, 20),
                (vec![1, 0, 2], 1, 45),
                (vec![1, 2, 0], 1, 45),
                (vec![0, 2, 2], 1, 84),
            ],
            vec![vec![0, 1, 1]],
        ),
    ] {
        let r = rule(group);
        let a = r.encode_dynkin(&label).unwrap();
        exact_dimension(&r, &label, dimension);
        assert_eq!(
            fusion(&r, &label, &label),
            outcomes.iter().map(|(l, n, _)| (l.clone(), *n)).collect()
        );
        for (l, _, dim) in &outcomes {
            exact_dimension(&r, l, *dim);
        }
        assert_eq!(
            outcomes
                .iter()
                .map(|(_, n, dim)| *n as u64 * dim)
                .sum::<u64>(),
            dimension * dimension
        );
        for channel in channels {
            let c = r.encode_dynkin(&channel).unwrap();
            assert_eq!(r.try_nsymbol(a, a, c).unwrap(), 2);
            // A unit leg makes the F entries independently fixed delta_mu,lambda,
            // while retaining both outer-multiplicity axes (no scalar shortcut).
            let f = r.try_f_symbol_generic(a, a, r.vacuum(), c, c, a).unwrap();
            assert_eq!(f.shape(), (2, 1, 1, 2));
            for i in 0..2 {
                for j in 0..2 {
                    numerics::assert_close(
                        "unit F multiplicity identity",
                        *f.get(i, 0, 0, j),
                        f64::from(i == j),
                        1,
                    );
                }
            }
            for (name, matrix) in [
                ("R", r.try_r_symbol_generic(a, a, c).unwrap()),
                ("A", r.try_a_symbol_generic(a, a, c).unwrap()),
                ("B", r.try_b_symbol_generic(a, a, c).unwrap()),
            ] {
                assert_eq!(matrix.shape(), (2, 2), "{name}");
                orthogonal(name, matrix.data(), 2);
            }
            // All four axes are outer multiplicities in a genuine nonunit
            // recoupling. This subblock need not itself be orthogonal: other
            // intermediate channels complete the full associator.
            let full_axes = r.try_f_symbol_generic(a, a, a, a, c, c).unwrap();
            assert_eq!(full_axes.shape(), (2, 2, 2, 2));
            assert!(full_axes.data().iter().all(|x| x.is_finite()));
            assert!(full_axes.data().iter().any(|x| x.abs() > 0.0));
            let exchange = r.try_r_symbol_generic(a, a, c).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let square = (0..2)
                        .map(|k| exchange.get(i, k) * exchange.get(k, j))
                        .sum();
                    numerics::assert_close(
                        "multiplicity inverse braid",
                        square,
                        f64::from(i == j),
                        2,
                    );
                }
            }
        }
    }
}
