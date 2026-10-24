use super::*;

fn scalar_structure(keys: &[FusionTreePairKey]) -> BlockStructure {
    BlockStructure::from_blocks(
        keys.iter()
            .cloned()
            .enumerate()
            .map(|(offset, key)| {
                let rank = key.codomain_uncoupled().len() + key.domain_uncoupled().len();
                BlockSpec::column_major_with_key(key.into(), vec![1; rank], offset).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn absorb_merge_matches_only_complete_asymmetric_interleaved_keys() {
    let rule = SU2FusionRule;
    let pair = |coupled, innerline| {
        let tree = FusionTreeKey::try_from_sector_ids_for_rule(
            &rule,
            [1, 1, 1],
            coupled,
            [false; 3],
            [innerline],
            [1, 1],
        )
        .unwrap();
        FusionTreePairKey::pair(tree.clone(), tree)
    };
    let inner_zero = pair(1, 0);
    let inner_two = pair(1, 2);
    let coupled_three = pair(3, 2);
    let destination_keys = vec![inner_zero.clone(), inner_two.clone(), coupled_three.clone()];

    for source_keys in [vec![inner_two], vec![inner_zero, coupled_three]] {
        let mut destination_keys = destination_keys.clone();
        destination_keys.sort();
        let mut source_keys = source_keys;
        source_keys.sort();
        let destination_structure = scalar_structure(&destination_keys);
        let source_structure = scalar_structure(&source_keys);
        let mut destination = (0..destination_keys.len())
            .map(|index| 100.0 + index as f64)
            .collect::<Vec<_>>();
        let before = destination.clone();
        let source = (0..source_keys.len())
            .map(|index| 10.0 + index as f64)
            .collect::<Vec<_>>();
        let source_values = source_keys
            .iter()
            .cloned()
            .zip(source.iter().copied())
            .collect::<HashMap<_, _>>();

        absorb_mapped(
            &destination_structure,
            &mut destination,
            &source_structure,
            &source,
            Ok,
        )
        .unwrap();

        for index in 0..destination_structure.block_count() {
            let block = destination_structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!()
            };
            assert_eq!(
                destination[block.offset()],
                source_values
                    .get(key)
                    .copied()
                    .unwrap_or(before[block.offset()])
            );
        }
    }
}

#[test]
fn mixed_compact_add_does_not_materialize_the_lazy_operand() {
    let densified = probe::record();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, indices| {
        (indices[0] + 2 * indices[1]) as f64
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();
    let eager = eager_adjoint_oracle(&dense);
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![2.0, 3.0],
        }],
    )
    .unwrap();
    let actual = diagonal.axpby(0.5, &lazy, -2.0).unwrap();
    let expected = diagonal.axpby(0.5, &eager, -2.0).unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    let reverse = lazy.axpby(-2.0, &diagonal, 0.5).unwrap();
    let expected_reverse = eager.axpby(-2.0, &diagonal, 0.5).unwrap();
    assert_eq!(
        reverse.dense_data().unwrap(),
        expected_reverse.dense_data().unwrap()
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    for (lhs, rhs, eager_lhs, eager_rhs) in [
        (&diagonal, &lazy, &diagonal, &eager),
        (&lazy, &diagonal, &eager, &diagonal),
    ] {
        let actual = lhs.compose(rhs).unwrap();
        let expected = eager_lhs.compose(eager_rhs).unwrap();
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        let actual = lhs
            .contract(
                rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        let expected = eager_lhs
            .contract(
                eager_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    numerics::assert_close(
        "diagonal.inner(&lazy)",
        diagonal.inner(&lazy).unwrap(),
        diagonal.inner(&eager).unwrap(),
        GATE_TERMS,
    );
    numerics::assert_close(
        "lazy.inner(&diagonal)",
        lazy.inner(&diagonal).unwrap(),
        eager.inner(&diagonal).unwrap(),
        GATE_TERMS,
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn mixed_compact_dense_inner_does_not_materialize_the_diagonal() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => Complex64::new(1.0, 2.0),
            (1, 1) => Complex64::new(4.0, -1.0),
            _ => Complex64::new(6.0, 7.0),
        }
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 0.5)],
        }],
    )
    .unwrap();

    let densified = probe::record();
    numerics::assert_close(
        "diagonal.inner(&dense)",
        diagonal.inner(&dense).unwrap(),
        Complex64::new(3.5, 0.0),
        GATE_TERMS,
    );
    numerics::assert_close(
        "dense.inner(&diagonal)",
        dense.inner(&diagonal).unwrap(),
        Complex64::new(3.5, 0.0),
        GATE_TERMS,
    );
    numerics::assert_close(
        "diagonal.inner(&lazy)",
        diagonal.inner(&lazy).unwrap(),
        Complex64::new(-7.5, -10.0),
        GATE_TERMS,
    );
    numerics::assert_close(
        "lazy.inner(&diagonal)",
        lazy.inner(&diagonal).unwrap(),
        Complex64::new(-7.5, 10.0),
        GATE_TERMS,
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn mixed_compact_dense_inner_weights_sectors_and_skips_structural_zeros() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let bond = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![3.0],
            },
        ],
    )
    .unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
        if indices[0] != indices[1] {
            if indices[0] == 0 {
                f64::NAN
            } else {
                f64::INFINITY
            }
        } else if *trees.coupled() == spin0 {
            [4.0, 5.0][indices[0]]
        } else {
            6.0
        }
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();

    let densified = probe::record();
    for dense in [&dense, &lazy] {
        assert_eq!(diagonal.inner(dense).unwrap(), 50.0);
        assert_eq!(dense.inner(&diagonal).unwrap(), 50.0);
    }
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let nonfinite_diagonal =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
            if indices == [0, 0] && *trees.coupled() == spin0 {
                f64::NAN
            } else {
                1.0
            }
        })
        .unwrap();
    assert!(diagonal.inner(&nonfinite_diagonal).unwrap().is_nan());
    assert!(nonfinite_diagonal.inner(&diagonal).unwrap().is_nan());

    let infinite_diagonal =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
            if indices == [0, 0] && *trees.coupled() == spin0 {
                f64::INFINITY
            } else {
                1.0
            }
        })
        .unwrap();
    assert!(diagonal.inner(&infinite_diagonal).unwrap().is_infinite());
    assert!(infinite_diagonal.inner(&diagonal).unwrap().is_infinite());
}

#[test]
fn mixed_compact_dense_inner_accepts_an_empty_bond() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    let diagonal: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &bond, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
    let dense: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, _| unreachable!()).unwrap();

    let densified = probe::record();
    assert_eq!(diagonal.inner(&dense).unwrap(), 0.0);
    assert_eq!(dense.inner(&diagonal).unwrap(), 0.0);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn mixed_compact_lazy_inner_maps_non_self_dual_parent_blocks() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: U1Irrep::new(2),
                values: vec![6.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![5.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![3.0, 4.0],
            },
        ],
    )
    .unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
        if indices[0] != indices[1] {
            -1000.0
        } else {
            match trees.coupled().charge() {
                -1 => 30.0,
                0 => [10.0, 20.0][indices[0]],
                2 => 40.0,
                charge => panic!("unexpected charge {charge}"),
            }
        }
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();

    let densified = probe::record();
    assert_eq!(diagonal.inner(&lazy).unwrap(), 500.0);
    assert_eq!(lazy.inner(&diagonal).unwrap(), 500.0);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

fn transform_seam_calls<T>(operation: impl FnOnce() -> T) -> usize {
    let calls = crate::typed::probe::record();
    let _output = operation();
    calls.get(crate::typed::probe::Event::TreeTransformSeamCall)
}

#[test]
fn exact_identity_transforms_share_unique_and_simple_bodies() {
    // What (#689): exact identity permute/braid/transpose/repartition
    // never reach the transform seam and return the same body allocation.
    // Body identity also pins zero payload copies more directly than an
    // allocator byte count can.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1_leg, &u1_leg], [&u1_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let u1_c64 = u1_f64.convert::<Complex64>();

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2_f64: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let su2_c64 = su2_f64.convert::<Complex64>();

    macro_rules! assert_identity_ops {
        ($tensor:expr) => {{
            let tensor = $tensor;
            let calls = transform_seam_calls(|| {
                for output in [
                    tensor.permute(&[0, 1], &[2]).unwrap(),
                    tensor.braid(&[0, 1], &[2], &[5, 3, 1]).unwrap(),
                    tensor.transpose(&[0, 1], &[2]).unwrap(),
                    tensor.repartition(2).unwrap(),
                ] {
                    assert!(Arc::ptr_eq(owned(tensor), owned(&output)));
                    assert!(Arc::ptr_eq(&owned(tensor).data, &owned(&output).data));
                    assert_eq!(
                        tensor.dense_data().unwrap().as_ptr(),
                        output.dense_data().unwrap().as_ptr()
                    );
                }
            });
            assert_eq!(calls, 0);
        }};
    }

    assert_identity_ops!(&u1_f64);
    assert_identity_ops!(&u1_c64);
    assert_identity_ops!(&su2_f64);
    assert_identity_ops!(&su2_c64);

    // Validation still precedes the braid shortcut.
    let calls = transform_seam_calls(|| {
        assert!(u1_f64.braid(&[0, 1], &[2], &[0, 0]).is_err());
    });
    assert_eq!(calls, 0);
    assert!(
        matches!(u1_f64.permute(&[0, 1], &[2, 3]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidPermutation { rank: 3, .. }))
    );
    assert!(
        matches!(u1_f64.transpose(&[0, 1], &[2, 3]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidPermutation { rank: 3, .. }))
    );
    assert!(u1_f64.repartition(4).is_err());

    // Negative control: the counter observes a real transform.
    let calls = transform_seam_calls(|| {
        let moved = u1_f64.permute(&[1, 0], &[2]).unwrap();
        assert!(!Arc::ptr_eq(owned(&u1_f64), owned(&moved)));
    });
    assert_eq!(calls, 1);
}

#[test]
fn high_rank_identity_has_no_inline_capacity_boundary() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)]).unwrap();
    let tensor: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg; 10], [&leg; 9], |_, _| 1.0).unwrap();
    let codomain_axes: Vec<_> = (0..10).collect();
    let domain_axes: Vec<_> = (10..19).collect();
    let levels = vec![0; 19];

    let calls = transform_seam_calls(|| {
        for output in [
            tensor.permute(&codomain_axes, &domain_axes).unwrap(),
            tensor.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
            tensor.transpose(&codomain_axes, &domain_axes).unwrap(),
        ] {
            assert!(Arc::ptr_eq(owned(&tensor), owned(&output)));
        }
    });
    assert_eq!(calls, 0);
}

#[test]
fn compact_identity_transforms_do_not_materialize() {
    let densified = probe::record();
    let factor = fixture().svd_compact(&[0], &[1]).unwrap().s;
    assert!(matches!(&*owned(&factor).data, TypedData::Diagonal(_)));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let calls = transform_seam_calls(|| {
        for output in [
            factor.permute(&[0], &[1]).unwrap(),
            factor.braid(&[0], &[1], &[2, 1]).unwrap(),
            factor.transpose(&[0], &[1]).unwrap(),
            factor.repartition(1).unwrap(),
        ] {
            assert!(Arc::ptr_eq(owned(&factor), owned(&output)));
            assert!(matches!(&*owned(&output).data, TypedData::Diagonal(_)));
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        }
    });
    assert_eq!(calls, 0);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn scalar_transpose_shares_the_body() {
    let scalar = fixture().trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(scalar.rank(), 0);
    let calls = transform_seam_calls(|| {
        let transposed = scalar.transpose(&[], &[]).unwrap();
        assert!(Arc::ptr_eq(owned(&scalar), owned(&transposed)));
    });
    assert_eq!(calls, 0);
}

#[test]
fn twist_on_a_compact_spectrum_stays_compact() {
    // What (#580): the compact twist arm scales
    // spectrum-per-sector and keeps `TypedData::Diagonal` — the space is
    // unchanged, so O(Σ_c k_c) storage survives — and its own identity
    // answer (θ ≡ 1 across the spectrum's sectors) is a body-sharing
    // clone.
    let s = fz2_fixture().svd_compact(&[0, 1], &[2]).unwrap().s;
    let twisted = s.twist(&[0], Direction::Forward).unwrap();
    assert!(matches!(&*owned(&twisted).data, TypedData::Diagonal(_)));
    assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&twisted).data));
    let inverse = s.twist(&[0], Direction::Inverse).unwrap();
    assert!(matches!(&*owned(&inverse).data, TypedData::Diagonal(_)));
    assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&inverse).data));

    let bosonic_s = fixture().svd_compact(&[0], &[1]).unwrap().s;
    let untouched = bosonic_s.twist(&[0], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(owned(&bosonic_s), owned(&untouched)));
    let untouched_inverse = bosonic_s.twist(&[0], Direction::Inverse).unwrap();
    assert!(Arc::ptr_eq(owned(&bosonic_s), owned(&untouched_inverse)));
}

/// #1337: filling the compact payload from a borrowed slice instead of
/// consuming the spectrum must move exactly the same bits, in the same
/// order, at every payload dtype.
///
/// The expected values are written out rather than derived from
/// `FactorScalar::from_real`, so the assertion is independent of the
/// conversion under test. The fixture carries a zero, a negative value, an
/// `f64` subnormal (which `f32` must flush to `+0.0`, not to a NaN or a
/// denormal of its own) and an `f32` subnormal (which must survive), and
/// declares its sectors out of engine order so a factor that kept the
/// caller's order or reordered values inside a sector fails.
#[test]
fn diagonal_spectrum_factor_converts_every_value_bitwise() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let one = U1Irrep::new(1);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3), (one, 2)]).unwrap();
    let authority: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 0.0).unwrap();
    let id = |label| TypedSectorAdmission::try_encode_label(provider.as_ref(), &label).unwrap();

    let tiny = f64::from(f32::MIN_POSITIVE) / 2.0;
    let spectrum = || {
        vec![
            symtn_matrixalgebra::SectorSpectrum {
                sector: id(one),
                values: vec![0.0_f64, -1.5],
            },
            symtn_matrixalgebra::SectorSpectrum {
                sector: id(zero),
                values: vec![1.0_f64, f64::MIN_POSITIVE / 2.0, tiny],
            },
        ]
    };

    let mut wide = spectrum();
    let wide_factor: TensorMap<_, f64> = diagonal_factor_on(
        &runtime,
        authority.logical_space(),
        &mut wide,
        <f64 as FactorScalar>::from_real,
    )
    .unwrap();
    let seen = wide_factor.diagview().unwrap();
    let bits: Vec<(U1Irrep, Vec<u64>)> = seen
        .iter()
        .map(|entry| {
            (
                entry.sector,
                entry.values.iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect();
    assert_eq!(
        bits,
        vec![
            (
                zero,
                vec![
                    1.0_f64.to_bits(),
                    (f64::MIN_POSITIVE / 2.0).to_bits(),
                    tiny.to_bits()
                ]
            ),
            (one, vec![0.0_f64.to_bits(), (-1.5_f64).to_bits()]),
        ]
    );

    let mut narrow = spectrum();
    let narrow_factor: TensorMap<_, f32> = diagonal_factor_on(
        &runtime,
        authority.logical_space(),
        &mut narrow,
        <f32 as FactorScalar>::from_real,
    )
    .unwrap();
    let seen = narrow_factor.diagview().unwrap();
    let bits: Vec<(U1Irrep, Vec<u32>)> = seen
        .iter()
        .map(|entry| {
            (
                entry.sector,
                entry.values.iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect();
    assert_eq!(
        bits,
        vec![
            (
                zero,
                vec![
                    1.0_f32.to_bits(),
                    0.0_f32.to_bits(),
                    (f32::MIN_POSITIVE / 2.0).to_bits()
                ]
            ),
            (one, vec![0.0_f32.to_bits(), (-1.5_f32).to_bits()]),
        ]
    );
}

#[test]
fn compact_arms_never_densify_their_spectrum_operand() {
    // What (#1548): the operations with a compact-diagonal arm read the stored
    // spectrum and never enter the operation-local densification. The probe
    // counts entries directly, which the integration byte ceilings cannot
    // tell apart from a scaled copy.
    let tensor = fixture();
    let Svd { u, s: d, vh } = tensor.svd_compact(&[0], &[1]).unwrap();
    let complex_d = tensor
        .convert::<Complex64>()
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    let densified = probe::record();

    let _ = d.scale(0.5).unwrap();
    let _ = d.adjoint().unwrap();
    let _ = d.axpby(0.75, &d, -0.5).unwrap();
    let _ = d.axpby(0.75, &tensor, -0.5).unwrap();
    let _ = tensor.axpby(0.75, &d, -0.5).unwrap();
    let _ = d.compose(&d).unwrap();
    let _ = u.compose(&d).unwrap();
    let _ = d.compose(&vh).unwrap();
    for p in [2.0, f64::INFINITY, 3.0] {
        let _ = d.norm(p).unwrap();
    }
    let _ = d.tr().unwrap();
    let _ = d.inner(&d).unwrap();
    let _ = d.exp(&[0], &[1]).unwrap();
    let _ = d.inv(&[0], &[1]).unwrap();
    let _ = d.pinv(&[0], &[1], 1e-12).unwrap();
    let _ = d.map_diagonal(|x| x.abs().sqrt()).unwrap();
    let _ = tensor
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let _ = tensor
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let _ = d
        .contract(
            &tensor,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let _ = d
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let _ = d.permute(&[1], &[0]).unwrap();
    let _ = d.transpose(&[1], &[0]).unwrap();
    let _ = d.repartition(1).unwrap();
    let _ = d.zeros_like().unwrap();
    let _ = d.convert::<Complex64>();
    let _ = complex_d.re();
    let _ = complex_d.im();
    let _ = d.trace_pairs(&[(0, 1)]).unwrap();
    let _ = d.diagview().unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    // A one-term rank-(1,1) braid reads the compact source directly.
    let _ = d.braid(&[1], &[0], &[0, 1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_arms_never_densify_their_spectrum_operand() {
    // What (#1866): checked Generic takes the same compose, contract,
    // rank-(1,1) transform and full trace arms, so a compact operand is read
    // as a spectrum there too.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(
        symtn_core::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
    );
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 1], 3)]).unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            1.0 + index[0] as f64 - 0.25 * index[1] as f64
        })
        .unwrap();
    let Svd { u, s: d, vh } = tensor.svd_compact(&[0], &[1]).unwrap();
    let densified = probe::record();

    let _ = d.compose(&d).unwrap();
    let _ = u.compose(&d).unwrap();
    let _ = d.compose(&vh).unwrap();
    for (lhs, rhs, codomain, domain) in [
        (&u, &d, &[0][..], &[1][..]),
        (&u, &d, &[1][..], &[0][..]),
        (&d, &vh, &[0][..], &[1][..]),
        (&d, &d, &[0][..], &[1][..]),
    ] {
        let spec = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain,
            domain,
        };
        let _ = lhs.contract(rhs, &spec).unwrap();
    }
    // The rank-(1,1) swap stays compact and the braid reads the spectrum.
    let swapped = d.permute(&[1], &[0]).unwrap();
    assert!(swapped.spectrum().is_some());
    let _ = d.transpose(&[1], &[0]).unwrap();
    let _ = d.braid(&[1], &[0], &[0, 1]).unwrap();
    // The full trace reads the spectrum from the compiled trace terms.
    let _ = d.trace_pairs(&[(0, 1)]).unwrap();
    let _ = d.trace_pairs(&[(1, 0)]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn compact_braid_off_diagonal_zeros_are_numerically_zero() {
    // This finite witness exercises the one-term admission; structural zeros
    // have no prescribed sign under a fermionic braid.
    use symtn_core::PreparedTreePairOperation;
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! probe {
        ($rule:expr, $sectors:expr, $negative_zeros:expr) => {{
            let leg = GradedSpace::try_new(Arc::new($rule), $sectors).unwrap();
            let tensor: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 1617).unwrap();
            let diagonal = tensor.svd_compact(&[0], &[1]).unwrap().s;
            let source = diagonal.logical_space().space().structure();
            let prepared = PreparedTreePairOperation::prepare_braid(
                diagonal.provider(),
                1,
                1,
                &[1],
                &[0],
                &[0],
                &[1],
            )
            .unwrap();
            let mut counts = Vec::new();
            let destination = diagonal
                .logical_space()
                .transformed_multiplicity_free(&TreeTransformOperation::braid([1], [0], [0], [1]))
                .unwrap();
            let destination_structure = destination.space().structure();
            let mut covered = vec![false; destination_structure.block_count()];
            for i in 0..source.block_count() {
                let source_block = source.block(i).unwrap();
                let pair = source_block.key().as_fusion_tree_pair().unwrap();
                let rows = prepared
                    .execute_multiplicity_free(diagonal.provider(), pair)
                    .unwrap();
                counts.push(rows.len());
                assert_eq!(rows.len(), 1);
                let (destination_pair, coefficient) = &rows[0];
                assert!(coefficient.is_finite() && *coefficient != 0.0);
                let destination_index = (0..destination_structure.block_count())
                    .find(|&j| {
                        destination_structure.block(j).unwrap().key()
                            == &symtn_core::BlockKey::FusionTree(destination_pair.clone())
                    })
                    .expect("braid destination exists");
                assert!(!std::mem::replace(&mut covered[destination_index], true));
                let destination_block = destination_structure.block(destination_index).unwrap();
                assert_eq!(source_block.shape(), destination_block.shape());
                assert_eq!(destination_block.shape()[0], destination_block.shape()[1]);
            }
            assert!(covered.iter().all(|&seen| seen));
            let densified = probe::record();
            let result = diagonal.braid(&[1], &[0], &[0, 1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            let dense_oracle = diagonal
                .materialize()
                .unwrap()
                .braid(&[1], &[0], &[0, 1])
                .unwrap();
            for (&actual, &expected) in result
                .dense_data()
                .unwrap()
                .iter()
                .zip(dense_oracle.dense_data().unwrap())
            {
                assert!((actual - expected).abs() <= 32.0 * f64::EPSILON * expected.abs().max(1.0));
            }
            assert!(diagonal
                .diagview()
                .unwrap()
                .iter()
                .flat_map(|entry| &entry.values)
                .all(|&value| value != 0.0));
            let output = result.dense_data().unwrap();
            let structure = result.logical_space().space().structure();
            let mut negative_zeros = 0;
            for block_index in 0..structure.block_count() {
                let block = structure.block(block_index).unwrap();
                assert_eq!(block.shape().len(), 2);
                assert_eq!(block.shape()[0], block.shape()[1]);
                for column in 0..block.shape()[1] {
                    for row in 0..block.shape()[0] {
                        if row != column {
                            let offset = block.offset()
                                + row * block.strides()[0]
                                + column * block.strides()[1];
                            assert_eq!(output[offset], 0.0);
                            negative_zeros +=
                                usize::from(output[offset].to_bits() == (-0.0_f64).to_bits());
                        }
                    }
                }
            }
            assert!(counts.iter().all(|&count| count == 1));
            assert_eq!(negative_zeros, $negative_zeros);
        }};
    }
    probe!(
        U1FusionRule,
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        0
    );
    probe!(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2)
        ],
        0
    );
    probe!(
        FermionParityFusionRule,
        [(Z2Irrep::EVEN, 3), (Z2Irrep::ODD, 2)],
        0
    );
    probe!(
        ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 3),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
        0
    );
}

#[test]
fn compact_fermionic_braid_matches_hand_diagonal_and_complex_sign() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![2.0, -3.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![5.0, -7.0, 11.0],
            },
        ],
    )
    .unwrap();
    // A Complex64 payload on a real-coefficient rule: the complex sign comes
    // from the payload, not from a categorical coefficient.
    let complex = real
        .convert::<Complex64>()
        .map_diagonal(|value| Complex64::new(value.re, value.re / 4.0))
        .unwrap();
    let real_output = real.braid(&[1], &[0], &[0, 1]).unwrap();
    let complex_output = complex.braid(&[1], &[0], &[0, 1]).unwrap();
    let real_spectra = real_output.diagview().unwrap();
    let complex_spectra = complex_output.diagview().unwrap();
    let source_spectra = real.diagview().unwrap();
    assert_eq!(real_spectra.len(), source_spectra.len());
    assert_eq!(complex_spectra.len(), source_spectra.len());
    for (real_entry, complex_entry) in real_spectra.iter().zip(&complex_spectra) {
        assert_eq!(real_entry.sector, complex_entry.sector);
        let source = source_spectra
            .iter()
            .find(|entry| entry.sector == real_entry.sector)
            .unwrap();
        assert_eq!(real_entry.values.len(), source.values.len());
        assert_eq!(complex_entry.values.len(), source.values.len());
        let sign = if real_entry.sector == Z2Irrep::ODD {
            -1.0
        } else {
            1.0
        };
        for ((&real_value, &complex_value), &source_value) in real_entry
            .values
            .iter()
            .zip(&complex_entry.values)
            .zip(&source.values)
        {
            assert_eq!(real_value, sign * source_value);
            assert_eq!(
                complex_value,
                Complex64::new(sign * source_value, sign * source_value / 4.0)
            );
        }
    }
    let structure = real_output.logical_space().space().structure();
    let real_data = real_output.dense_data().unwrap();
    let complex_data = complex_output.dense_data().unwrap();
    for block_index in 0..structure.block_count() {
        let block = structure.block(block_index).unwrap();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                if row != column {
                    let offset =
                        block.offset() + row * block.strides()[0] + column * block.strides()[1];
                    assert_eq!(real_data[offset], 0.0);
                    assert_eq!(complex_data[offset], Complex64::new(0.0, 0.0));
                }
            }
        }
    }
}

#[test]
fn compact_cat_braid_absorb_avoid_dense_source_materializations() {
    let diagonal = fixture().svd_compact(&[0], &[1]).unwrap().s;
    let densified = probe::record();
    let _ = diagonal.cat(&diagonal, Side::Domain).unwrap();
    let cat = densified.get(Event::DiagonalMaterialization);
    let densified = probe::record();
    let _ = diagonal.braid(&[1], &[0], &[0, 1]).unwrap();
    let braid = densified.get(Event::DiagonalMaterialization);
    let densified = probe::record();
    let _ = diagonal.absorb(&diagonal).unwrap();
    let absorb = densified.get(Event::DiagonalMaterialization);
    assert_eq!([cat, braid, absorb], [0, 0, 0]);
}

#[test]
fn absorb_compact_source_zeros_shared_off_diagonal_and_preserves_outer_region() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(Z2FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 4)]).unwrap();
    let narrow = GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 2)]).unwrap();
    let receiver: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            1.0 + index[0] as f64 + 10.0 * index[1] as f64
        })
        .unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &narrow,
        [SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![2.0, -3.0],
        }],
    )
    .unwrap();
    let result = receiver.absorb(&source).unwrap();
    let data = result.dense_data().unwrap();
    let block = result.logical_space().space().structure().block(0).unwrap();
    assert_eq!(block.shape(), &[4, 4]);
    for column in 0..4 {
        for row in 0..4 {
            let expected = if row < 2 && column < 2 {
                if row == column {
                    [2.0, -3.0][row]
                } else {
                    0.0
                }
            } else {
                1.0 + row as f64 + 10.0 * column as f64
            };
            let offset = block.offset() + row * block.strides()[0] + column * block.strides()[1];
            assert_eq!(data[offset], expected, "row={row}, column={column}");
        }
    }
}

#[test]
fn compact_absorb_handles_strided_prefixes_and_interleaved_sectors() {
    let key = |sector| {
        let tree = FusionTreeKey::try_from_sector_ids_for_rule(
            &Z2FusionRule,
            [sector],
            sector,
            [false],
            [],
            [],
        )
        .unwrap();
        BlockKey::FusionTree(FusionTreePairKey::pair(tree.clone(), tree))
    };
    let source = BlockStructure::from_blocks(vec![
        BlockSpec::column_major_with_key(key(0), vec![1, 1], 0).unwrap(),
        BlockSpec::column_major_with_key(key(1), vec![2, 2], 1).unwrap(),
    ])
    .unwrap();
    let destination =
        BlockStructure::from_blocks(vec![
            BlockSpec::with_key(key(1), vec![2, 2], vec![2, 5], 0).unwrap()
        ])
        .unwrap();
    let spectrum = [
        symtn_matrixalgebra::SectorSpectrum {
            sector: SectorId::new(0),
            values: vec![5.0],
        },
        symtn_matrixalgebra::SectorSpectrum {
            sector: SectorId::new(1),
            values: vec![7.0, 9.0],
        },
    ];
    let mut values = vec![42.0; 8];
    absorb_compact_source(&destination, &mut values, &source, &spectrum).unwrap();
    assert_eq!(values, [7.0, 42.0, 0.0, 42.0, 42.0, 0.0, 42.0, 9.0]);

    // A destination-only lower sector is likewise untouched before the
    // shared strided block; this exercises both sides of the sector merge.
    let destination = BlockStructure::from_blocks(vec![
        BlockSpec::column_major_with_key(key(0), vec![1, 1], 0).unwrap(),
        BlockSpec::with_key(key(1), vec![2, 2], vec![2, 5], 1).unwrap(),
    ])
    .unwrap();
    let source = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        key(1),
        vec![2, 2],
        0,
    )
    .unwrap()])
    .unwrap();
    let mut values = vec![42.0; 9];
    absorb_compact_source(&destination, &mut values, &source, &spectrum[1..]).unwrap();
    assert_eq!(values, [42.0, 7.0, 42.0, 0.0, 42.0, 42.0, 0.0, 42.0, 9.0]);

    let duplicate = [spectrum[1].clone(), spectrum[1].clone()];
    assert!(
        matches!(absorb_compact_source(&destination, &mut values, &source, &duplicate), Err(Error::InvalidArgument(message)) if message.contains("duplicate spectrum sectors"))
    );
    assert!(
        matches!(absorb_compact_source(&destination, &mut values, &source, &[]), Err(Error::InvalidArgument(message)) if message.contains("spectrum sector is absent"))
    );
    let wrong_shape = [symtn_matrixalgebra::SectorSpectrum {
        sector: SectorId::new(1),
        values: vec![7.0],
    }];
    assert!(
        matches!(absorb_compact_source(&destination, &mut values, &source, &wrong_shape), Err(Error::InvalidArgument(message)) if message.contains("spectrum shape disagrees"))
    );
}

#[test]
fn compact_cat_and_absorb_preserve_stored_bits_and_zero_structural_cells() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, 2)]).unwrap();
    let nan = f64::from_bits(0x7ff8_0000_0000_1617);
    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![-0.0, nan],
        }],
    )
    .unwrap();
    let dense = compact.materialize().unwrap();
    let compact_cat = compact.cat(&compact, Side::Domain).unwrap();
    let dense_cat = dense.cat(&dense, Side::Domain).unwrap();
    assert_eq!(
        compact_cat
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        dense_cat
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(compact_cat
        .dense_data()
        .unwrap()
        .iter()
        .any(|x| x.to_bits() == (-0.0f64).to_bits()));
    assert!(compact_cat
        .dense_data()
        .unwrap()
        .iter()
        .any(|x| x.to_bits() == nan.to_bits()));
    assert!(compact_cat
        .dense_data()
        .unwrap()
        .iter()
        .any(|x| x.to_bits() == 0.0f64.to_bits()));

    let receiver: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, index| {
            10.0 + index[0] as f64 + index[1] as f64
        })
        .unwrap();
    let compact_absorb = receiver.absorb(&compact).unwrap();
    let dense_absorb = receiver.absorb(&dense).unwrap();
    assert_eq!(
        compact_absorb
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        dense_absorb
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    let block = compact_absorb
        .logical_space()
        .space()
        .structure()
        .block(0)
        .unwrap();
    let values = compact_absorb.dense_data().unwrap();
    let offset =
        |row, column| block.offset() + row * block.strides()[0] + column * block.strides()[1];
    assert_eq!(values[offset(0, 0)].to_bits(), (-0.0f64).to_bits());
    assert_eq!(values[offset(1, 1)].to_bits(), nan.to_bits());
    assert_eq!(values[offset(0, 1)].to_bits(), 0.0f64.to_bits());
    assert_eq!(values[offset(1, 0)].to_bits(), 0.0f64.to_bits());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_trace_reads_the_spectrum_after_admission() {
    // What (#1866): an inadmissible pair is rejected without touching the
    // spectrum, and the admitted full trace reads it from the compiled terms
    // without materializing (it densified after admission before).
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(
        symtn_core::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
    );
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(2.0, 1.0),
                    Complex64::new(-1.0, 2.0),
                    Complex64::new(4.0, -2.0),
                ],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    assert!(diagonal.trace_pairs(&[(0, 0)]).is_err());
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let result = diagonal.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    // SU(3) fundamental quantum dimension is 3; vacuum dimension is 1.
    assert!((result.dense_data().unwrap()[0] - Complex64::new(19.0, 4.0)).norm() < 1e-12);
    assert!(matches!(
        owned(&diagonal).data.as_ref(),
        TypedData::Diagonal(_)
    ));
}

#[cfg(feature = "cgc-gen")]
mod late_trace;
