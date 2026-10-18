use super::*;
use crate::probe::{self, Event};

#[test]
fn fermionic_storage_contract_rejects_nonuniform_twist_before_gemm() {
    let rule = AdversarialNonuniformTwistRule;
    let leg = |is_dual| {
        SectorLeg::new(
            [
                (SectorId::new(0), 1),
                (SectorId::new(1), 1),
                (SectorId::new(2), 1),
            ],
            is_dual,
        )
    };
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false)]),
        FusionProductSpace::new([leg(true), leg(true)]),
    );
    let lhs = DynamicFusionMapSpace::from_final_homspace(&rule, lhs_hom).unwrap();
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(true), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    );
    let rhs = DynamicFusionMapSpace::from_final_homspace(&rule, rhs_hom).unwrap();
    let dst =
        crate::DynamicFusionMapSpace::contracted(&rule, &lhs, &rhs, &[1, 2], &[0, 1]).unwrap();
    let provider = Arc::new(rule);
    let lhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(lhs, Arc::clone(&provider))
            .unwrap();
    let rhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(rhs, Arc::clone(&provider))
            .unwrap();
    let dst_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(dst, Arc::clone(&provider))
            .unwrap();

    #[derive(Default)]
    struct NoCallGemm {
        calls: usize,
    }

    impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
        for NoCallGemm
    {
        fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
            lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
        }

        fn matmul_range_into(
            &mut self,
            _dst: &mut Vec<f64>,
            _dst_offset: usize,
            _lhs: &Vec<f64>,
            _lhs_offset: usize,
            _rhs: &Vec<f64>,
            _rhs_offset: usize,
            _rows: usize,
            _contracted: usize,
            _cols: usize,
        ) -> Result<(), OperationError> {
            self.calls += 1;
            Ok(())
        }

        fn matmul_range_with_ops_scaled_into(
            &mut self,
            _dst: &mut Vec<f64>,
            _dst_offset: usize,
            _lhs: &Vec<f64>,
            _lhs_offset: usize,
            _rhs: &Vec<f64>,
            _rhs_offset: usize,
            _rows: usize,
            _contracted: usize,
            _cols: usize,
            _lhs_op: MatrixOp,
            _rhs_op: MatrixOp,
            _alpha: f64,
        ) -> Result<(), OperationError> {
            self.calls += 1;
            Ok(())
        }
    }

    let lhs_values = vec![1.0; lhs_bound.space().required_len().unwrap()];
    let rhs_values = vec![1.0; rhs_bound.space().required_len().unwrap()];
    let mut output = vec![9.0; dst_bound.space().required_len().unwrap()];
    let before = output.clone();
    let mut gemm = NoCallGemm::default();
    let error = crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
        &mut gemm,
        &dst_bound,
        &mut output,
        crate::FusionOperand::direct(lhs_bound.space()),
        &lhs_values,
        crate::FusionOperand::direct(rhs_bound.space()),
        &rhs_values,
        TensorContractSpec::new(&[1, 2], &[0, 1], crate::OutputAxisOrder::identity()),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "storage-direct contraction supports only canonical fully-direct oriented \
                      operands"
        }
    ));
    assert_eq!(gemm.calls, 0);
    assert_eq!(output, before);
}

#[test]
fn oriented_storage_contract_and_compose_use_parent_rectangular_views() {
    let rule = Z2FusionRule;
    let provider = Arc::new(rule);
    let (m, k, n) = (2, 3, 4);
    let lhs_blocks = [
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ];
    let rhs_blocks = [
        vec![1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 4.0, 1.0, 0.0, 2.0, 5.0, 1.0],
        vec![2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 0.0, 4.0, 1.0, 3.0, 0.0, 2.0],
    ];
    let transpose = |matrix: &[f64], rows: usize, cols: usize| {
        (0..rows)
            .flat_map(|row| (0..cols).map(move |col| matrix[row + rows * col]))
            .collect::<Vec<_>>()
    };
    let product = |lhs: &[f64], rhs: &[f64]| {
        (0..n)
            .flat_map(|col| {
                (0..m).map(move |row| {
                    (0..k)
                        .map(|inner| lhs[row + m * inner] * rhs[inner + k * col])
                        .sum::<f64>()
                })
            })
            .collect::<Vec<_>>()
    };
    let expected = lhs_blocks
        .iter()
        .zip(&rhs_blocks)
        .flat_map(|(lhs, rhs)| product(lhs, rhs))
        .collect::<Vec<_>>();
    let matrix_space = |rows, cols| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new(
                    [(SectorId::new(0), rows), (SectorId::new(1), rows)],
                    false,
                )]),
                FusionProductSpace::new([SectorLeg::new(
                    [(SectorId::new(0), cols), (SectorId::new(1), cols)],
                    false,
                )]),
            ),
        )
        .unwrap()
    };

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let lhs_parent = if lhs_adjoint {
            matrix_space(k, m)
        } else {
            matrix_space(m, k)
        };
        let lhs_logical_typed = if lhs_adjoint {
            lhs_parent
                .validate_rule(&rule)
                .and_then(|()| lhs_parent.adjoint_view())
                .unwrap()
        } else {
            lhs_parent.clone()
        };
        let rhs_parent = if rhs_adjoint {
            matrix_space(n, k)
        } else {
            matrix_space(k, n)
        };
        let rhs_logical_typed = if rhs_adjoint {
            rhs_parent
                .validate_rule(&rule)
                .and_then(|()| rhs_parent.adjoint_view())
                .unwrap()
        } else {
            rhs_parent.clone()
        };
        let lhs_logical = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            lhs_logical_typed.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
        let rhs_logical = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            rhs_logical_typed.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
        let dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free(
            &lhs_logical,
            &rhs_logical,
            &[1],
            &[0],
        )
        .unwrap();
        assert!(Arc::ptr_eq(dst.provider_arc(), &provider));
        let lhs_operand = if lhs_adjoint {
            crate::FusionOperand::adjoint(&lhs_parent)
        } else {
            crate::FusionOperand::direct(&lhs_parent)
        };
        let rhs_operand = if rhs_adjoint {
            crate::FusionOperand::adjoint(&rhs_parent)
        } else {
            crate::FusionOperand::direct(&rhs_parent)
        };
        let lhs_values = lhs_blocks
            .iter()
            .flat_map(|block| {
                if lhs_adjoint {
                    transpose(block, m, k)
                } else {
                    block.clone()
                }
            })
            .collect::<Vec<_>>();
        let rhs_values = rhs_blocks
            .iter()
            .flat_map(|block| {
                if rhs_adjoint {
                    transpose(block, k, n)
                } else {
                    block.clone()
                }
            })
            .collect::<Vec<_>>();
        let fusion_operand_projection_prepares_rec = probe::record();

        let mut contracted = vec![0.0; dst.space().required_len().unwrap()];
        let mut contract_gemm = CpuOrientedGemm::default();
        crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
            &mut contract_gemm,
            &dst,
            &mut contracted,
            lhs_operand,
            &lhs_values,
            rhs_operand,
            &rhs_values,
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                crate::OutputAxisOrder::identity(),
                lhs_adjoint,
                rhs_adjoint,
            ),
        )
        .unwrap();
        assert_eq!(contracted, expected);
        assert_eq!(contract_gemm.calls.len(), 2);
        assert!(contract_gemm.calls[1].0 > 0 && contract_gemm.calls[1].1 > 0);
        assert_eq!(
            contract_gemm.calls[0].2,
            if lhs_adjoint {
                MatrixOp::Adjoint
            } else {
                MatrixOp::Identity
            }
        );
        assert_eq!(
            contract_gemm.calls[0].3,
            if rhs_adjoint {
                MatrixOp::Adjoint
            } else {
                MatrixOp::Identity
            }
        );

        let mut composed = vec![0.0; dst.space().required_len().unwrap()];
        super::compose_direct_on_storage(
            &mut CpuOrientedGemm::default(),
            &dst,
            &mut composed,
            lhs_operand,
            &lhs_values,
            rhs_operand,
            &rhs_values,
        )
        .unwrap();
        assert_eq!(composed, expected);
        assert_eq!(
            fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
            0
        );

        if lhs_adjoint && !rhs_adjoint {
            let outer_dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free(
                &lhs_logical,
                &rhs_logical,
                &[],
                &[],
            )
            .unwrap();
            let sentinel = vec![42.0; outer_dst.space().required_len().unwrap()];
            let mut rejected = sentinel.clone();
            let mut rejected_gemm = CpuOrientedGemm::default();
            let error = super::compose_direct_on_storage(
                &mut rejected_gemm,
                &outer_dst,
                &mut rejected,
                lhs_operand,
                &lhs_values,
                rhs_operand,
                &rhs_values,
            )
            .unwrap_err();
            // Composition derives its axes, so an outer-product destination
            // is a rank mismatch, rejected before any write or projection.
            assert!(
                matches!(
                    error,
                    OperationError::StructureRankMismatch {
                        expected: 2,
                        actual: 4
                    }
                ),
                "{error:?}"
            );
            assert!(rejected_gemm.calls.is_empty());
            assert_eq!(rejected, sentinel);
            assert_eq!(
                fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
                0
            );
        }
    }
}

#[test]
fn oriented_fermionic_storage_keeps_contract_and_compose_signs_distinct() {
    let rule = FermionParityFusionRule;
    let provider = Arc::new(rule);
    let odd = SectorId::new(1);
    let space = |codomain_dual, domain_dual| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(odd, 1)], codomain_dual)]),
                FusionProductSpace::new([SectorLeg::new([(odd, 1)], domain_dual)]),
            ),
        )
        .unwrap()
    };
    let lhs_logical_typed = space(false, true);
    let rhs_logical_typed = space(true, false);
    let lhs_values = vec![2.0];
    let rhs_values = vec![3.0];

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let lhs_parent = if lhs_adjoint {
            lhs_logical_typed.adjoint_view().unwrap()
        } else {
            lhs_logical_typed.clone()
        };
        let rhs_parent = if rhs_adjoint {
            rhs_logical_typed.adjoint_view().unwrap()
        } else {
            rhs_logical_typed.clone()
        };
        let lhs_logical = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            lhs_logical_typed.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
        let rhs_logical = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            rhs_logical_typed.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
        let dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free(
            &lhs_logical,
            &rhs_logical,
            &[1],
            &[0],
        )
        .unwrap();
        let lhs = if lhs_adjoint {
            crate::FusionOperand::adjoint(&lhs_parent)
        } else {
            crate::FusionOperand::direct(&lhs_parent)
        };
        let rhs = if rhs_adjoint {
            crate::FusionOperand::adjoint(&rhs_parent)
        } else {
            crate::FusionOperand::direct(&rhs_parent)
        };
        let fusion_operand_projection_prepares_rec = probe::record();

        let mut contracted = vec![0.0];
        let mut contract_gemm = CpuOrientedGemm::default();
        crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
            &mut contract_gemm,
            &dst,
            &mut contracted,
            lhs,
            &lhs_values,
            rhs,
            &rhs_values,
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                crate::OutputAxisOrder::identity(),
                lhs_adjoint,
                rhs_adjoint,
            ),
        )
        .unwrap();
        assert_eq!(contracted, [-6.0]);
        assert_eq!(contract_gemm.calls[0].4, -1.0);

        let mut composed = vec![0.0];
        super::compose_direct_on_storage(
            &mut CpuOrientedGemm::default(),
            &dst,
            &mut composed,
            lhs,
            &lhs_values,
            rhs,
            &rhs_values,
        )
        .unwrap();
        assert_eq!(composed, [6.0]);
        assert_eq!(
            fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
            0
        );
    }
}
