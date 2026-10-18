//! CopyC member tests: the direct core into the temporary, then one completed
//! output transform, both member-expanded. The Host replay itself is the one
//! route executor (`symtn_tensors::HostContractMembersWorkspace`).

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
    use crate::typed::{ContractSpec, GradedSpace, TensorMap};
    use std::sync::Arc;
    use symtn_dense::{
        DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
        DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite, MatrixOp,
    };
    use symtn_operations::{
        tree_transform_members_overwrite_raw, Rank2Gemm, StridedHostKernelAdapter,
        TreeTransformWorkspace,
    };

    #[derive(Default)]
    struct CoreCount {
        calls: usize,
        jobs: usize,
    }

    impl Rank2Gemm<f64> for CoreCount {
        fn matmul_rank2(
            &mut self,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            _: usize,
            _: usize,
            _: usize,
            _: f64,
            _: f64,
        ) -> Result<(), OperationError> {
            panic!("CopyC core must submit one batch")
        }

        fn matmul_rank2_batch(
            &mut self,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            jobs: &[symtn_operations::fusion_replay::Rank2GemmBatchJob],
            _: &[usize],
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            assert_eq!((alpha, beta), (1.0, 0.0));
            self.calls += 1;
            self.jobs += jobs.len();
            Ok(())
        }
    }

    include!("../../tests/common/spy_executor.rs");

    /// The CopyC temporary is born zero and only the core GEMMs write it,
    /// at member offsets that do not depend on `B` (#1746, #1859): its
    /// inactive block stays zero across `B` changes and warm
    /// replays without a refill, so a warm call submits exactly what the
    /// first call on a fresh temporary did.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_copy_c_inactive_temporary_stays_zero_without_refill() {
        let runtime = Runtime::builder().cuda(0).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 2),
                (U1Irrep::new(2), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 55).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 56).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        };
        let pack = |tensors: &[TensorMap<_, f64>]| {
            StackedTensorMap::pack(&tensors.iter().collect::<Vec<_>>())
                .unwrap()
                .to_cuda()
                .unwrap()
        };
        let plan = ContractPlan::new(
            &pack(std::slice::from_ref(&a)),
            &pack(std::slice::from_ref(&b)),
            &spec,
        )
        .unwrap();
        assert!(plan.copy_c().is_some());
        assert_eq!(
            plan.resolution
                .core_plan()
                .inactive_destination_regions()
                .len(),
            1
        );
        let mut workspace = plan.workspace().unwrap();
        let mut first_submissions = None;
        for (call, count) in [2, 1, 2].into_iter().enumerate() {
            let lhs: Vec<_> = (0..count)
                .map(|i| a.scale(1.0 + (call * 2 + i) as f64 / 8.0).unwrap())
                .collect();
            let rhs: Vec<_> = (0..count)
                .map(|i| b.scale(1.0 - (call + i) as f64 / 16.0).unwrap())
                .collect();
            let expected: Vec<_> = lhs
                .iter()
                .zip(&rhs)
                .map(|(l, r)| l.contract(r, &spec).unwrap())
                .collect();
            let (lhs, rhs) = (pack(&lhs), pack(&rhs));
            let before = symtn_dense::cuda_transfer_stats();
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            let after = symtn_dense::cuda_transfer_stats();
            let submitted = after.gemm_calls - before.gemm_calls;
            let first = *first_submissions.get_or_insert(submitted);
            assert_eq!(submitted, first, "call {call}: no temporary refill");
            let poisoned: Vec<_> = expected
                .iter()
                .map(|e| e.scale(f64::NAN).unwrap())
                .collect();
            let mut dst = pack(&poisoned);
            let before = symtn_dense::cuda_transfer_stats();
            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                .unwrap();
            let after = symtn_dense::cuda_transfer_stats();
            assert_eq!(
                after.gemm_calls - before.gemm_calls,
                first,
                "call {call}: no temporary refill"
            );
            let actual = dst.to_host().unwrap();
            for (i, expected) in expected.iter().enumerate() {
                for (&value, &reference) in actual
                    .member(i)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(expected.dense_data().unwrap())
                {
                    assert!(value.is_finite(), "call {call} member {i}");
                    let tolerance =
                        128.0 * 64.0_f64.sqrt() * f64::EPSILON * reference.abs().max(1.0);
                    assert!((value - reference).abs() <= tolerance);
                }
            }
        }
    }

    #[test]
    fn labeled_copy_c_routes_are_admitted_at_every_payload_precision() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        macro_rules! cases {
            ($v:expr, $c:expr, $dtype:ty) => {{
                let v = $v;
                let c = $c;
                let tensor = |out: [&GradedSpace<_>; 2], input: [&GradedSpace<_>; 2], seed| {
                    TensorMap::<_, $dtype>::rand_with_seed(&runtime, out, input, seed).unwrap()
                };
                let a = tensor([v, v], [v, v], 61);
                let b = tensor([v, v], [v, v], 62);
                let cases = [
                    ("C1p", a.clone(), b.clone(), [3, 2], [1, 0], [1, 0, 3, 2]),
                    ("C2p", a, b, [0, 1], [2, 3], [3, 2, 1, 0]),
                    (
                        "S1",
                        tensor([c, c], [v, v], 101),
                        tensor([v, v], [c, c], 102),
                        [3, 2],
                        [1, 0],
                        [2, 3, 0, 1],
                    ),
                    (
                        "S2",
                        tensor([v, v], [c, c], 103),
                        tensor([c, c], [v, v], 104),
                        [0, 1],
                        [2, 3],
                        [3, 2, 1, 0],
                    ),
                ];
                for (name, lhs, rhs, lhs_axes, rhs_axes, output) in cases {
                    let spec = ContractSpec {
                        lhs: &lhs_axes,
                        rhs: &rhs_axes,
                        codomain: &output[..2],
                        domain: &output[2..],
                    };
                    for members in [1, 2, 17] {
                        let left = StackedTensorMap::pack(&vec![&lhs; members]).unwrap();
                        let right = StackedTensorMap::pack(&vec![&rhs; members]).unwrap();
                        let plan = ContractPlan::new(&left, &right, &spec).unwrap();
                        plan.copy_c()
                            .unwrap_or_else(|| panic!("{name} did not choose CopyC"));
                        let mut workspace = plan.workspace().unwrap();
                        plan.execute(&left, &right, &mut workspace).unwrap();
                    }
                }
            }};
        }
        let u1_v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 8),
                (U1Irrep::new(0), 8),
                (U1Irrep::new(1), 8),
            ],
        )
        .unwrap();
        let u1_c = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        cases!(&u1_v, &u1_c, f64);
        cases!(&u1_v, &u1_c, crate::typed::Complex64);
        cases!(&u1_v, &u1_c, f32);
        cases!(&u1_v, &u1_c, crate::typed::Complex32);
        let su2_v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let su2_c =
            GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 1)])
                .unwrap();
        cases!(&su2_v, &su2_c, f64);
        cases!(&su2_v, &su2_c, crate::typed::Complex64);
        cases!(&su2_v, &su2_c, f32);
        cases!(&su2_v, &su2_c, crate::typed::Complex32);
    }

    #[test]
    fn su2_copy_c_orientations_are_unconjugated_scaled_single_tasks() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 61).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 62).unwrap();
        for (lhs_axes, rhs_axes, codomain, domain) in [
            ([3, 2], [1, 0], [1, 0], [3, 2]),
            ([0, 1], [2, 3], [3, 2], [1, 0]),
        ] {
            let spec = ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &codomain,
                domain: &domain,
            };
            let lhs = StackedTensorMap::pack(&[&a]).unwrap();
            let rhs = StackedTensorMap::pack(&[&b]).unwrap();
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            let transform = plan.copy_c().expect("route must choose CopyC").transform();
            assert!(!transform.storage_conjugate());
            let coefficients: Vec<_> = transform
                .blocks()
                .iter()
                .map(|block| match *block {
                    symtn_operations::TreeTransformBlock::Single { coefficient, .. } => {
                        transform.single_coefficient(coefficient).unwrap()
                    }
                    symtn_operations::TreeTransformBlock::Multi { .. } => {
                        panic!("reached SU2 CopyC route must contain only Single moves")
                    }
                })
                .collect();
            assert!(coefficients.contains(&-1.0));
            assert!(coefficients
                .iter()
                .any(|&coefficient| coefficient != 1.0 && (coefficient - 1.0).abs() < 1e-12));
        }
    }

    /// What: the two primitives the executor replays a B > 1 CopyC route
    /// with — the planned core's signed stacked replay and the member
    /// output transform — each submit one member-expanded batch (jobs × B).
    /// Payload values are not checked here; the executor's member values
    /// are pinned against the base sequence and a dense oracle in
    /// `symtn-tensors` (`route_host_core_tests`).
    #[test]
    fn copy_c_core_and_transform_primitives_submit_member_expanded_batches() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 61).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 62).unwrap();
        let spec = ContractSpec {
            lhs: &[3, 2],
            rhs: &[1, 0],
            codomain: &[1, 0],
            domain: &[3, 2],
        };
        let mut core_jobs_per_member = 0;
        let mut transform_jobs_per_member = 0;
        for members in [1, 2, 17] {
            let lhs = StackedTensorMap::pack(&vec![&a; members]).unwrap();
            let rhs = StackedTensorMap::pack(&vec![&b; members]).unwrap();
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            let copy = plan.copy_c().expect("C1p must choose CopyC");
            assert!(
                copy.transform().blocks().iter().any(|block| match *block {
                    symtn_operations::TreeTransformBlock::Single { coefficient, .. } =>
                        copy.transform().single_coefficient(coefficient).unwrap() != 1.0,
                    symtn_operations::TreeTransformBlock::Multi { .. } => true,
                }),
                "this selected SU2 CopyC route must contain a scaled Single move"
            );
            // The member expansion the executor replays at B > 1.
            let core = Arc::clone(plan.resolution.direct_core().unwrap().0);
            let replay = StackedDirectReplay::new_signed(core, members).unwrap();
            let [dst_len, left_len, right_len] = replay.member_lens();
            let mut core_output = vec![f64::NAN; dst_len * members];
            let left = vec![1.0; left_len * members];
            let right = vec![1.0; right_len * members];
            let mut core = CoreCount::default();
            replay
                .execute_host(
                    &mut StridedHostKernelAdapter::default(),
                    &mut core,
                    &mut StackedStorageViewMut::new::<f64>(
                        &mut core_output,
                        dst_len,
                        members,
                        dst_len,
                    )
                    .unwrap(),
                    &StackedStorageView::new::<f64>(&left, left_len, members, left_len).unwrap(),
                    &StackedStorageView::new::<f64>(&right, right_len, members, right_len).unwrap(),
                    true,
                )
                .unwrap();
            assert_eq!(core.calls, 1);
            if members == 1 {
                core_jobs_per_member = core.jobs;
            }
            assert!(core_jobs_per_member > 0);
            assert_eq!(core.jobs, core_jobs_per_member * members);

            let mut output = vec![f64::NAN; plan.member_len * members];
            let transform_counts = Arc::new(SpyCounts::default());
            let mut transform = SpyExecutor::counting(&transform_counts);
            tree_transform_members_overwrite_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut transform,
                &mut TreeTransformWorkspace::default(),
                copy.transform(),
                plan.space.space().structure(),
                copy.temporary_structure(),
                &mut output,
                &vec![1.0; copy.temporary_len() * members],
                members,
                1,
                &[],
            )
            .unwrap();
            if members == 1 {
                transform_jobs_per_member = transform_counts.batch_jobs();
            }
            assert_eq!(
                transform_jobs_per_member,
                copy.transform().recoupling_plan().jobs().len()
            );
            assert_eq!(transform_jobs_per_member, 0);
            assert_eq!(
                (
                    transform_counts.of(&[Kernel::MatmulBatch, Kernel::MatmulBatchOps]),
                    transform_counts.batch_jobs()
                ),
                (
                    usize::from(transform_jobs_per_member > 0),
                    transform_jobs_per_member * members
                )
            );
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }
}
