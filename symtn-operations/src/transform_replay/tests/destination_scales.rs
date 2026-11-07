use super::*;
use num_complex::Complex64;
use symtn_dense::CpuEngine;

/// Five 2x2 destination blocks: two Singles (one transposed), one two-block
/// Multi, and an inactive block; four 2x2 source blocks.
fn fixture() -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 4]).unwrap());
    let dst = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 5]).unwrap());
    let structure = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 2.0),
            TreeTransformBlockSpec::single(1, 1, -0.5).with_source_axes([1, 0]),
            TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![1.0, 2.0, 3.0, 4.0]),
        ],
    )
    .unwrap();
    (dst, src, structure)
}

/// θ = -1 on the transposed Single and on both Multi destinations (the
/// contraction compiler keeps θ uniform per Multi block); block 0 unlisted.
const SCALES: [(usize, f64); 3] = [(4, -1.0), (8, -1.0), (12, -1.0)];

fn source<D: From<f64> + core::ops::Mul<Output = D> + Copy>(members: usize, unit: D) -> Vec<D> {
    (0..16 * members)
        .map(|index| unit * D::from(0.25 * index as f64 + 1.0))
        .collect()
}

/// The literal two-step sequence: the unscaled overwrite, then block `b`
/// multiplied by θ_b in place.
fn oracle<D>(
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    structure: &TreeTransformStructure<f64>,
    source: &[D],
) -> Vec<D>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + ConjugateValue,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut out = vec![D::zero(); 20];
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        structure,
        dst,
        src,
        &mut out,
        source,
        D::one(),
        &[],
    )
    .unwrap();
    for &(offset, theta) in &SCALES {
        for value in &mut out[offset..offset + 4] {
            *value = value.scale_by_coefficient(theta);
        }
    }
    out
}

fn check<D>(unit: D)
where
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + From<f64>
        + core::fmt::Debug,
{
    // One worker, and four workers (with the built-in executor, on forced
    // `Outer` lanes where it honors forced policies; through the forwarding
    // executor, which does not, under `Auto`). A member batch moves every
    // member on the caller's token around one dense submission.
    let engine = |threads: usize, forced: bool| {
        let strategy = if forced {
            tenferro_cpu::CpuBatchStrategy::Outer
        } else {
            tenferro_cpu::CpuBatchStrategy::Auto
        };
        crate::host_pool::eager_test_engine(threads)
            .with_batch_policy(tenferro_cpu::CpuBatchPolicy::new(strategy))
    };
    let honors = DefaultDenseExecutor::new().honors_forced_batch_policy();
    let routes = [(1, false), (4, false), (4, true)];
    let (dst, src, structure) = fixture();
    for members in [1, 2, 3] {
        let source = source(members, unit);
        let expected: Vec<D> = source
            .chunks(16)
            .flat_map(|member| oracle(&dst, &src, &structure, member))
            .collect();
        assert!(
            expected[4..16].iter().any(|value| *value != D::zero()),
            "vacuous twisted blocks"
        );
        for (threads, barrier) in routes {
            let mut actual = vec![D::from(f64::NAN); 20 * members];
            let mut workspace = TreeTransformWorkspace::default();
            let forced = honors && !barrier;
            let mut engine = engine(threads, forced);
            let (dst, src, source) = (&dst, &src, &source[..]);
            let ran = if barrier {
                overwrite_members(
                    &mut engine,
                    &mut super::recording::Forwarding::default(),
                    &mut workspace,
                    &structure,
                    [dst, src],
                    &mut actual,
                    source,
                    members,
                )
            } else {
                overwrite_members(
                    &mut engine,
                    &mut DefaultDenseExecutor::new(),
                    &mut workspace,
                    &structure,
                    [dst, src],
                    &mut actual,
                    source,
                    members,
                )
            };
            ran.unwrap();
            let what = format!("B={members} threads={threads} barrier={barrier}");
            assert_eq!(actual, expected, "{what}");
            if threads > 1 && members == 1 && forced {
                assert!(workspace.lanes.placement().lanes() > 1, "{what}: one lane");
            }
        }
        if members == 1 {
            for threads in [1, 4] {
                let mut actual = vec![D::from(f64::NAN); 20];
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    &mut engine(threads, honors),
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &structure,
                    &dst,
                    &src,
                    &mut actual,
                    &source,
                    D::one(),
                    &SCALES,
                )
                .unwrap();
                assert_eq!(actual, expected, "threads={threads}");
            }
        }
    }
}

/// The scaled member overwrite of `source` through `dense`.
#[allow(clippy::too_many_arguments)]
fn overwrite_members<E, D>(
    engine: &mut CpuEngine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<f64>,
    [dst, src]: [&Arc<BlockStructure>; 2],
    actual: &mut [D],
    source: &[D],
    members: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + ConjugateValue,
{
    tree_transform_members_overwrite_raw(
        engine,
        &mut StridedHostKernelAdapter::default(),
        dense,
        workspace,
        structure,
        dst,
        src,
        actual,
        source,
        members,
        &SCALES,
    )
}

#[test]
fn destination_scales_fold_into_single_moves_and_multi_scatters() {
    // What: folding θ_b into the move writing block b equals the unscaled
    // overwrite followed by an in-place scale of block b, bit for bit, for
    // Singles and Multi scatters, on one worker and on four workers (the
    // built-in and the forwarding executor), B = 1/2/3, f64/c64.
    check(1.0_f64);
    check(Complex64::new(1.0, -0.5));
}

#[test]
fn unsorted_destination_scales_are_rejected_before_any_write() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (dst, src, structure) = fixture();
    let source = source(2, 1.0_f64);
    let bad = [(8, -1.0), (4, -1.0)];
    for members in [1, 2] {
        let mut out = vec![7.0; 20 * members];
        let error = tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst,
            &src,
            &mut out,
            &source[..16 * members],
            members,
            &bad,
        )
        .unwrap_err();
        assert!(
            matches!(error, OperationError::InvalidArgument { message } if message.contains("strictly increasing"))
        );
        assert!(out.iter().all(|&value| value == 7.0));
    }
}
