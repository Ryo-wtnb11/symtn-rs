use super::*;
use num_complex::Complex64;

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
    let mut out = vec![D::zero(); 20];
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
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
        1,
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
        for threads in [1, 4] {
            let mut actual = vec![D::from(f64::NAN); 20 * members];
            tree_transform_members_overwrite_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst,
                &src,
                &mut actual,
                &source,
                members,
                threads,
                &SCALES,
            )
            .unwrap();
            assert_eq!(actual, expected, "B={members} threads={threads}");
        }
        if members == 1 {
            for threads in [1, 4] {
                let mut actual = vec![D::from(f64::NAN); 20];
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
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
                    threads,
                )
                .unwrap();
                assert_eq!(actual, expected, "threads={threads}");
            }
        }
    }
}

#[test]
fn destination_scales_fold_into_single_moves_and_multi_scatters() {
    // What: folding θ_b into the move writing block b equals the unscaled
    // overwrite followed by an in-place scale of block b, bit for bit, for
    // Singles and Multi scatters, serial and threaded, B = 1/2/3, f64/c64.
    check(1.0_f64);
    check(Complex64::new(1.0, -0.5));
}

#[test]
fn unsorted_destination_scales_are_rejected_before_any_write() {
    let (dst, src, structure) = fixture();
    let source = source(2, 1.0_f64);
    let bad = [(8, -1.0), (4, -1.0)];
    for members in [1, 2] {
        let mut out = vec![7.0; 20 * members];
        let error = tree_transform_members_overwrite_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst,
            &src,
            &mut out,
            &source[..16 * members],
            members,
            1,
            &bad,
        )
        .unwrap_err();
        assert!(
            matches!(error, OperationError::InvalidArgument { message } if message.contains("strictly increasing"))
        );
        assert!(out.iter().all(|&value| value == 7.0));
    }
}
