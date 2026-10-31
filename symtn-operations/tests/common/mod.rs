//! Fixtures and the independent oracle shared by the device tree-transform
//! tests and by the CPU-only test that proves the oracle itself.
//!
//! The oracle is an explicit index walk over the fixture's own block metadata:
//! it never reads a `TreeTransformStructure`, its baked fused layouts, or any
//! executor state, so "device matches the oracle" and "host matches the oracle"
//! are independent statements about the same fixture.

#![allow(dead_code)]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn_core::{BlockKey, BlockSpec, BlockStructure};
use symtn_operations::{TreeTransformBlockSpec, TreeTransformStructure};

/// VectorInterface's `scale(x, α) = (iszero(α) ? zero(x) : x) * α`.
fn vi_scale<T: TestScalar>(value: T, alpha: T) -> T {
    if alpha == T::zero() {
        T::zero().mul(alpha)
    } else {
        alpha.mul(value)
    }
}

/// Payload dtypes the fixtures are replayed with, plus the host arithmetic the
/// oracle needs. Deliberately not a symtn trait: the oracle must not share a
/// scalar contract with the code under test.
pub trait TestScalar: Copy + std::fmt::Debug + PartialEq + 'static {
    const NAME: &'static str;
    /// Machine epsilon of this payload's real lane, widened, so a tolerance
    /// can be written in epsilons rather than as an absolute constant.
    const EPSILON: f64;
    fn from_parts(re: f64, im: f64) -> Self;
    fn zero() -> Self {
        Self::from_parts(0.0, 0.0)
    }
    fn scale(self, factor: f64) -> Self;
    fn add(self, other: Self) -> Self;
    /// Payload multiplication, for the caller scale `alpha`.
    fn mul(self, other: Self) -> Self;
    fn conjugate(self) -> Self;
    fn distance(self, other: Self) -> f64;
    /// Deterministic, non-degenerate sample for buffer index `index`.
    fn sample(index: usize) -> Self {
        Self::from_parts(1.0 + (index as f64) * 0.5, -0.25 + (index as f64) * 0.125)
    }
    fn is_nan(self) -> bool;
}

impl TestScalar for f32 {
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_parts(re: f64, _im: f64) -> Self {
        re as Self
    }

    fn scale(self, factor: f64) -> Self {
        self * factor as Self
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn conjugate(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        f64::from((self - other).abs())
    }

    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl TestScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }

    fn scale(self, factor: f64) -> Self {
        self * factor as f32
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn conjugate(self) -> Self {
        Complex32::new(self.re, -self.im)
    }

    fn distance(self, other: Self) -> f64 {
        f64::from((self - other).norm())
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl TestScalar for f64 {
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;

    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }

    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn conjugate(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl TestScalar for Complex64 {
    const NAME: &'static str = "Complex64";
    const EPSILON: f64 = f64::EPSILON;

    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }

    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn conjugate(self) -> Self {
        Complex64::new(self.re, -self.im)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// One block of a fixture's storage: extents, element strides, flat offset.
#[derive(Clone, Debug)]
pub struct Block {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Block {
    pub fn packed(shape: Vec<usize>, offset: usize) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut running = 1usize;
        for &dim in &shape {
            strides.push(running);
            running *= dim;
        }
        Self {
            shape,
            strides,
            offset,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn end_exclusive(&self) -> usize {
        if self.shape.contains(&0) {
            return self.offset;
        }
        self.offset
            + self
                .shape
                .iter()
                .zip(&self.strides)
                .map(|(dim, stride)| (dim - 1) * stride)
                .sum::<usize>()
            + 1
    }

    fn spec(&self, index: usize) -> BlockSpec {
        BlockSpec::with_key(
            BlockKey::ordinal(index),
            self.shape.clone(),
            self.strides.clone(),
            self.offset,
        )
        .unwrap()
    }
}

/// One replayed block pair: `axes[dst_axis] = src_axis`.
#[derive(Clone, Debug)]
pub struct Pair {
    pub dst_block: usize,
    pub src_block: usize,
    pub axes: Vec<usize>,
    pub coefficient: f64,
}

/// One replayed recoupling group: every destination block is the `u`-weighted
/// sum of every source block, each source read through `axes`.
///
/// `u` is row-major `U[dst][src]`, i.e. `u[dst * src_blocks.len() + src]` — the
/// layout `TreeTransformBlockSpec::multi` documents. The oracle applies it as
/// the plain sum below and never reads the compiled structure's recoupling
/// plan, its packed columns or its GEMM orientation, so the two statements stay
/// independent.
#[derive(Clone, Debug)]
pub struct Group {
    pub dst_blocks: Vec<usize>,
    pub src_blocks: Vec<usize>,
    pub axes: Vec<usize>,
    pub u: Vec<f64>,
}

impl Group {
    fn coefficient(&self, dst_index: usize, src_index: usize) -> f64 {
        self.u[dst_index * self.src_blocks.len() + src_index]
    }
}

/// A complete tree-transform fixture: two block structures, the block pairs and
/// recoupling groups the transform replays, and whether the source is read
/// conjugated.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub rank: usize,
    pub dst_blocks: Vec<Block>,
    pub src_blocks: Vec<Block>,
    pub pairs: Vec<Pair>,
    pub groups: Vec<Group>,
    pub conjugate: bool,
}

impl Fixture {
    pub fn dst_structure(&self) -> Arc<BlockStructure> {
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                self.rank,
                self.dst_blocks
                    .iter()
                    .enumerate()
                    .map(|(index, block)| block.spec(index))
                    .collect(),
            )
            .unwrap(),
        )
    }

    pub fn src_structure(&self) -> Arc<BlockStructure> {
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                self.rank,
                self.src_blocks
                    .iter()
                    .enumerate()
                    .map(|(index, block)| block.spec(index))
                    .collect(),
            )
            .unwrap(),
        )
    }

    pub fn dst_len(&self) -> usize {
        self.dst_blocks
            .iter()
            .map(Block::end_exclusive)
            .max()
            .unwrap_or(0)
    }

    pub fn src_len(&self) -> usize {
        self.src_blocks
            .iter()
            .map(Block::end_exclusive)
            .max()
            .unwrap_or(0)
    }

    pub fn compile(&self) -> TreeTransformStructure<f64> {
        let specs: Vec<TreeTransformBlockSpec<f64>> = self
            .pairs
            .iter()
            .map(|pair| {
                TreeTransformBlockSpec::single(pair.dst_block, pair.src_block, pair.coefficient)
                    .with_source_axes(pair.axes.iter().copied())
            })
            .chain(self.groups.iter().map(|group| {
                TreeTransformBlockSpec::multi(
                    group.dst_blocks.clone(),
                    group.src_blocks.clone(),
                    group.u.clone(),
                )
                .with_source_axes(group.axes.iter().copied())
            }))
            .collect();
        TreeTransformStructure::compile_structures_with_storage_conjugation(
            &self.dst_structure(),
            &self.src_structure(),
            &specs,
            self.conjugate,
        )
        .unwrap()
    }

    pub fn source<T: TestScalar>(&self) -> Vec<T> {
        (0..self.src_len()).map(T::sample).collect()
    }

    /// The expected destination buffer, computed by walking every block pair's
    /// index space explicitly.
    ///
    /// `Overwrite` assigns the active elements and zeroes every destination
    /// element that belongs to a block no pair writes; accumulate (`beta == 1`)
    /// adds into `destination` and leaves untouched blocks alone. Nothing here
    /// consults the compiled structure.
    pub fn expected<T: TestScalar>(
        &self,
        source: &[T],
        destination: &[T],
        overwrite: bool,
    ) -> Vec<T> {
        self.expected_scaled(source, destination, overwrite, T::from_parts(1.0, 0.0))
    }

    /// [`Self::expected`] with the caller scale `alpha`, applied where the host
    /// applies it: `(alpha * coefficient) * src` for a Single block and
    /// `alpha * (U x)` for a recoupling group — never on a pack, never inside
    /// the recoupling sum, and never on a zero fill, which stays an exact zero
    /// whatever `alpha` is. A zero scale is VectorInterface's
    /// `scale(x, 0) = zero(x) * 0`: an exact zero whatever the source
    /// holds, so a NaN source does not reach the destination.
    pub fn expected_scaled<T: TestScalar>(
        &self,
        source: &[T],
        destination: &[T],
        overwrite: bool,
        alpha: T,
    ) -> Vec<T> {
        let mut expected = destination.to_vec();
        if overwrite {
            let written: Vec<usize> = self
                .pairs
                .iter()
                .map(|pair| pair.dst_block)
                .chain(
                    self.groups
                        .iter()
                        .flat_map(|group| group.dst_blocks.clone()),
                )
                .collect();
            for (index, block) in self.dst_blocks.iter().enumerate() {
                if written.contains(&index) {
                    continue;
                }
                for position in positions(block, &(0..self.rank).collect::<Vec<_>>()) {
                    expected[position] = T::zero();
                }
            }
        }
        for pair in &self.pairs {
            let dst = &self.dst_blocks[pair.dst_block];
            let src = &self.src_blocks[pair.src_block];
            let identity: Vec<usize> = (0..self.rank).collect();
            for (dst_position, src_position) in
                positions(dst, &identity).zip(positions(src, &pair.axes))
            {
                let mut value = source[src_position];
                if self.conjugate {
                    value = value.conjugate();
                }
                let value = vi_scale(value, alpha.scale(pair.coefficient));
                expected[dst_position] = if overwrite {
                    value
                } else {
                    expected[dst_position].add(value)
                };
            }
        }
        let identity: Vec<usize> = (0..self.rank).collect();
        for group in &self.groups {
            for (dst_index, &dst_block) in group.dst_blocks.iter().enumerate() {
                let dst = &self.dst_blocks[dst_block];
                let dst_positions: Vec<usize> = positions(dst, &identity).collect();
                let mut column = vec![T::zero(); dst_positions.len()];
                for (src_index, &src_block) in group.src_blocks.iter().enumerate() {
                    let coefficient = group.coefficient(dst_index, src_index);
                    let src = &self.src_blocks[src_block];
                    for (slot, src_position) in column.iter_mut().zip(positions(src, &group.axes)) {
                        let mut value = source[src_position];
                        if self.conjugate {
                            value = value.conjugate();
                        }
                        *slot = slot.add(value.scale(coefficient));
                    }
                }
                for (&dst_position, value) in dst_positions.iter().zip(column) {
                    let value = vi_scale(value, alpha);
                    expected[dst_position] = if overwrite {
                        value
                    } else {
                        expected[dst_position].add(value)
                    };
                }
            }
        }
        expected
    }
}

/// Flat positions of `block`, walked in the *destination* axis order given by
/// `axes` (`axes[dst_axis] = block_axis`), fastest destination axis first.
fn positions<'a>(block: &'a Block, axes: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
    let extents: Vec<usize> = axes.iter().map(|&axis| block.shape[axis]).collect();
    let strides: Vec<usize> = axes.iter().map(|&axis| block.strides[axis]).collect();
    let count: usize = extents.iter().product();
    (0..count).map(move |linear| {
        let mut remaining = linear;
        let mut position = block.offset;
        for (extent, stride) in extents.iter().zip(&strides) {
            position += (remaining % extent) * stride;
            remaining /= extent;
        }
        position
    })
}

fn reversed(rank: usize) -> Vec<usize> {
    (0..rank).rev().collect()
}

fn permuted(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    axes.iter().map(|&axis| shape[axis]).collect()
}

/// A destination block whose extents are `axes`-permuted from the source's.
fn permuted_pair(
    src_shape: Vec<usize>,
    axes: Vec<usize>,
    coefficient: f64,
    dst_offset: usize,
    src_offset: usize,
) -> (Block, Block, Pair) {
    let dst_shape = permuted(&src_shape, &axes);
    (
        Block::packed(dst_shape, dst_offset),
        Block::packed(src_shape, src_offset),
        Pair {
            dst_block: 0,
            src_block: 0,
            axes,
            coefficient,
        },
    )
}

fn single_pair_fixture(
    name: &'static str,
    src_shape: Vec<usize>,
    axes: Vec<usize>,
    coefficient: f64,
    conjugate: bool,
) -> Fixture {
    let rank = src_shape.len();
    let (dst, src, pair) = permuted_pair(src_shape, axes, coefficient, 0, 0);
    Fixture {
        name,
        rank,
        dst_blocks: vec![dst],
        src_blocks: vec![src],
        pairs: vec![pair],
        groups: Vec::new(),
        conjugate,
    }
}

/// Rank 2–6 permutes and transposes, including a fermionic sign (coefficient
/// −1), a coefficient that is neither 1 nor −1, and a conjugated source.
pub fn rank_sweep() -> Vec<Fixture> {
    vec![
        single_pair_fixture("rank2_transpose", vec![3, 2], vec![1, 0], 1.0, false),
        single_pair_fixture("rank2_fermionic_sign", vec![3, 2], vec![1, 0], -1.0, false),
        single_pair_fixture(
            "rank3_cyclic_permute",
            vec![2, 3, 4],
            vec![2, 0, 1],
            1.0,
            false,
        ),
        single_pair_fixture(
            "rank3_coefficient_not_unit",
            vec![2, 3, 4],
            vec![1, 2, 0],
            0.625,
            false,
        ),
        single_pair_fixture(
            "rank4_reverse_conjugated",
            vec![2, 3, 2, 2],
            reversed(4),
            -1.0,
            true,
        ),
        single_pair_fixture(
            "rank5_permute",
            vec![2, 2, 3, 2, 2],
            vec![3, 0, 4, 1, 2],
            0.75,
            false,
        ),
        single_pair_fixture(
            "rank6_reverse",
            vec![2, 2, 2, 3, 2, 2],
            reversed(6),
            -1.0,
            false,
        ),
        single_pair_fixture("rank2_identity", vec![4, 5], vec![0, 1], 1.0, false),
    ]
}

/// Several blocks whose storage interleaves: the destination blocks are not in
/// the source's order and their layouts are gapped sub-boxes of a wider parent.
pub fn interleaved_multi_block() -> Fixture {
    // Destination storage: block 0 occupies the even columns of a 4x4 parent,
    // block 1 the odd ones, so the two layouts interleave in memory.
    let dst_blocks = vec![
        Block {
            shape: vec![4, 2],
            strides: vec![1, 8],
            offset: 0,
        },
        Block {
            shape: vec![4, 2],
            strides: vec![1, 8],
            offset: 4,
        },
    ];
    let src_blocks = vec![
        Block::packed(vec![2, 4], 0),
        Block::packed(vec![2, 4], 8),
        Block::packed(vec![3, 3], 16),
    ];
    Fixture {
        name: "interleaved_multi_block",
        rank: 2,
        dst_blocks,
        src_blocks,
        // Reversed pair order as well: block order is the structure's, and the
        // destinations are disjoint, so the result must not depend on it.
        pairs: vec![
            Pair {
                dst_block: 1,
                src_block: 0,
                axes: vec![1, 0],
                coefficient: -1.0,
            },
            Pair {
                dst_block: 0,
                src_block: 1,
                axes: vec![1, 0],
                coefficient: 2.5,
            },
        ],
        groups: Vec::new(),
        conjugate: false,
    }
}

/// A destination with layouts no pair writes, which `Overwrite` must zero even
/// when they hold NaN.
pub fn inactive_destination_layouts() -> Fixture {
    let dst_blocks = vec![
        Block::packed(vec![2, 3], 0),
        Block::packed(vec![3, 2], 6),
        Block::packed(vec![2, 2], 12),
    ];
    let src_blocks = vec![Block::packed(vec![3, 2], 0)];
    Fixture {
        name: "inactive_destination_layouts",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs: vec![Pair {
            dst_block: 0,
            src_block: 0,
            axes: vec![1, 0],
            coefficient: -1.0,
        }],
        groups: Vec::new(),
        conjugate: false,
    }
}

/// Inactive destinations whose compiled zero region does not fuse to one
/// axis: a padded block with an extent-1 middle axis, `[2, 1, 3]`
/// with strides `[1, 1, 4]`, which normalizes to `[2, 3]` / `[1, 4]` and
/// leaves storage gaps that no block owns; and, in
/// [`rank_zero_inactive_destination`], a scalar block, whose region is `[1]`.
pub fn padded_inactive_destination_layouts() -> Fixture {
    let dst_blocks = vec![
        Block::packed(vec![2, 3, 1], 0),
        Block {
            shape: vec![2, 1, 3],
            strides: vec![1, 1, 4],
            offset: 6,
        },
    ];
    let src_blocks = vec![Block::packed(vec![3, 2, 1], 0)];
    Fixture {
        name: "padded_inactive_destination_layouts",
        rank: 3,
        dst_blocks,
        src_blocks,
        pairs: vec![Pair {
            dst_block: 0,
            src_block: 0,
            axes: vec![1, 0, 2],
            coefficient: -1.5,
        }],
        groups: Vec::new(),
        conjugate: false,
    }
}

/// Rank-0 blocks, one written and one inactive: see
/// [`padded_inactive_destination_layouts`].
pub fn rank_zero_inactive_destination() -> Fixture {
    Fixture {
        name: "rank_zero_inactive_destination",
        rank: 0,
        dst_blocks: vec![Block::packed(Vec::new(), 0), Block::packed(Vec::new(), 1)],
        src_blocks: vec![Block::packed(Vec::new(), 0)],
        pairs: vec![Pair {
            dst_block: 1,
            src_block: 0,
            axes: Vec::new(),
            coefficient: 2.0,
        }],
        groups: Vec::new(),
        conjugate: false,
    }
}

/// A structure with more distinct baked layouts than Tenferro's default
/// 64-entry cuTENSOR plan bound, so a replay that does not raise the bound
/// evicts a plan it needs again on the next block.
pub fn many_distinct_signatures(blocks: usize) -> Fixture {
    let mut dst_blocks = Vec::with_capacity(blocks);
    let mut src_blocks = Vec::with_capacity(blocks);
    let mut pairs = Vec::with_capacity(blocks);
    let mut dst_offset = 0usize;
    let mut src_offset = 0usize;
    for index in 0..blocks {
        // Distinct extents give distinct fused signatures; the source is
        // transposed so the move is never a plain copy.
        let shape = vec![2 + index % 7, 3 + index / 7];
        let dst = Block::packed(vec![shape[1], shape[0]], dst_offset);
        let src = Block::packed(shape, src_offset);
        dst_offset += dst.element_count();
        src_offset += src.element_count();
        pairs.push(Pair {
            dst_block: index,
            src_block: index,
            axes: vec![1, 0],
            coefficient: if index % 2 == 0 { 1.0 } else { -1.0 },
        });
        dst_blocks.push(dst);
        src_blocks.push(src);
    }
    Fixture {
        name: "many_distinct_signatures",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs,
        groups: Vec::new(),
        conjugate: false,
    }
}

/// A fixture containing a zero-extent block pair next to a live one.
pub fn zero_extent_block() -> Fixture {
    Fixture {
        name: "zero_extent_block",
        rank: 2,
        dst_blocks: vec![
            Block::packed(vec![2, 3], 0),
            Block {
                shape: vec![0, 2],
                strides: vec![1, 1],
                offset: 6,
            },
        ],
        src_blocks: vec![
            Block::packed(vec![3, 2], 0),
            Block {
                shape: vec![2, 0],
                strides: vec![1, 1],
                offset: 6,
            },
        ],
        pairs: vec![
            Pair {
                dst_block: 0,
                src_block: 0,
                axes: vec![1, 0],
                coefficient: 1.5,
            },
            Pair {
                dst_block: 1,
                src_block: 1,
                axes: vec![1, 0],
                coefficient: 1.0,
            },
        ],
        groups: Vec::new(),
        conjugate: false,
    }
}

/// A destination layout the host proves injective only through its exact
/// overlap fallback: dims [3, 2] with strides [2, 3] addresses 0, 2, 4, 3, 5, 7
/// — distinct, but each stride does not exceed the span of the faster axis, so
/// the device's cumulative-span rule (and Tenferro's) rejects it.
///
/// Host replays it; the device must report it as a capability boundary, and
/// must do so before it has cost the caller anything.
pub fn expert_interleaved_destination() -> Fixture {
    Fixture {
        name: "expert_interleaved_destination",
        rank: 2,
        dst_blocks: vec![Block {
            shape: vec![3, 2],
            strides: vec![2, 3],
            offset: 0,
        }],
        src_blocks: vec![Block::packed(vec![2, 3], 0)],
        pairs: vec![Pair {
            dst_block: 0,
            src_block: 0,
            axes: vec![1, 0],
            coefficient: 1.0,
        }],
        groups: Vec::new(),
        conjugate: false,
    }
}

/// The fixtures whose every coefficient is exactly 1, where an f64 device move
/// is a multiply by one and therefore bit-identical to the host's copy.
///
/// A recoupling group never qualifies: its destination is a GEMM result, not a
/// move, so bit-equality with the host's own GEMM is not a contract.
pub fn unit_coefficient_fixtures() -> Vec<Fixture> {
    all_fixtures()
        .into_iter()
        .filter(|fixture| {
            fixture.groups.is_empty() && fixture.pairs.iter().all(|pair| pair.coefficient == 1.0)
        })
        .collect()
}

/// A square recoupling group whose `U` is *not* symmetric, so replaying it with
/// `U` where the host uses `Uᵀ` gives a different answer.
///
/// Both destination blocks are the weighted sum of both transposed source
/// blocks, which is the shape of an SU(2) recoupling over two fusion channels
/// of one coupled sector: the degeneracy box is permuted and the channels mix.
pub fn recoupling_non_symmetric_u() -> Fixture {
    Fixture {
        name: "recoupling_non_symmetric_u",
        rank: 2,
        dst_blocks: vec![Block::packed(vec![2, 3], 0), Block::packed(vec![2, 3], 6)],
        src_blocks: vec![Block::packed(vec![3, 2], 0), Block::packed(vec![3, 2], 6)],
        pairs: Vec::new(),
        groups: vec![Group {
            dst_blocks: vec![0, 1],
            src_blocks: vec![0, 1],
            axes: vec![1, 0],
            // U[0] = [1, 2], U[1] = [0.5, -3]: U != U^T, and no row or column
            // is a multiple of another, so neither a transposed nor a
            // conjugated orientation reproduces it.
            u: vec![1.0, 2.0, 0.5, -3.0],
        }],
        conjugate: false,
    }
}

/// A recoupling group with more sources than destinations, where `U` is not
/// even square: a transposed orientation is a shape error rather than a wrong
/// value, and the `(rows, contracted, cols)` triple is pinned by construction.
pub fn recoupling_rectangular() -> Fixture {
    Fixture {
        name: "recoupling_rectangular",
        rank: 2,
        dst_blocks: vec![Block::packed(vec![2, 3], 0), Block::packed(vec![2, 3], 6)],
        src_blocks: vec![
            Block::packed(vec![3, 2], 0),
            Block::packed(vec![3, 2], 6),
            Block::packed(vec![3, 2], 12),
        ],
        pairs: Vec::new(),
        groups: vec![Group {
            dst_blocks: vec![0, 1],
            src_blocks: vec![0, 1, 2],
            axes: vec![1, 0],
            u: vec![1.0, 2.0, 3.0, -0.5, 0.25, 4.0],
        }],
        conjugate: false,
    }
}

/// Recoupling groups and Single blocks in one structure, over interleaved
/// destination layouts and beside a destination block nothing writes.
///
/// The two groups have different element counts, so the compile-time recoupling
/// plan sorts them into an order that is not the block order and gives them
/// different workspace offsets — the device must follow the plan's offsets, not
/// the block's.
pub fn mixed_single_and_multi() -> Fixture {
    let dst_blocks = vec![
        // Even and odd columns of one 4x4 parent: the two group destinations
        // interleave in memory.
        Block {
            shape: vec![4, 2],
            strides: vec![1, 8],
            offset: 0,
        },
        Block {
            shape: vec![4, 2],
            strides: vec![1, 8],
            offset: 4,
        },
        Block::packed(vec![3, 2], 16),
        Block::packed(vec![3, 2], 22),
        Block::packed(vec![2, 2], 28),
        // Never written: Overwrite must zero it.
        Block::packed(vec![2, 2], 32),
    ];
    let src_blocks = vec![
        Block::packed(vec![2, 4], 0),
        Block::packed(vec![2, 4], 8),
        Block::packed(vec![2, 3], 16),
        Block::packed(vec![2, 3], 22),
        Block::packed(vec![2, 2], 28),
    ];
    Fixture {
        name: "mixed_single_and_multi",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs: vec![Pair {
            dst_block: 4,
            src_block: 4,
            axes: vec![1, 0],
            coefficient: -1.5,
        }],
        groups: vec![
            Group {
                dst_blocks: vec![0, 1],
                src_blocks: vec![0, 1],
                axes: vec![1, 0],
                u: vec![0.25, -2.0, 1.0, 0.5],
            },
            Group {
                dst_blocks: vec![2, 3],
                src_blocks: vec![2, 3],
                axes: vec![1, 0],
                u: vec![1.0, -0.75, 2.5, 0.125],
            },
        ],
        conjugate: false,
    }
}

/// A rank-3 recoupling group read from a conjugated source, with a coefficient
/// that is neither 1 nor -1 on every entry of `U`.
pub fn conjugated_recoupling() -> Fixture {
    Fixture {
        name: "conjugated_recoupling",
        rank: 3,
        dst_blocks: vec![
            Block::packed(vec![2, 3, 2], 0),
            Block::packed(vec![2, 3, 2], 12),
        ],
        src_blocks: vec![
            Block::packed(vec![3, 2, 2], 0),
            Block::packed(vec![3, 2, 2], 12),
        ],
        pairs: Vec::new(),
        groups: vec![Group {
            dst_blocks: vec![0, 1],
            src_blocks: vec![0, 1],
            axes: vec![1, 0, 2],
            u: vec![0.625, -1.25, 2.0, 0.375],
        }],
        conjugate: true,
    }
}

/// A recoupling group with one source and several destinations, and its
/// mirror with several sources and one destination.
///
/// `1xN` and `Nx1` are the shapes a real fusion-tree population produces
/// whenever a coupled sector has one channel on one side and several on the
/// other. They also pin the GEMM's degenerate dimensions: a `1xN` job is
/// `contracted == 1` and an `Nx1` job is `cols == 1`, either of which a
/// shape-inferring lowering could silently transpose.
pub fn recoupling_one_to_many() -> Fixture {
    Fixture {
        name: "recoupling_one_to_many",
        rank: 2,
        dst_blocks: vec![
            Block::packed(vec![2, 3], 0),
            Block::packed(vec![2, 3], 6),
            Block::packed(vec![2, 3], 12),
        ],
        src_blocks: vec![Block::packed(vec![3, 2], 0)],
        pairs: Vec::new(),
        groups: vec![Group {
            dst_blocks: vec![0, 1, 2],
            src_blocks: vec![0],
            axes: vec![1, 0],
            u: vec![1.5, -0.25, 3.0],
        }],
        conjugate: false,
    }
}

pub fn recoupling_many_to_one() -> Fixture {
    Fixture {
        name: "recoupling_many_to_one",
        rank: 2,
        dst_blocks: vec![Block::packed(vec![2, 3], 0)],
        src_blocks: vec![
            Block::packed(vec![3, 2], 0),
            Block::packed(vec![3, 2], 6),
            Block::packed(vec![3, 2], 12),
        ],
        pairs: Vec::new(),
        groups: vec![Group {
            dst_blocks: vec![0],
            src_blocks: vec![0, 1, 2],
            axes: vec![1, 0],
            u: vec![1.5, -0.25, 3.0],
        }],
        conjugate: false,
    }
}

/// Two recoupling groups of exactly the same `(element_count, src_count,
/// dst_count)`, so their GEMMs share one cuTENSOR plan and the recoupling plan
/// orders them by block index rather than by shape.
///
/// This is the common case in a real fusion-tree population — many coupled
/// sectors with the same channel count — and the one where a device that
/// addressed a job's matrix by anything but its own offset would silently use
/// its neighbour's.
pub fn equal_shape_recoupling_jobs() -> Fixture {
    Fixture {
        name: "equal_shape_recoupling_jobs",
        rank: 2,
        dst_blocks: (0..4).map(|i| Block::packed(vec![2, 3], i * 6)).collect(),
        src_blocks: (0..4).map(|i| Block::packed(vec![3, 2], i * 6)).collect(),
        pairs: Vec::new(),
        groups: vec![
            Group {
                dst_blocks: vec![0, 1],
                src_blocks: vec![0, 1],
                axes: vec![1, 0],
                u: vec![1.0, 2.0, 0.5, -3.0],
            },
            Group {
                dst_blocks: vec![2, 3],
                src_blocks: vec![2, 3],
                axes: vec![1, 0],
                // Deliberately a different matrix of the same shape: swapping
                // the two jobs' matrices must change the answer.
                u: vec![-0.75, 4.0, 2.25, 1.0],
            },
        ],
        conjugate: false,
    }
}

/// A recoupling group one of whose scatter destinations is the interleaved
/// layout only the host's exact overlap fallback admits.
///
/// The device must reject the whole structure — including the pack columns and
/// the coefficient upload it would otherwise do first — on the strength of that
/// one scatter region.
pub fn expert_interleaved_recoupling_destination() -> Fixture {
    Fixture {
        name: "expert_interleaved_recoupling_destination",
        rank: 2,
        dst_blocks: vec![
            Block::packed(vec![3, 2], 0),
            Block {
                shape: vec![3, 2],
                strides: vec![2, 3],
                offset: 6,
            },
        ],
        src_blocks: vec![Block::packed(vec![2, 3], 0), Block::packed(vec![2, 3], 6)],
        pairs: Vec::new(),
        groups: vec![Group {
            dst_blocks: vec![0, 1],
            src_blocks: vec![0, 1],
            axes: vec![1, 0],
            u: vec![1.0, 2.0, 0.5, -3.0],
        }],
        conjugate: false,
    }
}

/// The recoupling fixtures, for tests that are about the Multi path itself.
pub fn recoupling_fixtures() -> Vec<Fixture> {
    vec![
        recoupling_non_symmetric_u(),
        recoupling_rectangular(),
        recoupling_one_to_many(),
        recoupling_many_to_one(),
        equal_shape_recoupling_jobs(),
        mixed_single_and_multi(),
        conjugated_recoupling(),
    ]
}

/// Every fixture the device and host suites both replay.
pub fn all_fixtures() -> Vec<Fixture> {
    let mut fixtures = rank_sweep();
    fixtures.push(interleaved_multi_block());
    fixtures.push(inactive_destination_layouts());
    fixtures.push(padded_inactive_destination_layouts());
    fixtures.push(rank_zero_inactive_destination());
    fixtures.push(zero_extent_block());
    fixtures.extend(recoupling_fixtures());
    fixtures
}
