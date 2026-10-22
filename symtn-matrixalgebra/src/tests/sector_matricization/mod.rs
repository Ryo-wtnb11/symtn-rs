use super::*;
use crate::probe::{GenericPairPublicationProbe, PlacementIndexProbe, ScatterVisitProbe};
use std::cell::RefCell;
use symtn_core::{BlockSpec, FusionTreePairKey, Z2FusionRule};

#[derive(Clone, Copy)]
struct TestGenericRule;

impl symtn_core::testing::Provider for TestGenericRule {}

impl FusionRule for TestGenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> symtn_core::FusionStyleKind {
        symtn_core::FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> symtn_core::BraidingStyleKind {
        symtn_core::BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> symtn_core::SectorVec {
        match (left.id(), right.id()) {
            (0, sector) | (sector, 0) => [SectorId::new(sector)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => symtn_core::SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

fn generic_pair(coupled: usize, row_vertex: usize, col_vertex: usize) -> FusionTreePairKey {
    FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [1, 1],
        coupled,
        [false; 2],
        [false; 2],
        std::iter::empty::<usize>(),
        std::iter::empty::<usize>(),
        [row_vertex],
        [col_vertex],
    )
    .unwrap()
}

fn full_identity_pair(
    row_inner: usize,
    row_dual: bool,
    col_inner: usize,
    col_dual: bool,
) -> FusionTreePairKey {
    FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2, 3],
        [4, 5, 6],
        9,
        [false, row_dual, false],
        [true, false, col_dual],
        [row_inner],
        [col_inner],
        [1, 2],
        [2, 1],
    )
    .unwrap()
}

fn z2_single_sector_matrix(
    rows: usize,
    cols: usize,
) -> (FusionTreeHomSpace, SectorMatricization<f64>) {
    let even = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, cols)], false)]),
    );
    let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
    (
        homspace,
        SectorMatricization {
            sector: even,
            rows,
            cols,
            row_trees: vec![(key.codomain_tree().clone(), 0, vec![rows])],
            col_trees: vec![(key.domain_tree().clone(), 0, vec![cols])],
            data: vec![0.0; rows * cols],
        },
    )
}

fn vertex_tree_factor_fixture(
    reverse: bool,
) -> (
    FusionTreeHomSpace,
    SectorMatricization<Complex64>,
    FactorPair<Complex64>,
) {
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(x, 1)], false),
            SectorLeg::new([(x, 3)], false),
        ]),
    );
    let keys = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap();
    let mut row_trees = Vec::new();
    let mut col_trees = Vec::new();
    for key in keys
        .iter()
        .filter(|key| coupled_of_generic(key.codomain_tree()) == x)
    {
        if !row_trees.contains(key.codomain_tree()) {
            row_trees.push(key.codomain_tree().clone());
        }
        if !col_trees.contains(key.domain_tree()) {
            col_trees.push(key.domain_tree().clone());
        }
    }
    assert_eq!(row_trees.len(), 2);
    assert_eq!(col_trees.len(), 2);
    assert_eq!(
        row_trees
            .iter()
            .map(|tree| tree.vertices()[0].get())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(
        col_trees
            .iter()
            .map(|tree| tree.vertices()[0].get())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    if reverse {
        row_trees.reverse();
        col_trees.reverse();
    }
    let row_trees = row_trees
        .into_iter()
        .enumerate()
        .map(|(index, tree)| (tree, 2 * index, vec![2, 1]))
        .collect();
    let col_trees = col_trees
        .into_iter()
        .enumerate()
        .map(|(index, tree)| (tree, 3 * index, vec![1, 3]))
        .collect();
    let left = (0..8)
        .map(|index| Complex64::new(10.0 + index as f64, -1.0 - index as f64 / 4.0))
        .collect::<Vec<_>>();
    let right = (0..12)
        .map(|index| Complex64::new(30.0 + index as f64, 2.0 + index as f64 / 3.0))
        .collect::<Vec<_>>();
    (
        homspace,
        SectorMatricization {
            sector: x,
            rows: 4,
            cols: 6,
            row_trees,
            col_trees,
            data: vec![Complex64::new(0.0, 0.0); 24],
        },
        FactorPair {
            sector: x,
            kept: 2,
            left,
            left_rows: 4,
            right,
            right_leading: 2,
        },
    )
}

fn source_extent<D>(matrix: &SectorMatricization<D>, source_trees: FactorSide) -> usize {
    match source_trees {
        FactorSide::Left => matrix.rows,
        FactorSide::Right => matrix.cols,
    }
}

/// Stages `a x b` (left) or `b x a` (right) selected factors per sector
/// with `kept` deliberately unrelated to the admitted bond; the opposite
/// side carries a sentinel payload that must survive publication.
fn staged_one_sided_pairs<D: FactorScalar>(
    matrices: &[SectorMatricization<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    source_trees: FactorSide,
    values: &dyn Fn(usize) -> D,
) -> Vec<FactorPair<D>> {
    matrices
        .iter()
        .map(|matrix| {
            let a = source_extent(matrix, source_trees);
            let b = dimensions[&matrix.sector];
            let tag = matrix.sector.id() + 1;
            let selected = (0..a * b)
                .map(|k| values(1000 * tag + k))
                .collect::<Vec<_>>();
            let opposite = (0..3).map(|k| values(7 * tag + k)).collect();
            match side {
                FactorSide::Left => FactorPair {
                    sector: matrix.sector,
                    kept: 99,
                    left: selected,
                    left_rows: a,
                    right: opposite,
                    right_leading: 5,
                },
                FactorSide::Right => FactorPair {
                    sector: matrix.sector,
                    kept: 99,
                    left: opposite,
                    left_rows: 5,
                    right: selected,
                    right_leading: b,
                },
            }
        })
        .collect()
}

fn selected_of<D>(pair: &FactorPair<D>, side: FactorSide) -> &Vec<D> {
    match side {
        FactorSide::Left => &pair.left,
        FactorSide::Right => &pair.right,
    }
}

fn opposite_of<D>(pair: &FactorPair<D>, side: FactorSide) -> &Vec<D> {
    match side {
        FactorSide::Left => &pair.right,
        FactorSide::Right => &pair.left,
    }
}

/// Checks every output element against the literal coordinates
/// `F[o + q + a*j]` (left) / `F[j + b*(o + q)]` (right), independent of
/// the production layout proof. A sector without a matricization is
/// checked against a separately built `d x d` identity with `o` counted
/// over its earlier output blocks (the output tree order is its basis).
fn assert_literal_one_sided_layout<D>(
    structure: &BlockStructure,
    data: &[D],
    matrices: &[SectorMatricization<D>],
    selected: &[Vec<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    source_trees: FactorSide,
) where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    assert_literal_one_sided_blocks(
        structure,
        data,
        matrices,
        Some(selected),
        dimensions,
        side,
        source_trees,
    );
}

/// [`assert_literal_one_sided_layout`] that skips populated sectors when
/// `selected` is `None` (numerical factors), keeping the literal identity
/// check and the exact output coverage.
fn assert_literal_one_sided_blocks<D, M>(
    structure: &BlockStructure,
    data: &[D],
    matrices: &[SectorMatricization<M>],
    selected: Option<&[Vec<D>]>,
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    source_trees: FactorSide,
) where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let mut verified = 0usize;
    let mut identity_offsets = BTreeMap::<SectorId, usize>::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("factor blocks carry fusion-tree keys")
        };
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        let shape = block.shape();
        let strides = block.strides();
        let total = shape.iter().product::<usize>();
        let identity;
        let (o, a, b, factor) = if let Some((matrix_index, matrix)) = matrices
            .iter()
            .enumerate()
            .find(|(_, matrix)| matrix.sector == tree.coupled())
        {
            let Some(selected) = selected else {
                verified += total;
                continue;
            };
            let trees = match source_trees {
                FactorSide::Left => &matrix.row_trees,
                FactorSide::Right => &matrix.col_trees,
            };
            let (_, o, tree_shape) = trees
                .iter()
                .find(|(candidate, _, _)| candidate == tree)
                .unwrap();
            let b = dimensions[&matrix.sector];
            assert_eq!(total, tree_shape.iter().product::<usize>() * b);
            (
                *o,
                source_extent(matrix, source_trees),
                b,
                &selected[matrix_index],
            )
        } else {
            let d = dimensions[&tree.coupled()];
            let mut literal = vec![D::zero(); d * d];
            for diagonal in 0..d {
                literal[diagonal + d * diagonal] = D::one();
            }
            identity = literal;
            let o = identity_offsets.entry(tree.coupled()).or_default();
            let start = *o;
            *o += total / d;
            (start, d, d, &identity)
        };
        for flat in 0..total {
            let mut remaining = flat;
            let mut destination = block.offset();
            let mut coordinates = vec![0usize; shape.len()];
            for axis in 0..shape.len() {
                coordinates[axis] = remaining % shape[axis];
                remaining /= shape[axis];
                destination += coordinates[axis] * strides[axis];
            }
            let tree_axes = match side {
                FactorSide::Left => 0..shape.len() - 1,
                FactorSide::Right => 1..shape.len(),
            };
            let mut q = 0usize;
            let mut span = 1usize;
            for axis in tree_axes {
                q += coordinates[axis] * span;
                span *= shape[axis];
            }
            let j = match side {
                FactorSide::Left => coordinates[shape.len() - 1],
                FactorSide::Right => coordinates[0],
            };
            let expected = match side {
                FactorSide::Left => factor[o + q + a * j],
                FactorSide::Right => factor[j + b * (o + q)],
            };
            assert_eq!(data[destination], expected);
            verified += 1;
        }
    }
    assert_eq!(verified, data.len());
}

fn assert_buffers_intact<D>(pairs: &[FactorPair<D>], side: FactorSide) {
    assert!(pairs.iter().all(|pair| !selected_of(pair, side).is_empty()));
}

/// One even sector whose selected tree has zero legs: the factor block on
/// that side is rank-1 (`shape == [b]`). `padding` extra source rows or
/// columns precede the tree so the canonical proof declines and the
/// scatter fallback publishes the block.
fn zero_leg_side_matrix<D: FactorScalar>(
    zero_leg_side: FactorSide,
    extent: usize,
    padding: usize,
) -> (FusionTreeHomSpace, SectorMatricization<D>) {
    let even = SectorId::new(0);
    let leg = FusionProductSpace::new([SectorLeg::new([(even, extent)], false)]);
    let homspace = match zero_leg_side {
        FactorSide::Left => FusionTreeHomSpace::new(FusionProductSpace::new([]), leg),
        FactorSide::Right => FusionTreeHomSpace::new(leg, FusionProductSpace::new([])),
    };
    let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
    let (rows, cols, row_trees, col_trees) = match zero_leg_side {
        FactorSide::Left => (
            1 + padding,
            extent,
            vec![(key.codomain_tree().clone(), padding, vec![])],
            vec![(key.domain_tree().clone(), 0, vec![extent])],
        ),
        FactorSide::Right => (
            extent,
            1 + padding,
            vec![(key.codomain_tree().clone(), 0, vec![extent])],
            vec![(key.domain_tree().clone(), padding, vec![])],
        ),
    };
    (
        homspace,
        SectorMatricization {
            sector: even,
            rows,
            cols,
            row_trees,
            col_trees,
            data: vec![D::zero(); rows * cols],
        },
    )
}

/// Records every checked provider query in call order so that two
/// publication routes can be compared query by query.
struct RecordingGeneric {
    rule: TestGenericRule,
    log: RefCell<Vec<String>>,
    /// Its own identity, so that the sector-structure cache never answers a
    /// recorder's walk from another recorder's: each test counts queries.
    identity: symtn_core::RuleIdentity,
}

impl RecordingGeneric {
    fn new() -> Self {
        Self {
            rule: TestGenericRule,
            log: RefCell::new(Vec::new()),
            identity: symtn_core::RuleIdentity::new_unique::<Self>(),
        }
    }

    fn record<T>(&self, entry: String, value: T) -> Result<T, std::convert::Infallible> {
        self.log.borrow_mut().push(entry);
        Ok(value)
    }

    fn log(&self) -> Vec<String> {
        self.log.borrow().clone()
    }
}

impl symtn_core::testing::Provider for RecordingGeneric {}

impl CheckedGenericFusion for RecordingGeneric {
    type Error = std::convert::Infallible;

    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> symtn_core::FusionStyleKind {
        self.rule.fusion_style()
    }

    fn braiding_style(&self) -> symtn_core::BraidingStyleKind {
        self.rule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.record(format!("dual {sector:?}"), self.rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<symtn_core::SectorVec, Self::Error> {
        self.record(
            format!("channels {left:?} {right:?}"),
            self.rule.fusion_channels(left, right),
        )
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<symtn_core::SectorVec, Self::Error> {
        self.record(
            format!("channels_in_table {left:?} {right:?}"),
            self.rule.fusion_channels(left, right),
        )
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.record(
            format!("nsymbol {left:?} {right:?} {coupled:?}"),
            self.rule.nsymbol(left, right, coupled),
        )
    }
}

fn block_rows(structure: &BlockStructure) -> Vec<(BlockKey, Vec<usize>, Vec<usize>, usize)> {
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .map(|block| {
            (
                block.key().clone(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
        })
        .collect()
}

mod checked_one_sided;
mod compact_plan;
mod mf_one_sided;
mod paired_publication;
