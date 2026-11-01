//! `TensorMap::block(&c)` / `blocks()` (TensorKit `block(t, c)` /
//! `blocks(t)`) against oracles that never read the coupled payload layout:
//!
//! - the label oracle: a tensor filled by `from_subblock_fn` with a function of
//!   its tree labels and local indices, assembled into sector matrices from
//!   the subblock labels and the legs' degeneracies;
//! - the physical dense expansion (U(1), SU(2)): `M ≅ ⊕_c B_c ⊗ 1_{dim c}`
//!   unitarily, so `tr((M†M)^k) = Σ_c dim(c) tr((B_c†B_c)^k)`;
//! - composition: `block(A∘B, c) = block(A, c) · block(B, c)`;
//! - the lazy adjoint `block(t', c) = block(t, c)'` over the parent's payload.
//!
//! The CUDA gate runs with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test integration coupled_blocks:: -- --ignored` on a CUDA host.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use num_complex::Complex64;
use symtn::expert::HostReadableStorage;
use symtn::sector::TypedSectorAdmission;
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::{BlockFusionTrees, CoupledBlock, CoupledBlockPayload};
use symtn::typed::{
    GradedSpace, MultiplicityIndex, Runtime, SectorSpectrum, TensorMap, TensorScalar,
};

use crate::legs::{su2_legs, u1_legs};

trait Val: TensorScalar + Copy + PartialEq + Debug {
    fn make(hash: u64) -> Self;
    fn c(self) -> Complex64;
}

impl Val for f64 {
    fn make(hash: u64) -> Self {
        ((hash % 2001) as f64 - 1000.0) / 257.0
    }
    fn c(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Val for Complex64 {
    fn make(hash: u64) -> Self {
        Complex64::new(
            f64::make(hash),
            f64::make(hash.rotate_left(29) ^ 0x9E37_79B9),
        )
    }
    fn c(self) -> Complex64 {
        self
    }
}

/// A value fixed by the tree labels and local index alone.
fn label_value<S: Debug, D: Val>(trees: &BlockFusionTrees<S>, index: &[usize]) -> D {
    let mut hasher = DefaultHasher::new();
    format!("{trees:?}").hash(&mut hasher);
    index.hash(&mut hasher);
    D::make(hasher.finish())
}

/// Column-major dense matrix.
#[derive(Clone, Debug)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
        }
    }
    fn at(&self, r: usize, c: usize) -> Complex64 {
        self.data[r + c * self.rows]
    }
    fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let mut out = Mat::zeros(self.rows, other.cols);
        for c in 0..other.cols {
            for k in 0..self.cols {
                let b = other.at(k, c);
                for r in 0..self.rows {
                    out.data[r + c * out.rows] += self.at(r, k) * b;
                }
            }
        }
        out
    }
    fn adjoint(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for c in 0..self.cols {
            for r in 0..self.rows {
                out.data[c + r * out.rows] = self.at(r, c).conj();
            }
        }
        out
    }
    fn trace(&self) -> Complex64 {
        (0..self.rows.min(self.cols)).map(|i| self.at(i, i)).sum()
    }
    fn gram_moments(&self) -> [Complex64; 3] {
        let g = self.adjoint().mul(self);
        let g2 = g.mul(&g);
        [g.trace(), g2.trace(), g2.mul(&g).trace()]
    }
}

fn read<R, D: Val, S: HostReadableStorage<D>>(block: &CoupledBlock<'_, R, D, S>) -> Mat {
    let mut out = Mat::zeros(block.rows(), block.cols());
    for c in 0..block.cols() {
        for r in 0..block.rows() {
            out.data[r + c * block.rows()] = block.get(r, c).unwrap().c();
        }
    }
    assert!(block.get(block.rows(), 0).is_none());
    assert!(block.get(0, block.cols()).is_none());
    out
}

fn assert_mat_close(what: &str, got: &Mat, want: &Mat) {
    assert_eq!(
        (got.rows, got.cols),
        (want.rows, want.cols),
        "{what}: shape"
    );
    let scale = want.data.iter().map(|z| z.norm()).fold(1.0, f64::max);
    for (g, w) in got.data.iter().zip(&want.data) {
        assert!(
            (g - w).norm() <= 1e-12 * scale * want.rows.max(1) as f64,
            "{what}: {g} vs {w}"
        );
    }
}

fn product(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().product()
}

/// A tree's row or column: its label key and its range in the matrix.
type Extents = Vec<(String, std::ops::Range<usize>)>;

/// One expected sector: label, matrix, row trees, column trees.
type Expected<S> = (S, Mat, Extents, Extents);

fn tree_key<S: Debug>(uncoupled: &[S], innerlines: &[S], vertices: &[MultiplicityIndex]) -> String {
    format!("{uncoupled:?}{innerlines:?}{vertices:?}")
}

/// Expected sector matrices of `t`, assembled only from its subblock labels,
/// its legs' degeneracies and `value`. Row trees are the codomain trees in
/// first-appearance order over `subblocks()`, column trees likewise; each tree
/// spans its degeneracy extent, column-major in its legs.
fn expected_blocks<R, D>(
    t: &TensorMap<R, D>,
    mut value: impl FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> Complex64,
) -> Vec<Expected<R::Sector>>
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let codomain = t.codomain();
    let domain = t.domain();
    let dims = |legs: &[GradedSpace<R>], sectors: &[R::Sector]| -> Vec<usize> {
        legs.iter()
            .zip(sectors)
            .map(|(leg, sector)| leg.degeneracy(sector).unwrap())
            .collect()
    };
    struct Sector<S> {
        label: S,
        rows: Vec<(String, Vec<usize>)>,
        cols: Vec<(String, Vec<usize>)>,
        pairs: Vec<(usize, usize, BlockFusionTrees<S>)>,
    }
    let mut sectors: Vec<Sector<R::Sector>> = Vec::new();
    for index in 0..t.subblock_count() {
        let trees = t.subblock_fusion_trees(index).unwrap();
        let row_key = tree_key(
            trees.codomain_uncoupled(),
            trees.codomain_innerlines(),
            trees.codomain_vertices(),
        );
        let col_key = tree_key(
            trees.domain_uncoupled(),
            trees.domain_innerlines(),
            trees.domain_vertices(),
        );
        let position = match sectors.iter().position(|s| &s.label == trees.coupled()) {
            Some(position) => position,
            None => {
                sectors.push(Sector {
                    label: trees.coupled().clone(),
                    rows: Vec::new(),
                    cols: Vec::new(),
                    pairs: Vec::new(),
                });
                sectors.len() - 1
            }
        };
        let sector = &mut sectors[position];
        let row = match sector.rows.iter().position(|(key, _)| *key == row_key) {
            Some(row) => row,
            None => {
                sector
                    .rows
                    .push((row_key, dims(&codomain, trees.codomain_uncoupled())));
                sector.rows.len() - 1
            }
        };
        let col = match sector.cols.iter().position(|(key, _)| *key == col_key) {
            Some(col) => col,
            None => {
                sector
                    .cols
                    .push((col_key, dims(&domain, trees.domain_uncoupled())));
                sector.cols.len() - 1
            }
        };
        sector.pairs.push((row, col, trees));
    }

    let offsets = |extents: &[(String, Vec<usize>)]| -> (Vec<usize>, usize) {
        let mut offsets = Vec::new();
        let mut total = 0;
        for (_, dims) in extents {
            offsets.push(total);
            total += product(dims.iter().copied());
        }
        (offsets, total)
    };
    let unravel = |mut linear: usize, dims: &[usize]| -> Vec<usize> {
        dims.iter()
            .map(|&n| {
                let i = linear % n;
                linear /= n;
                i
            })
            .collect()
    };
    sectors
        .into_iter()
        .map(|sector| {
            let (row_offsets, rows) = offsets(&sector.rows);
            let (col_offsets, cols) = offsets(&sector.cols);
            let mut matrix = Mat::zeros(rows, cols);
            for (row, col, trees) in &sector.pairs {
                let row_dims = &sector.rows[*row].1;
                let col_dims = &sector.cols[*col].1;
                for a in 0..product(row_dims.iter().copied()) {
                    for b in 0..product(col_dims.iter().copied()) {
                        let mut local = unravel(a, row_dims);
                        local.extend(unravel(b, col_dims));
                        let r = row_offsets[*row] + a;
                        let c = col_offsets[*col] + b;
                        matrix.data[r + c * rows] = value(trees, &local);
                    }
                }
            }
            let extents = |keys: &[(String, Vec<usize>)], offsets: &[usize]| -> Extents {
                keys.iter()
                    .zip(offsets)
                    .map(|((key, dims), &start)| {
                        (key.clone(), start..start + product(dims.iter().copied()))
                    })
                    .collect()
            };
            let rows = extents(&sector.rows, &row_offsets);
            let cols = extents(&sector.cols, &col_offsets);
            (sector.label, matrix, rows, cols)
        })
        .collect()
}

fn labelled_extents<R>(
    trees: Vec<(
        symtn::typed::FusionTreeLabels<R::Sector>,
        std::ops::Range<usize>,
    )>,
) -> Extents
where
    R: TypedSectorAdmission,
{
    trees
        .into_iter()
        .map(|(tree, range)| {
            (
                tree_key(tree.uncoupled(), tree.innerlines(), tree.vertices()),
                range,
            )
        })
        .collect()
}

/// `blocks()` and every `block(&c)` equal `expected` exactly, sector for
/// sector and in the same (storage) order.
fn assert_blocks_equal<R, D>(what: &str, t: &TensorMap<R, D>, expected: &[Expected<R::Sector>])
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let blocks: Vec<_> = t.blocks().unwrap().collect();
    assert_eq!(blocks.len(), expected.len(), "{what}: sector count");
    for ((sector, block), (want_sector, want, rows, cols)) in blocks.iter().zip(expected) {
        assert_eq!(sector, want_sector, "{what}: sector order");
        assert_eq!(
            &labelled_extents::<R>(block.row_trees().unwrap()),
            rows,
            "{what}: row trees of {sector:?}"
        );
        assert_eq!(
            &labelled_extents::<R>(block.col_trees().unwrap()),
            cols,
            "{what}: column trees of {sector:?}"
        );
        for (tree, _) in block
            .row_trees()
            .unwrap()
            .iter()
            .chain(&block.col_trees().unwrap())
        {
            assert_eq!(tree.coupled(), sector, "{what}: tree coupled sector");
        }
        let got = read(block);
        assert_eq!(got.rows, want.rows, "{what}: rows of {sector:?}");
        assert_eq!(got.cols, want.cols, "{what}: cols of {sector:?}");
        assert_eq!(got.data, want.data, "{what}: entries of {sector:?}");
        assert_eq!(
            read(&t.block(sector).unwrap()).data,
            want.data,
            "{what}: block({sector:?})"
        );
    }
}

/// The subblock values of `t` (pre-existing accessor) placed by labels.
fn subblock_value<'a, R, D>(
    t: &'a TensorMap<R, D>,
) -> impl FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> Complex64 + 'a
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    // A lazy adjoint has no borrowed subblocks; the oracle materializes it.
    let dense = t.materialize(engine).unwrap();
    move |trees, index| {
        let views: HashMap<_, _> = dense.subblocks().unwrap().collect();
        views[trees].get(index).unwrap().c()
    }
}

fn label_case<R, D>(
    what: &str,
    rt: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
) -> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>
        + symtn::typed::TypedTensorConstructionDispatch<R, D>,
    D: Val,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let t = TensorMap::<R, D>::from_subblock_fn(
        engine,
        rt,
        codomain.iter().copied(),
        domain.iter().copied(),
        label_value::<R::Sector, D>,
    )
    .unwrap();
    let expected = expected_blocks(&t, |trees, index| {
        label_value::<R::Sector, D>(trees, index).c()
    });
    assert!(
        expected.len() > 1,
        "{what}: fixture must have several coupled sectors"
    );
    assert_blocks_equal(what, &t, &expected);
    t
}

/// The lazy adjoint's blocks are `block(t, c)'` over the parent's own payload,
/// read without materializing; afterwards they also match the materialized
/// adjoint's subblocks.
fn adjoint_case<R, D>(what: &str, t: &TensorMap<R, D>)
where
    R: TypedSectorAdmission + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R> + symtn::typed::TypedAdjointSpace<R>,
    D: Val,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let lazy = t.adjoint(engine).unwrap();
    adjoint_view_case(what, t, &lazy);
}

fn adjoint_view_case<R, D>(what: &str, t: &TensorMap<R, D>, lazy: &TensorMap<R, D>)
where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let parent = t.dense_data().unwrap().as_ptr();
    for (sector, block) in lazy.blocks().unwrap() {
        let CoupledBlockPayload::Dense {
            storage,
            adjoint: true,
            ..
        } = block.payload()
        else {
            panic!("{what}: a lazy adjoint block must be a flagged parent view");
        };
        assert_eq!(
            storage.as_ptr(),
            parent,
            "{what}: the view borrows the parent payload"
        );
        let want = read(&t.block(&sector).unwrap()).adjoint();
        assert_eq!(
            read(&block).data,
            want.data,
            "{what}: block(t', {sector:?})"
        );
    }
    let expected = expected_blocks(lazy, subblock_value(lazy));
    assert_blocks_equal(&format!("{what} (materialized subblocks)"), lazy, &expected);
}

/// Returns whether some factor block was TensorKit's empty view.
fn compose_case<R, D>(what: &str, a: &TensorMap<R, D>, b: &TensorMap<R, D>) -> bool
where
    R: TypedSectorAdmission,
    R::Mode:
        symtn::typed::TypedTensorModeDispatch<R> + symtn::typed::TypedTensorContractDispatch<R, D>,
    D: Val,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let ab = a.compose(engine, b).unwrap();
    let mut count = 0;
    let mut empty_factor = false;
    for (sector, block) in ab.blocks().unwrap() {
        // A factor without a stored block gives TensorKit's empty `d₁ × 0` or
        // `0 × d₂` view, so the plain product is the zero block there.
        let (a_c, b_c) = (a.block(&sector).unwrap(), b.block(&sector).unwrap());
        empty_factor |= a_c.cols() == 0 || b_c.rows() == 0;
        let want = read(&a_c).mul(&read(&b_c));
        count += 1;
        assert_mat_close(
            &format!("{what}: block(A∘B, {sector:?})"),
            &read(&block),
            &want,
        );
    }
    assert!(count > 1, "{what}: several coupled sectors");
    empty_factor
}

macro_rules! physical_case {
    ($engine:ident, $what:expr, $t:expr, $dim:expr) => {{
        let t = $t;
        let physical = t.to_physical_dense($engine).unwrap();
        let rows: usize = physical.shape[..t.codomain_rank()].iter().product();
        let cols: usize = physical.shape[t.codomain_rank()..].iter().product();
        let m = Mat {
            rows,
            cols,
            data: physical.data.iter().map(|&x| Val::c(x)).collect(),
        };
        let mut want = [Complex64::new(0.0, 0.0); 3];
        for (sector, block) in t.blocks().unwrap() {
            let dim = ($dim)(&sector) as f64;
            for (total, moment) in want.iter_mut().zip(read(&block).gram_moments()) {
                *total += moment * dim;
            }
        }
        for (k, (got, want)) in m.gram_moments().into_iter().zip(want).enumerate() {
            assert!(
                (got - want).norm() <= 1e-10 * want.norm().max(1.0),
                "{}: tr((M†M)^{}) {got} vs Σ dim(c) tr((B†B)^{}) {want}",
                $what,
                k + 1,
                k + 1
            );
        }
    }};
}

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn fz2u1_legs() -> (GradedSpace<Fz2U1>, GradedSpace<Fz2U1>) {
    let sector = |q: i32| {
        let parity = if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        };
        ProductSector::new(parity, U1Irrep::new(q))
    };
    let v = GradedSpace::try_new(
        Arc::new(Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap()),
        [(-1, 2), (0, 1), (1, 2), (2, 1)].map(|(q, n)| (sector(q), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

/// Label, adjoint and composition oracles over `[v, w] ← [v]` and
/// `[v] ← [v, w]` (`w` dual), for both payload dtypes.
macro_rules! multiplicity_free {
    ($engine:ident, $rt:expr, $legs:expr, $name:expr, |$t:ident| $extra:expr) => {{
        let (v, w) = $legs;
        {
            let a = label_case::<_, f64>(concat!($name, " f64"), $rt, &[&v, &w], &[&v]);
            let b = label_case::<_, f64>(concat!($name, " f64 B"), $rt, &[&v], &[&v, &w]);
            adjoint_case(concat!($name, " f64 adjoint"), &a);
            assert!(compose_case(concat!($name, " f64 compose"), &a, &b));
            let $t = &a;
            $extra;
        }
        {
            let a = label_case::<_, Complex64>(concat!($name, " c64"), $rt, &[&v, &w], &[&v]);
            let b = label_case::<_, Complex64>(concat!($name, " c64 B"), $rt, &[&v], &[&v, &w]);
            adjoint_case(concat!($name, " c64 adjoint"), &a);
            assert!(compose_case(concat!($name, " c64 compose"), &a, &b));
            let $t = &a;
            $extra;
        }
    }};
}

#[test]
fn u1_blocks_match_label_physical_adjoint_and_composition_oracles() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    multiplicity_free!(engine, &rt, u1_legs(), "U(1)", |t| physical_case!(
        engine,
        "U(1)",
        t,
        |_: &U1Irrep| 1
    ));
}

#[test]
fn su2_blocks_match_label_physical_adjoint_and_composition_oracles() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    multiplicity_free!(engine, &rt, su2_legs(), "SU(2)", |t| physical_case!(
        engine,
        "SU(2)",
        t,
        |c: &SU2Irrep| c.twice_spin() + 1
    ));
}

#[test]
fn fz2u1_blocks_match_label_adjoint_and_composition_oracles() {
    let rt = Runtime::builder().build().unwrap();
    multiplicity_free!(engine, &rt, fz2u1_legs(), "fZ2xU(1)", |t| {
        let _ = t;
    });
}

#[test]
fn compact_diagonal_block_is_its_stored_spectrum() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    let (v, _) = u1_legs();
    let spectra = [
        (-1, vec![1.5, -2.0]),
        (0, vec![0.25]),
        (1, vec![3.0, -0.5, 4.0]),
    ];
    let d = TensorMap::<_, Complex64>::diagonal(
        engine,
        &rt,
        &v,
        spectra.iter().map(|(q, values)| SectorSpectrum {
            sector: U1Irrep::new(*q),
            values: values
                .iter()
                .map(|&x| Complex64::new(x, -x / 2.0))
                .collect(),
        }),
    )
    .unwrap();
    for (q, values) in &spectra {
        let block = d.block(&U1Irrep::new(*q)).unwrap();
        let CoupledBlockPayload::Diagonal(stored) = block.payload() else {
            panic!("a compact diagonal block must borrow its spectrum");
        };
        assert_eq!(stored.len(), values.len());
        assert_eq!((block.rows(), block.cols()), (values.len(), values.len()));
        for i in 0..values.len() {
            for j in 0..values.len() {
                let want = if i == j {
                    Complex64::new(values[i], -values[i] / 2.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_eq!(block.get(i, j), Some(want));
            }
        }
    }
    assert_eq!(d.blocks().unwrap().len(), spectra.len());
}

#[test]
fn absent_coupled_sector_is_tensorkits_empty_view() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let t =
        TensorMap::<_, f64>::from_subblock_fn(engine, &rt, [&v, &w], [&v], label_value).unwrap();
    let lazy = t.clone().adjoint(engine).unwrap();
    // `-2` fuses in the codomain `v ⊗ w'` (3·2 = 6 states) but not in the
    // domain `v`; `7` fuses nowhere.
    for (sector, rows, cols) in [(-2, 6, 0), (7, 0, 0)] {
        let block = t.block(&U1Irrep::new(sector)).unwrap();
        assert_eq!((block.rows(), block.cols()), (rows, cols), "{sector}");
        assert!(block.get(0, 0).is_none());
        assert!(block.row_trees().unwrap().is_empty());
        assert!(matches!(
            block.payload(),
            CoupledBlockPayload::Dense { adjoint: false, .. }
        ));
        let swapped = lazy.block(&U1Irrep::new(sector)).unwrap();
        assert_eq!((swapped.rows(), swapped.cols()), (cols, rows), "{sector}'");
        assert!(swapped.get(0, 0).is_none());
        assert!(matches!(
            swapped.payload(),
            CoupledBlockPayload::Dense { adjoint: true, .. }
        ));
    }
    assert!(t
        .blocks()
        .unwrap()
        .all(|(sector, _)| sector != U1Irrep::new(-2)));

    let d = TensorMap::<_, f64>::diagonal(
        engine,
        &rt,
        &v,
        [(-1, 2), (0, 1), (1, 3)].map(|(q, n)| SectorSpectrum {
            sector: U1Irrep::new(q),
            values: vec![1.0; n],
        }),
    )
    .unwrap();
    let block = d.block(&U1Irrep::new(5)).unwrap();
    assert_eq!((block.rows(), block.cols()), (0, 0));
    assert!(matches!(block.payload(), CoupledBlockPayload::Diagonal(values) if values.is_empty()));
}

#[cfg(feature = "cgc-gen")]
mod su3 {
    use symtn::typed::Engine;

    use super::*;
    use symtn::sector::LieGroupFusionRule;

    pub(super) fn legs() -> [GradedSpace<LieGroupFusionRule>; 3] {
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        let octet =
            GradedSpace::try_new(Arc::clone(&provider), [(vec![1i64, 1], 2), (vec![0, 0], 1)])
                .unwrap();
        let chiral = GradedSpace::try_new(provider, [(vec![1i64, 0], 2), (vec![0, 1], 1)]).unwrap();
        let dual = chiral.try_dual().unwrap();
        [octet, chiral, dual]
    }

    fn run<D: Val>(engine: &mut dyn Engine, what: &str) {
        let rt = Runtime::builder().build().unwrap();
        let [octet, chiral, dual] = legs();
        let a = label_case::<_, D>(what, &rt, &[&octet, &octet], &[&octet]);
        assert!(
            (0..a.subblock_count()).any(|i| a
                .subblock_fusion_trees(i)
                .unwrap()
                .codomain_vertices()[0]
                .get()
                == 2),
            "{what}: fixture must carry outer multiplicity"
        );
        let b = label_case::<_, D>(what, &rt, &[&octet], &[&chiral, &dual]);
        compose_case(what, &a, &b);
        let wide = label_case::<_, D>(what, &rt, &[&octet], &[&octet, &octet]);
        assert!(compose_case(what, &a, &wide), "{what}: empty factor block");
        adjoint_view_case(what, &b, &b.adjoint(engine).unwrap());
        // Only a label the provider cannot encode is an error.
        assert!(
            matches!(
                a.block(&vec![1i64]),
                Err(symtn::typed::GenericTensorError::Structure(
                    symtn::typed::CheckedGenericStructureError::Provider(
                        symtn::sector::LieGroupFusionRuleError::LabelLength { found: 1, .. }
                    )
                ))
            ),
            "{what}: foreign label"
        );

        let values = |n: usize| (0..n).map(|i| D::make(i as u64 + 11)).collect::<Vec<_>>();
        let d = TensorMap::<_, D>::diagonal(
            engine,
            &rt,
            &octet,
            [
                SectorSpectrum {
                    sector: vec![1i64, 1],
                    values: values(2),
                },
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: values(1),
                },
            ],
        )
        .unwrap();
        for (sector, n) in [(vec![1i64, 1], 2), (vec![0, 0], 1)] {
            let block = d.block(&sector).unwrap();
            let CoupledBlockPayload::Diagonal(stored) = block.payload() else {
                panic!("{what}: a compact diagonal block must borrow its spectrum");
            };
            assert_eq!(stored, values(n).as_slice(), "{what}: diagonal {sector:?}");
            assert_eq!(
                block.get(0, n - 1).unwrap().c(),
                if n == 1 {
                    values(1)[0].c()
                } else {
                    Complex64::new(0.0, 0.0)
                }
            );
        }
    }

    #[test]
    fn checked_generic_su3_blocks_match_label_adjoint_and_composition_oracles() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        run::<f64>(engine, "SU(3) f64");
        run::<Complex64>(engine, "SU(3) c64");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use symtn::typed::CudaStorage;

    /// The device view has the Host view's geometry, and the device region it
    /// names holds the Host block's entries. The device buffer is read back
    /// verbatim through `to_host`, which downloads the canonical payload (the
    /// parent's, for a lazy adjoint) without relayout.
    fn device_case<R, D>(cuda: &mut dyn symtn::typed::Engine, what: &str, host: &TensorMap<R, D>)
    where
        R: TypedSectorAdmission,
        R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
        D: Val + symtn::typed::CudaPayload + symtn::typed::CudaScalar,
    {
        let device: TensorMap<R, D, CudaStorage<D>> = host.to_cuda(cuda).unwrap();
        let back = device.to_host(cuda).unwrap();
        let host_blocks: Vec<_> = host.blocks().unwrap().collect();
        let device_blocks: Vec<_> = device.blocks().unwrap().collect();
        assert_eq!(host_blocks.len(), device_blocks.len(), "{what}");
        for ((sector, h), (device_sector, d)) in host_blocks.iter().zip(&device_blocks) {
            assert_eq!(sector, device_sector, "{what}");
            assert_eq!((h.rows(), h.cols()), (d.rows(), d.cols()), "{what}");
            let (
                CoupledBlockPayload::Dense {
                    offset: host_offset,
                    adjoint: host_adjoint,
                    ..
                },
                CoupledBlockPayload::Dense {
                    storage,
                    offset,
                    adjoint,
                },
            ) = (h.payload(), d.payload())
            else {
                panic!("{what}: dense views expected");
            };
            assert_eq!((host_offset, host_adjoint), (offset, adjoint), "{what}");
            let CoupledBlockPayload::Dense {
                storage: downloaded,
                ..
            } = back.block(sector).unwrap().payload()
            else {
                panic!("{what}: dense download expected");
            };
            assert_eq!(
                downloaded.len(),
                symtn::expert::TensorStorage::len(storage),
                "{what}"
            );
            let data = downloaded.as_slice();
            let want = read(h);
            for c in 0..d.cols() {
                for r in 0..d.rows() {
                    let value = if adjoint {
                        data[offset + c + r * d.cols()].c().conj()
                    } else {
                        data[offset + r + c * d.rows()].c()
                    };
                    assert_eq!(value, want.at(r, c), "{what}: ({r}, {c}) of {sector:?}");
                }
            }
        }
    }

    fn run<R, D>(
        engine: &mut dyn symtn::typed::Engine,
        cuda: &mut dyn symtn::typed::Engine,
        what: &str,
        rt: &Runtime,
        v: &GradedSpace<R>,
        w: &GradedSpace<R>,
    ) where
        R: TypedSectorAdmission + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>,
        R::Mode: symtn::typed::TypedTensorModeDispatch<R>
            + symtn::typed::TypedTensorConstructionDispatch<R, D>
            + symtn::typed::TypedAdjointSpace<R>,
        D: Val + symtn::typed::CudaPayload + symtn::typed::CudaScalar,
    {
        let t = label_case::<R, D>(what, rt, &[v, w], &[v]);
        device_case(cuda, what, &t);
        let adjoint = t.adjoint(engine).unwrap();
        device_case(cuda, &format!("{what} adjoint"), &adjoint);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn device_block_views_address_the_host_block_entries() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let cuda = &mut crate::common::cuda_engine();
        let rt = Runtime::builder().build().unwrap();
        let (v, w) = u1_legs();
        run::<_, f64>(engine, cuda, "U(1) f64", &rt, &v, &w);
        run::<_, Complex64>(engine, cuda, "U(1) c64", &rt, &v, &w);
        let (v, w) = su2_legs();
        run::<_, f64>(engine, cuda, "SU(2) f64", &rt, &v, &w);
        run::<_, Complex64>(engine, cuda, "SU(2) c64", &rt, &v, &w);
        let (v, w) = fz2u1_legs();
        run::<_, f64>(engine, cuda, "fZ2xU(1) f64", &rt, &v, &w);
        run::<_, Complex64>(engine, cuda, "fZ2xU(1) c64", &rt, &v, &w);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn checked_generic_su3_device_block_views_address_the_host_block_entries() {
        let rt = Runtime::builder().build().unwrap();
        let [octet, _, _] = super::su3::legs();
        let t = label_case::<_, f64>("SU(3) f64", &rt, &[&octet, &octet], &[&octet]);
        device_case("SU(3) f64", &t);
        device_case("SU(3) f64 adjoint", &t.adjoint().unwrap());
        let t = label_case::<_, Complex64>("SU(3) c64", &rt, &[&octet, &octet], &[&octet]);
        device_case("SU(3) c64", &t);
        device_case("SU(3) c64 adjoint", &t.adjoint().unwrap());
    }
}
