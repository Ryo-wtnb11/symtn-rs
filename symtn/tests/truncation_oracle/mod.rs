//! Independent expectation for the truncated-factorization compositions
//! (`svd_compact` / `eigh_full` / `eig_full` → `diagview` → `find_truncated`
//! → `restrict_leg`, #1300, #1534).
//!
//! Nothing here calls a symtn factorization or a symtn truncation decision:
//!
//! * [`sector_matrices!`] reads each coupled sector's reduced matrix (TensorKit
//!   `block(t, c)`) from the tensor's public blocks. Rows are grouped by
//!   codomain tree and columns by domain tree, both in the sorted
//!   order of their labels, and a block's multi-index is linearized
//!   first-axis-fastest. Singular values do not depend on the arrangement; for
//!   an endomorphism rows and columns share it, so the matrix is the map in
//!   one basis and its eigenvalues are the map's.
//! * [`singular_values`] is a one-sided (Hestenes) Jacobi SVD written here;
//!   for a Hermitian block it gives `|lambda|`. [`triangular_eigenvalues`] is
//!   the hand answer for the triangular fixtures the general
//!   eigendecomposition is tested on.
//! * Factors are checked through gauge-free relations (`t * vh^H = u * s`,
//!   `u^H u = 1`, `t * v = v * d`, ...), so an exactly degenerate spectrum,
//!   whose kept basis is a free choice, is covered by the same assertions.
//! * [`select`] is a hand implementation of each policy's documented rule
//!   (`symtn::typed::Truncation`, TensorKit's `findtruncated`), written over
//!   a flat sorted candidate list rather than `select_truncation`'s
//!   per-sector sorts and heap; it returns kept positions per sector.

#![allow(dead_code)]

use num_complex::Complex64;
use symtn::sector::{ProductSector, SU2Irrep, U1Irrep, Z2Irrep};
use symtn::typed::{SectorSpectrum, SpectrumMagnitude};

/// Closed-form quantum dimension of each multiplicity-free sector type these
/// tests use, so no weight is read back from symtn: 1 for the abelian labels,
/// `2j + 1` for SU(2), the product of the parts for a product sector.
pub trait ClosedFormDim {
    fn closed_form_dim(&self) -> f64;
}

impl ClosedFormDim for U1Irrep {
    fn closed_form_dim(&self) -> f64 {
        1.0
    }
}

impl ClosedFormDim for Z2Irrep {
    fn closed_form_dim(&self) -> f64 {
        1.0
    }
}

impl ClosedFormDim for SU2Irrep {
    fn closed_form_dim(&self) -> f64 {
        (self.twice_spin() + 1) as f64
    }
}

impl<L: ClosedFormDim, R: ClosedFormDim> ClosedFormDim for ProductSector<L, R> {
    fn closed_form_dim(&self) -> f64 {
        self.left().closed_form_dim() * self.right().closed_form_dim()
    }
}

/// A dense column-major complex matrix.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Complex64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
        }
    }

    pub fn at(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    fn col(&self, col: usize) -> &[Complex64] {
        &self.data[col * self.rows..(col + 1) * self.rows]
    }
}

/// One stored block, with its values listed first-axis-fastest over `shape`.
pub struct BlockEntry<S, K> {
    pub coupled: S,
    pub row: K,
    pub col: K,
    pub shape: Vec<usize>,
    pub values: Vec<Complex64>,
}

/// Every multi-index of `shape`, first axis fastest.
pub fn multi_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|mut flat| {
            shape
                .iter()
                .map(|&extent| {
                    let index = flat % extent;
                    flat /= extent;
                    index
                })
                .collect()
        })
        .collect()
}

/// Stacks the blocks of each coupled sector into its reduced matrix.
pub fn assemble<S: PartialEq + Clone, K: Ord + Clone>(
    codomain_rank: usize,
    blocks: Vec<BlockEntry<S, K>>,
) -> Vec<(S, Matrix)> {
    // Per sector: the row and column groups with their offsets and sizes.
    struct Groups<S, K> {
        sector: S,
        rows: Vec<(K, usize, usize)>,
        cols: Vec<(K, usize, usize)>,
    }
    fn place<K: PartialEq + Clone>(groups: &mut Vec<(K, usize, usize)>, key: &K, size: usize) {
        if !groups.iter().any(|(k, _, _)| k == key) {
            groups.push((key.clone(), 0, size));
        }
    }
    // Sorted by tree key, so the rows and columns of an endomorphism come in
    // the same order and the matrix diagonal is the map's diagonal.
    fn lay_out<K: Ord>(groups: &mut [(K, usize, usize)]) {
        groups.sort_by(|a, b| a.0.cmp(&b.0));
        let mut offset = 0;
        for group in groups.iter_mut() {
            group.1 = offset;
            offset += group.2;
        }
    }
    let mut sectors: Vec<Groups<S, K>> = Vec::new();
    for block in &blocks {
        let rows: usize = block.shape[..codomain_rank].iter().product();
        let cols: usize = block.shape[codomain_rank..].iter().product();
        let index = match sectors.iter().position(|g| g.sector == block.coupled) {
            Some(index) => index,
            None => {
                sectors.push(Groups {
                    sector: block.coupled.clone(),
                    rows: Vec::new(),
                    cols: Vec::new(),
                });
                sectors.len() - 1
            }
        };
        place(&mut sectors[index].rows, &block.row, rows);
        place(&mut sectors[index].cols, &block.col, cols);
    }
    for g in &mut sectors {
        lay_out(&mut g.rows);
        lay_out(&mut g.cols);
    }
    let mut matrices: Vec<(S, Matrix)> = sectors
        .iter()
        .map(|g| {
            let rows = g.rows.iter().map(|(_, _, s)| s).sum();
            let cols = g.cols.iter().map(|(_, _, s)| s).sum();
            (g.sector.clone(), Matrix::zeros(rows, cols))
        })
        .collect();
    for block in blocks {
        let index = sectors
            .iter()
            .position(|g| g.sector == block.coupled)
            .unwrap();
        let g = &sectors[index];
        let &(_, row0, block_rows) = g.rows.iter().find(|(k, _, _)| *k == block.row).unwrap();
        let &(_, col0, _) = g.cols.iter().find(|(k, _, _)| *k == block.col).unwrap();
        let matrix = &mut matrices[index].1;
        for (flat, value) in block.values.into_iter().enumerate() {
            let (row, col) = (flat % block_rows.max(1), flat / block_rows.max(1));
            let rows = matrix.rows;
            matrix.data[(row0 + row) + (col0 + col) * rows] = value;
        }
    }
    matrices
}

/// Each coupled sector's reduced matrix of a tensor, as `Vec<(sector,
/// Matrix)>`. The including crate declares `numerics` at its root.
#[macro_export]
macro_rules! sector_matrices {
    ($tensor:expr) => {{
        // The oracle reads values, so a lazy or compact input is
        // materialized explicitly.
        let tensor = $tensor.materialize().unwrap();
        let blocks = tensor
            .subblocks()
            .unwrap()
            .map(|(trees, view)| {
                let shape = view.shape().to_vec();
                let values = $crate::truncation_oracle::multi_indices(&shape)
                    .iter()
                    .map(|index| $crate::numerics::Numeric::wide(*view.get(index).unwrap()))
                    .collect();
                $crate::truncation_oracle::BlockEntry {
                    coupled: trees.coupled().clone(),
                    row: (
                        trees.codomain_uncoupled().to_vec(),
                        trees.codomain_innerlines().to_vec(),
                        trees.codomain_vertices().to_vec(),
                    ),
                    col: (
                        trees.domain_uncoupled().to_vec(),
                        trees.domain_innerlines().to_vec(),
                        trees.domain_vertices().to_vec(),
                    ),
                    shape,
                    values,
                }
            })
            .collect();
        $crate::truncation_oracle::assemble(tensor.codomain_rank(), blocks)
    }};
}

/// Singular values of `a`, descending, by one-sided (Hestenes) Jacobi: rotate
/// column pairs until every pair is orthogonal; the column norms are then the
/// singular values. For a Hermitian `a` they are `|lambda|`.
pub fn singular_values(a: &Matrix) -> Vec<f64> {
    let (m, n) = (a.rows, a.cols);
    let mut u = a.clone();
    for _sweep in 0..100 {
        let mut rotated = false;
        for i in 0..n {
            for j in i + 1..n {
                let (ci, cj) = (u.col(i), u.col(j));
                let alpha: f64 = ci.iter().map(|z| z.norm_sqr()).sum();
                let beta: f64 = cj.iter().map(|z| z.norm_sqr()).sum();
                let gamma: Complex64 = ci.iter().zip(cj).map(|(x, y)| x.conj() * y).sum();
                let g = gamma.norm();
                if g == 0.0 || g <= 4.0 * f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                // Rephase column j so the pair's inner product is real, then
                // apply the real Jacobi rotation that zeroes it.
                let w = (gamma / g).conj();
                let zeta = (beta - alpha) / (2.0 * g);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for k in 0..m {
                    let xi = u.data[k + i * m];
                    let xj = u.data[k + j * m] * w;
                    u.data[k + i * m] = xi * c - xj * s;
                    u.data[k + j * m] = xi * s + xj * c;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut values: Vec<f64> = (0..n)
        .map(|k| u.col(k).iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt())
        .collect();
    values.sort_by(|x, y| y.total_cmp(x));
    // A wide matrix has at most `m` nonzero singular values.
    values.truncate(m.min(n));
    values
}

/// The eigenvalues of a triangular matrix are its diagonal. Panics unless `a`
/// is exactly upper or lower triangular, so a fixture cannot drift out of the
/// case this answer covers. Listed in symtn's published `eig` order,
/// ascending lexicographic `(re, im)` (#1985), sorted here independently.
pub fn triangular_eigenvalues(a: &Matrix) -> Vec<Complex64> {
    assert_eq!(a.rows, a.cols, "an endomorphism block");
    let zero = |strict_lower: bool| {
        (0..a.rows).all(|row| {
            (0..a.cols)
                .filter(|&col| if strict_lower { row > col } else { row < col })
                .all(|col| a.at(row, col) == Complex64::new(0.0, 0.0))
        })
    };
    assert!(
        zero(true) || zero(false),
        "the eig fixtures must be triangular"
    );
    let mut values: Vec<Complex64> = (0..a.rows).map(|k| a.at(k, k)).collect();
    values.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
    values
}

/// The reference values at the hand selection's kept positions of `sector`.
pub fn kept_values<S: PartialEq, V: Copy>(
    offers: &[Offer<S>],
    kept: &[Vec<usize>],
    sector: &S,
    reference: &[V],
) -> Vec<V> {
    let (_, positions) = offers
        .iter()
        .zip(kept)
        .find(|(offer, _)| offer.sector == *sector)
        .expect("a kept sector is an offered sector");
    positions.iter().map(|&p| reference[p]).collect()
}

/// The kept spectrum of a restricted factor has, sector by sector, the
/// offered magnitudes at the hand selection's kept positions, within the
/// tolerance of its own dtype. The offers are descending; the kept values are
/// compared as a multiset of magnitudes, since an `eigh` spectrum is stored
/// ascending (#1985).
#[track_caller]
pub fn assert_kept_magnitudes<S, V>(
    what: &str,
    got: &[SectorSpectrum<S, V>],
    offers: &[Offer<S>],
    kept: &[Vec<usize>],
    terms: usize,
) where
    S: PartialEq + std::fmt::Debug,
    V: SpectrumMagnitude + crate::numerics::Numeric,
{
    for entry in got {
        let offer = offers
            .iter()
            .find(|offer| offer.sector == entry.sector)
            .unwrap_or_else(|| panic!("{what}: {:?} was not offered", entry.sector));
        let want = kept_values(offers, kept, &entry.sector, &offer.magnitudes);
        assert_eq!(
            entry.values.len(),
            want.len(),
            "{what}: {:?} kept count",
            entry.sector
        );
        // Scaled by the sector's largest value, not the largest kept one: a
        // policy keeping the smallest values sees the block's absolute error.
        let bound = crate::numerics::tolerance::<V>(
            terms,
            offer.magnitudes.iter().copied().fold(0.0, f64::max),
        );
        let mut got: Vec<f64> = entry
            .values
            .iter()
            .map(|&value| SpectrumMagnitude::magnitude(value))
            .collect();
        got.sort_by(|x, y| y.total_cmp(x));
        for (index, (&got, &want)) in got.iter().zip(&want).enumerate() {
            assert!(
                (got - want).abs() <= bound,
                "{what}: {:?} value {index} is {got} against the oracle {want} (bound {bound:e})",
                entry.sector
            );
        }
    }
}

/// The truncation error against the oracle's discarded norm, at the tolerance
/// of the payload dtype the spectrum was computed in (`payload` only names it).
#[track_caller]
pub fn assert_error_close<T: crate::numerics::Numeric>(
    what: &str,
    _payload: &[T],
    got: f64,
    want: f64,
    terms: usize,
) {
    let bound = crate::numerics::tolerance::<T>(terms, want);
    assert!(
        (got - want).abs() <= bound,
        "{what}: truncation error {got} against the oracle {want} (bound {bound:e})"
    );
}

/// One coupled sector's offered spectrum: quantum dimension and magnitudes,
/// in the order the fixture lists them.
#[derive(Clone, Debug)]
pub struct Offer<S> {
    pub sector: S,
    pub dim: f64,
    pub magnitudes: Vec<f64>,
}

/// A truncation policy, restated for the hand selection.
#[derive(Clone, Debug)]
pub enum Policy<S> {
    Full,
    Rank(usize),
    /// Keep `v >= rtol * sqrt(sum dim v^2)`.
    RelativeCutoff(f64),
    /// Keep `v >= rtol * max v`.
    RelativeInfCutoff(f64),
    /// Discard ascending while `sum dim v^2 <= (rtol * norm)^2`.
    RelativeError(f64),
    /// Keep `v >= atol`.
    AbsoluteCutoff(f64),
    /// Keep `v <= max(atol, rtol * sqrt(sum dim v^2))`.
    Below {
        atol: f64,
        rtol: f64,
    },
    /// Keep the smallest values while `sum dim <= rank`.
    RankSmallest(usize),
    /// A fixed per-sector prefix (absent sector: zero).
    Space(Vec<(S, usize)>),
    /// Both components must keep a value.
    And(Box<Policy<S>>, Box<Policy<S>>),
}

fn weighted_norm<S>(offers: &[Offer<S>]) -> f64 {
    offers
        .iter()
        .map(|o| o.dim * o.magnitudes.iter().map(|v| v * v).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// Kept positions per offer, ascending, over the offer's magnitudes in the
/// order they are listed (TensorKit `findtruncated` over a `SectorVector`).
/// The flat candidate list is sorted with the whole key spelled out: keep
/// order `(v desc, sector, position asc)`, discard order
/// `(v asc, sector, position asc)`. An exact cross-sector tie goes to the
/// sector that sorts first under `S: Ord`, which is TensorKit's order for
/// every fixture that has a tie (non-negative U(1) charges, SU(2) spins).
pub fn select<S: Ord + Clone>(offers: &[Offer<S>], policy: &Policy<S>) -> Vec<Vec<usize>> {
    let all = |o: &Offer<S>| (0..o.magnitudes.len()).collect::<Vec<usize>>();
    let candidates = || -> Vec<(f64, &S, usize, usize)> {
        offers
            .iter()
            .enumerate()
            .flat_map(|(i, o)| {
                o.magnitudes
                    .iter()
                    .enumerate()
                    .map(move |(p, &v)| (v, &o.sector, p, i))
            })
            .collect()
    };
    let filter = |threshold: f64| -> Vec<Vec<usize>> {
        offers
            .iter()
            .map(|o| {
                (0..o.magnitudes.len())
                    .filter(|&p| o.magnitudes[p] >= threshold)
                    .collect()
            })
            .collect()
    };
    match policy {
        Policy::Full => offers.iter().map(all).collect(),
        Policy::Rank(rank) | Policy::RankSmallest(rank) => {
            let largest = matches!(policy, Policy::Rank(_));
            let mut candidates = candidates();
            candidates.sort_by(|a, b| {
                let by_value = if largest {
                    b.0.total_cmp(&a.0)
                } else {
                    a.0.total_cmp(&b.0)
                };
                by_value.then_with(|| a.1.cmp(b.1)).then(a.2.cmp(&b.2))
            });
            let mut kept = vec![Vec::new(); offers.len()];
            let mut used = 0.0;
            for (_, _, p, i) in candidates {
                if used + offers[i].dim > *rank as f64 {
                    break;
                }
                used += offers[i].dim;
                kept[i].push(p);
            }
            for positions in &mut kept {
                positions.sort_unstable();
            }
            kept
        }
        Policy::RelativeCutoff(rtol) => filter(rtol * weighted_norm(offers)),
        Policy::AbsoluteCutoff(atol) => filter(*atol),
        Policy::Below { atol, rtol } => {
            let threshold = atol.max(rtol * weighted_norm(offers));
            offers
                .iter()
                .map(|o| {
                    (0..o.magnitudes.len())
                        .filter(|&p| o.magnitudes[p] <= threshold)
                        .collect()
                })
                .collect()
        }
        Policy::RelativeInfCutoff(rtol) => {
            let max = offers
                .iter()
                .flat_map(|o| o.magnitudes.iter().copied())
                .fold(0.0, f64::max);
            filter(rtol * max)
        }
        Policy::RelativeError(rtol) => {
            let bound = rtol * weighted_norm(offers);
            let values: usize = offers.iter().map(|o| o.magnitudes.len()).sum();
            let limit = bound * bound * (1.0 + (values + 5) as f64 * f64::EPSILON);
            let mut candidates = candidates();
            candidates.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.1.cmp(b.1))
                    .then(a.2.cmp(&b.2))
            });
            let mut discarded = vec![Vec::new(); offers.len()];
            let mut total = 0.0;
            for (v, _, p, i) in candidates {
                let next = total + offers[i].dim * v * v;
                if next > limit {
                    break;
                }
                total = next;
                discarded[i].push(p);
            }
            offers
                .iter()
                .zip(&discarded)
                .map(|(o, gone)| all(o).into_iter().filter(|p| !gone.contains(p)).collect())
                .collect()
        }
        Policy::Space(ranks) => offers
            .iter()
            .map(|o| {
                let rank = ranks
                    .iter()
                    .find(|(s, _)| *s == o.sector)
                    .map_or(0, |&(_, r)| r);
                let mut order = all(o);
                order.sort_by(|&a, &b| o.magnitudes[b].total_cmp(&o.magnitudes[a]).then(a.cmp(&b)));
                order.truncate(rank);
                order.sort_unstable();
                order
            })
            .collect(),
        Policy::And(left, right) => select(offers, left)
            .into_iter()
            .zip(select(offers, right))
            .map(|(a, b)| a.into_iter().filter(|p| b.contains(p)).collect())
            .collect(),
    }
}

/// `sqrt(sum_c dim(c) sum_{discarded} v^2)`.
pub fn discarded_norm<S>(offers: &[Offer<S>], kept: &[Vec<usize>]) -> f64 {
    offers
        .iter()
        .zip(kept)
        .map(|(o, k)| {
            o.dim
                * o.magnitudes
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| !k.contains(p))
                    .map(|(_, v)| v * v)
                    .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
}

/// The expected kept bond as `(sector, count)` pairs sorted by sector, zero
/// counts dropped.
pub fn kept_pairs<S: Ord + Clone>(offers: &[Offer<S>], kept: &[Vec<usize>]) -> Vec<(S, usize)> {
    let mut pairs: Vec<(S, usize)> = offers
        .iter()
        .zip(kept)
        .filter(|(_, k)| !k.is_empty())
        .map(|(o, k)| (o.sector.clone(), k.len()))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

/// `Vec<Offer>` of a tensor's per-sector singular values; `$dim` maps a sector
/// to its quantum dimension.
#[macro_export]
macro_rules! singular_offers {
    ($tensor:expr, $dim:expr) => {{
        sector_matrices!($tensor)
            .iter()
            .map(|(sector, matrix)| $crate::truncation_oracle::Offer {
                sector: sector.clone(),
                dim: $dim(sector),
                magnitudes: $crate::truncation_oracle::singular_values(matrix),
            })
            .collect::<Vec<_>>()
    }};
}

/// The policy sweep every composition case runs: `(name, Truncation, Policy)`.
/// `$target` is the fixture's `truncspace` target.
#[macro_export]
macro_rules! policies {
    ($target:expr) => {{
        use symtn::typed::Truncation;
        use $crate::truncation_oracle::Policy;
        let target = &$target;
        let pairs: Vec<_> = target
            .sectors()
            .unwrap()
            .into_iter()
            .zip(target.degeneracies().iter().copied())
            .collect();
        vec![
            ("Full", Truncation::Full, Policy::Full),
            ("Rank(huge)", Truncation::rank(4096), Policy::Rank(4096)),
            ("Rank(4)", Truncation::rank(4), Policy::Rank(4)),
            ("Rank(3)", Truncation::rank(3), Policy::Rank(3)),
            ("Rank(1)", Truncation::rank(1), Policy::Rank(1)),
            ("Rank(0)", Truncation::rank(0), Policy::Rank(0)),
            (
                "Tolerance",
                Truncation::relative_cutoff(0.25).unwrap(),
                Policy::RelativeCutoff(0.25),
            ),
            (
                "ToleranceInf",
                Truncation::relative_inf_cutoff(0.4).unwrap(),
                Policy::RelativeInfCutoff(0.4),
            ),
            (
                "AbsoluteCutoff",
                Truncation::absolute_cutoff(0.5).unwrap(),
                Policy::AbsoluteCutoff(0.5),
            ),
            (
                "Below(atol)",
                Truncation::below(0.5, 0.0).unwrap(),
                Policy::Below {
                    atol: 0.5,
                    rtol: 0.0,
                },
            ),
            (
                "Below(rtol)",
                Truncation::below(0.0, 0.25).unwrap(),
                Policy::Below {
                    atol: 0.0,
                    rtol: 0.25,
                },
            ),
            (
                "RankSmallest(3)",
                Truncation::rank_smallest(3),
                Policy::RankSmallest(3),
            ),
            (
                "DiscardWeight",
                Truncation::relative_error(0.2).unwrap(),
                Policy::RelativeError(0.2),
            ),
            (
                "Space",
                Truncation::space(target.truncspace()),
                Policy::Space(pairs.clone()),
            ),
            (
                "Space & Rank",
                Truncation::space(target.truncspace()).and(Truncation::rank(2)),
                Policy::And(Box::new(Policy::Space(pairs)), Box::new(Policy::Rank(2))),
            ),
        ]
    }};
}

/// The kept bond leg is exactly the hand selection: same `(sector, count)`
/// pairs, not dual.
#[macro_export]
macro_rules! assert_kept_bond {
    ($bond:expr, $offers:expr, $kept:expr, $case:expr) => {{
        let bond = &$bond;
        let want = $crate::truncation_oracle::kept_pairs(&$offers, &$kept);
        let mut got: Vec<_> = bond
            .sectors()
            .unwrap()
            .into_iter()
            .zip(bond.degeneracies().iter().copied())
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, want, "{}: kept bond", $case);
        assert!(!bond.is_dual(), "{}: the bond is not dual", $case);
    }};
}

/// A factor's block geometry is the canonical layout of its own homspace,
/// i.e. that of a fresh `zeros` on the same spaces.
#[macro_export]
macro_rules! assert_canonical_layout {
    ($tensor:expr, $what:expr) => {{
        let got = &$tensor;
        let want: symtn::typed::TensorMap<_, f64> = symtn::typed::TensorMap::zeros(
            got.runtime(),
            got.codomain().iter(),
            got.domain().iter(),
        )
        .unwrap();
        assert_eq!(
            got.subblock_count(),
            want.subblock_count(),
            "{} block count",
            $what
        );
        for index in 0..want.subblock_count() {
            let (left, right) = (got.subblock(index).unwrap(), want.subblock(index).unwrap());
            assert_eq!(left.key(), right.key(), "{} block {index} key", $what);
            assert_eq!(left.shape(), right.shape(), "{} block {index} shape", $what);
            assert_eq!(
                left.strides(),
                right.strides(),
                "{} block {index} strides",
                $what
            );
            assert_eq!(
                left.offset(),
                right.offset(),
                "{} block {index} offset",
                $what
            );
        }
    }};
}
