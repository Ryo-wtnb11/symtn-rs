//! Fixtures and the independent oracle for the `EighFullPlan` gates
//! (#1499).
//!
//! The oracle reads tensors only through the public block API. Each coupled
//! sector's matrix is assembled from its fusion-tree blocks (rows and columns
//! keyed by tree, in key order), and its eigenvalues come from a textbook
//! cyclic Jacobi iteration, never from the solver under test. Eigenvectors
//! are compared through gauge-invariant quantities: the reconstruction
//! `A = V Λ Vᵀ`, `VᵀV = 1`, and the projector onto each group of equal
//! eigenvalues (grouped by value, not by column position).

use std::collections::BTreeMap;
use std::fmt::Debug;

use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use super::{block_matrix, members};

/// One coupled sector's matrix: `rows x cols`, column-major.
pub struct SectorMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl SectorMatrix {
    fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row + self.rows * col]
    }
}

/// Every coupled sector of `tensor` as one matrix, keyed by the sector's
/// `Debug` label. Row (column) trees are laid out in key order, so two
/// tensors with the same codomain trees get the same row order.
pub fn sector_matrices<R>(tensor: &TensorMap<R, f64>) -> BTreeMap<String, SectorMatrix>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
{
    type Keys = BTreeMap<String, usize>;
    let mut sectors: BTreeMap<String, (Keys, Keys, Vec<_>)> = BTreeMap::new();
    for index in 0..tensor.subblock_count() {
        let coupled = format!(
            "{:?}",
            tensor.subblock_fusion_trees(index).unwrap().coupled()
        );
        let block = block_matrix(tensor, index);
        let entry = sectors.entry(coupled).or_default();
        entry.0.insert(block.0.clone(), block.2);
        entry.1.insert(block.1.clone(), block.3);
        entry.2.push(block);
    }
    let offsets = |keys: &Keys| {
        let mut running = 0;
        let offsets: BTreeMap<String, usize> = keys
            .iter()
            .map(|(key, &extent)| {
                let offset = running;
                running += extent;
                (key.clone(), offset)
            })
            .collect();
        (offsets, running)
    };
    sectors
        .into_iter()
        .map(|(coupled, (row_keys, col_keys, blocks))| {
            let (row_at, rows) = offsets(&row_keys);
            let (col_at, cols) = offsets(&col_keys);
            let mut data = vec![0.0; rows * cols];
            for (row_key, col_key, block_rows, block_cols, matrix) in blocks {
                for col in 0..block_cols {
                    for row in 0..block_rows {
                        data[row_at[&row_key] + row + rows * (col_at[&col_key] + col)] =
                            matrix[row + block_rows * col];
                    }
                }
            }
            (coupled, SectorMatrix { rows, cols, data })
        })
        .collect()
}

/// Eigenvalues of a real symmetric matrix, ascending, by cyclic Jacobi
/// rotations until every off-diagonal entry is below `eps` of the norm.
pub fn jacobi_eigenvalues(matrix: &SectorMatrix) -> Vec<f64> {
    let n = matrix.rows;
    assert_eq!(n, matrix.cols);
    let mut a = matrix.data.clone();
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _sweep in 0..100 {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in p + 1..n {
                off = off.max(a[p + n * q].abs());
            }
        }
        if off <= f64::EPSILON * norm * 1e-2 || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p + n * q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q + n * q] - a[p + n * p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k + n * p], a[k + n * q]);
                    a[k + n * p] = c * akp - s * akq;
                    a[k + n * q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p + n * k], a[q + n * k]);
                    a[p + n * k] = c * apk - s * aqk;
                    a[q + n * k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut values: Vec<f64> = (0..n).map(|i| a[i + n * i]).collect();
    values.sort_by(f64::total_cmp);
    values
}

/// `count` Hermitian tensors `X + X†` over `codomain ← codomain`, with
/// dyadic entries (so the sum is exactly symmetric), distinct per member.
pub fn hermitian_members<R>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    count: usize,
    salt: usize,
) -> Vec<TensorMap<R, f64>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    members::<R, f64>(runtime, codomain, codomain, count, salt)
        .into_iter()
        .map(|x| x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap())
        .collect()
}

/// A single-leg endomorphism `leg ← leg` per member whose every coupled
/// sector matrix is `entry(member, i, j)` (symmetric by the caller).
pub fn single_leg<R>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    count: usize,
    entry: impl Fn(usize, usize, usize) -> f64,
) -> Vec<TensorMap<R, f64>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    (0..count)
        .map(|member| {
            TensorMap::from_subblock_fn(runtime, [leg], [leg], |_, index| {
                entry(member, index[0], index[1])
            })
            .unwrap()
        })
        .collect()
}

/// Checkerboard symmetric sector matrices (`a_ij = 0` for even `i + j`):
/// they anticommute with `diag((-1)^i)`, so the spectrum is symmetric about
/// zero and `|λ|` ties in ±λ pairs.
pub fn plus_minus_entry(member: usize, i: usize, j: usize) -> f64 {
    if (i + j).is_multiple_of(2) {
        0.0
    } else {
        let (lo, hi) = (i.min(j), i.max(j));
        ((lo * 7 + hi * 3 + member * 5) % 11) as f64 / 4.0 + 0.25
    }
}

/// `2 I + u uᵀ` sector matrices: eigenvalue 2 with multiplicity `n - 1`
/// and `2 + |u|²` once.
pub fn degenerate_entry(member: usize, i: usize, j: usize) -> f64 {
    let u = |k: usize| ((k + member) % 3) as f64 / 2.0 + 0.5;
    f64::from(u8::from(i == j)) * 2.0 + u(i) * u(j)
}

/// The per-member contract of an eigendecomposition `(d, v)` of `source`,
/// against the Jacobi oracle and against a reference decomposition of the
/// same member (eager), through gauge-invariant quantities only.
///
/// Tolerances follow `docs/testing_numerics.md` with `terms = n²` per
/// sector (a backward-stable `n x n` solver's error on one entry) over the
/// sector's largest `|λ|`. The projector comparison is further multiplied by
/// `scale / gap`, the conditioning of an invariant subspace separated from
/// the rest of the spectrum by `gap` (Davis–Kahan).
pub fn check_member<R>(
    label: &str,
    source: &TensorMap<R, f64>,
    d: &TensorMap<R, f64>,
    v: &TensorMap<R, f64>,
    reference: (&TensorMap<R, f64>, &TensorMap<R, f64>),
    eps: f64,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
{
    let a = sector_matrices(source);
    let dm = sector_matrices(d);
    let vm = sector_matrices(v);
    let ref_d = sector_matrices(reference.0);
    let ref_v = sector_matrices(reference.1);
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        vm.keys().collect::<Vec<_>>(),
        "{label}: sectors"
    );
    for (sector, matrix) in &a {
        let n = matrix.rows;
        let what = format!("{label} sector {sector}");
        let values: Vec<f64> = (0..n).map(|i| dm[sector].at(i, i)).collect();
        let ref_values: Vec<f64> = (0..n).map(|i| ref_d[sector].at(i, i)).collect();
        let oracle = jacobi_eigenvalues(matrix);
        let scale = oracle
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()))
            .max(1e-300);
        let bound = 32.0 * (n * n) as f64 * eps * scale;
        // Ascending in `d`, as eager orders it.
        for pair in values.windows(2) {
            assert!(pair[0] <= pair[1], "{what}: order {values:?}");
        }
        // Multisets, sorted by value.
        let sorted = |mut list: Vec<f64>| {
            list.sort_by(f64::total_cmp);
            list
        };
        for ((got, want), reference) in sorted(values.clone())
            .iter()
            .zip(&oracle)
            .zip(&sorted(ref_values.clone()))
        {
            assert!(
                (got - want).abs() <= bound,
                "{what}: λ {got} vs oracle {want}"
            );
            assert!(
                (got - reference).abs() <= bound,
                "{what}: λ {got} vs eager {reference}"
            );
        }
        let vs = &vm[sector];
        assert_eq!((vs.rows, vs.cols), (n, n), "{what}: V shape");
        for i in 0..n {
            for j in 0..n {
                let rebuilt: f64 = (0..n).map(|k| vs.at(i, k) * values[k] * vs.at(j, k)).sum();
                assert!(
                    (rebuilt - matrix.at(i, j)).abs() <= bound,
                    "{what}: (V Λ Vᵀ)[{i},{j}] = {rebuilt} vs {}",
                    matrix.at(i, j)
                );
                let gram: f64 = (0..n).map(|k| vs.at(k, i) * vs.at(k, j)).sum();
                let unit = f64::from(u8::from(i == j));
                assert!(
                    (gram - unit).abs() <= 32.0 * (n * n) as f64 * eps,
                    "{what}: (VᵀV)[{i},{j}] = {gram}"
                );
            }
        }
        let groups = |values: &[f64]| {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&x, &y| values[x].total_cmp(&values[y]));
            let mut groups: Vec<Vec<usize>> = Vec::new();
            for index in order {
                match groups.last_mut() {
                    Some(group) if (values[index] - values[group[0]]).abs() <= 1e-6 * scale => {
                        group.push(index)
                    }
                    _ => groups.push(vec![index]),
                }
            }
            groups
        };
        let got_groups = groups(&values);
        let ref_groups = groups(&ref_values);
        assert_eq!(
            got_groups.len(),
            ref_groups.len(),
            "{what}: eigenvalue groups"
        );
        let mut gap = scale;
        let means: Vec<f64> = got_groups.iter().map(|g| values[g[0]]).collect();
        for pair in means.windows(2) {
            gap = gap.min(pair[1] - pair[0]);
        }
        let conditioning = scale / gap.max(f64::MIN_POSITIVE);
        for (group, ref_group) in got_groups.iter().zip(&ref_groups) {
            assert_eq!(group.len(), ref_group.len(), "{what}: group multiplicity");
            let projector = |v: &SectorMatrix, cols: &[usize], i: usize, j: usize| -> f64 {
                cols.iter().map(|&k| v.at(i, k) * v.at(j, k)).sum()
            };
            for i in 0..n {
                for j in 0..n {
                    let got = projector(vs, group, i, j);
                    let want = projector(&ref_v[sector], ref_group, i, j);
                    assert!(
                        (got - want).abs() <= 32.0 * (n * n) as f64 * eps * conditioning,
                        "{what}: projector[{i},{j}] {got} vs eager {want}"
                    );
                }
            }
        }
    }
}

/// Whether a sector list contains a `|λ|` tie between distinct values, the
/// ±λ tie the `±λ` fixtures assert.
pub fn has_plus_minus_tie<R>(d: &TensorMap<R, f64>) -> bool
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
{
    sector_matrices(d).values().any(|m| {
        let values: Vec<f64> = (0..m.rows).map(|i| m.at(i, i)).collect();
        values.iter().any(|&x| {
            x > 0.0
                && values
                    .iter()
                    .any(|&y| y < 0.0 && ((x + y).abs() <= 1e-12 * x))
        })
    })
}

/// Whether some sector has an eigenvalue of multiplicity > 1.
pub fn has_degenerate_group<R>(d: &TensorMap<R, f64>) -> bool
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
{
    sector_matrices(d).values().any(|m| {
        let values: Vec<f64> = (0..m.rows).map(|i| m.at(i, i)).collect();
        values.iter().enumerate().any(|(i, &x)| {
            values[i + 1..]
                .iter()
                .any(|&y| (x - y).abs() <= 1e-9 * x.abs().max(1.0))
        })
    })
}
