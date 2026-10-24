//! Checked Generic contraction whose staged operands and output both need a
//! tree transform, so its core GEMM runs on uncommitted staged structures
//! (the coupled-region core plan of #2129), against oracles that share no
//! core-plan code:
//!
//! - the block-product oracle: `block(A_p ∘ B_p, c) = block(A_p, c) ·
//!   block(B_p, c)` for the core-ordered operands (TensorKit `mul!`, one GEMM
//!   per coupled sector), for GenericToy (outer multiplicity two), SU(2) and
//!   fusion-symbols SU(3) (outer multiplicity two);
//! - the reference steps `C = permute(A_p ∘ B_p)` (TensorKit
//!   `blas_contract!`) for SU(2) and SU(3);
//! - SU(2) through the checked SU(N) provider against the multiplicity-free
//!   SU(2) route, itself checked against the physical dense expansion
//!   `C = Σ A B` (TensorKit `convert(Array, ·)`).

use std::fmt::Debug;
use std::sync::Arc;

use symtn::sector::TypedSectorAdmission;
use symtn::typed::{
    BlockFusionTrees, ContractSpec, CoupledBlock, GradedSpace, Runtime, TensorMap,
    TypedTensorConstructionDispatch, TypedTensorContractDispatch, TypedTensorModeDispatch,
    TypedTensorTransformDispatch,
};

#[path = "../../tests/support/toy_rules.rs"]
mod toy_rules;

use toy_rules::{GenericLabel, GenericToy};

/// The dispatch a provider needs for the fixtures below.
trait Mode<R: TypedSectorAdmission>:
    TypedTensorModeDispatch<R>
    + TypedTensorConstructionDispatch<R, f64>
    + TypedTensorContractDispatch<R, f64>
    + TypedTensorTransformDispatch<R, f64>
{
}

impl<R: TypedSectorAdmission, M> Mode<R> for M where
    M: TypedTensorModeDispatch<R>
        + TypedTensorConstructionDispatch<R, f64>
        + TypedTensorContractDispatch<R, f64>
        + TypedTensorTransformDispatch<R, f64>
{
}

/// `A[a0, a1 | a2, a3] B[b0, b1 | b2, b3]` with `a2 = b1`, `a1 = b3`; the
/// result is `C[b0, a0 | a3, b2]`, a non-identity transform on every stage.
const LHS_AXES: [usize; 2] = [2, 1];
const RHS_AXES: [usize; 2] = [1, 3];
const CODOMAIN: [usize; 2] = [2, 0];
const DOMAIN: [usize; 2] = [1, 3];

fn spec() -> ContractSpec<'static> {
    ContractSpec {
        lhs: &LHS_AXES,
        rhs: &RHS_AXES,
        codomain: &CODOMAIN,
        domain: &DOMAIN,
    }
}

/// A value in `[-3.9, 3.9]` fixed by `text` and `seed` (FNV-1a).
fn hashed_value(text: &str, seed: u64) -> f64 {
    let hash = text
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64 ^ seed, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    ((hash % 2001) as f64 - 1000.0) / 257.0
}

/// A value fixed by the tree labels (as `Debug` text) and local index.
fn label_value<S: Debug>(trees: &BlockFusionTrees<S>, index: &[usize], seed: u64) -> f64 {
    let text = format!(
        "{:?}{:?}{:?}{:?}{:?}{:?}{:?}{index:?}",
        trees.codomain_uncoupled(),
        trees.codomain_innerlines(),
        trees.codomain_vertices(),
        trees.domain_uncoupled(),
        trees.domain_innerlines(),
        trees.domain_vertices(),
        trees.coupled(),
    );
    hashed_value(&text, seed)
}

fn operands<R: TypedSectorAdmission>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    w: &GradedSpace<R>,
    value: impl Fn(&BlockFusionTrees<R::Sector>, &[usize], u64) -> f64,
) -> (TensorMap<R, f64>, TensorMap<R, f64>)
where
    R::Mode: Mode<R>,
{
    let a = TensorMap::from_subblock_fn(runtime, [v, w], [v, w], |t, i| value(t, i, 1)).unwrap();
    let b = TensorMap::from_subblock_fn(runtime, [w, v], [v, w], |t, i| value(t, i, 2)).unwrap();
    (a, b)
}

fn read<R>(block: &CoupledBlock<'_, R, f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(block.rows() * block.cols());
    for c in 0..block.cols() {
        for r in 0..block.rows() {
            out.push(block.get(r, c).unwrap());
        }
    }
    out
}

fn assert_close(what: &str, got: &[f64], want: &[f64], terms: usize) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-12 * scale * terms as f64,
            "{what}: {g} vs {w}"
        );
    }
}

/// TensorKit `blas_contract!` steps on the core-ordered operands
/// `A_p ← (a0, a3 | a2, a1)` and `B_p ← (b1, b3 | b0, b2)`: with
/// `check_steps`, `C = permute(A_p ∘ B_p)`; always `block(A_p ∘ B_p, c) =
/// block(A_p, c) · block(B_p, c)` from the operands' coupled blocks. Returns
/// the number of nonzero core sectors.
fn block_product_case<R>(
    what: &str,
    a: &TensorMap<R, f64>,
    b: &TensorMap<R, f64>,
    check_steps: bool,
) -> usize
where
    R: TypedSectorAdmission,
    R::Mode: Mode<R>,
    R::Sector: Debug,
{
    let a_p = a.permute(&[0, 3], &[2, 1]).unwrap();
    let b_p = b.permute(&[1, 3], &[0, 2]).unwrap();
    let core = a_p.compose(&b_p).unwrap();
    if check_steps {
        let c = a.contract(b, &spec()).unwrap();
        let steps = core.permute(&CODOMAIN, &DOMAIN).unwrap();
        assert_eq!(c.codomain(), steps.codomain(), "{what}: codomain");
        assert_eq!(c.domain(), steps.domain(), "{what}: domain");
        assert_close(
            &format!("{what}: contract vs reference steps"),
            c.dense_data().unwrap(),
            steps.dense_data().unwrap(),
            1,
        );
    }
    let mut sectors = 0;
    for (sector, block) in core.blocks().unwrap() {
        let (lhs, rhs) = (a_p.block(&sector).unwrap(), b_p.block(&sector).unwrap());
        let row_labels = |trees: Vec<_>| format!("{trees:?}");
        assert_eq!(
            row_labels(lhs.col_trees().unwrap()),
            row_labels(rhs.row_trees().unwrap()),
            "{what}: contracted trees align"
        );
        let (l, r) = (read(&lhs), read(&rhs));
        let (rows, inner, cols) = (lhs.rows(), lhs.cols(), rhs.cols());
        let mut want = vec![0.0; rows * cols];
        for j in 0..cols {
            for k in 0..inner {
                for i in 0..rows {
                    want[i + rows * j] += l[i + rows * k] * r[k + inner * j];
                }
            }
        }
        assert_close(
            &format!("{what}: block({sector:?})"),
            &read(&block),
            &want,
            inner.max(1),
        );
        if want.iter().any(|v| v.abs() > 0.5) {
            sectors += 1;
        }
    }
    sectors
}

#[test]
fn generic_toy_core_composition_matches_block_products() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(GenericToy),
        [(GenericLabel::Vacuum, 2), (GenericLabel::X, 3)],
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    let (a, b) = operands(&runtime, &v, &w, label_value);
    // GenericToy's identity F-symbols with non-unit quantum dimensions are
    // not a consistent category, so equivalent bend/braid sequences differ
    // and the reference-step equality is checked on SU(2) and SU(3) only.
    assert!(block_product_case("GenericToy", &a, &b, false) >= 2);
}

#[cfg(feature = "cgc-gen")]
mod racah {
    use super::*;
    use std::collections::BTreeMap;
    use symtn::sector::{LieGroupFusionRule, SU2FusionRule, SU2Irrep};

    #[test]
    fn su3_staged_contraction_matches_block_products() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let rule = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        // `8 ⊗ 8 → 8` has outer multiplicity two.
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
        )
        .unwrap();
        let w = GradedSpace::try_new(rule, [(vec![1i64, 1], 1), (vec![0, 1], 2)])
            .unwrap()
            .try_dual()
            .unwrap();
        let (a, b) = operands(&runtime, &v, &w, label_value);
        assert!(block_product_case("SU(3)", &a, &b, true) >= 2);
    }

    /// Twice-spin labels of one block (SU(2) has no vertices), so the SU(N=2)
    /// and multiplicity-free SU(2) providers fill and key the same entries.
    type Twice = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, usize);

    fn twice<S>(trees: &BlockFusionTrees<S>, spin: impl Fn(&S) -> usize) -> Twice {
        let map = |labels: &[S]| labels.iter().map(&spin).collect::<Vec<_>>();
        (
            map(trees.codomain_uncoupled()),
            map(trees.codomain_innerlines()),
            map(trees.domain_uncoupled()),
            map(trees.domain_innerlines()),
            spin(trees.coupled()),
        )
    }

    #[expect(clippy::ptr_arg, reason = "the SU(N) sector label is a `Vec<i64>`")]
    fn sun_spin(sector: &Vec<i64>) -> usize {
        sector[0] as usize
    }

    fn su2_spin(sector: &SU2Irrep) -> usize {
        sector.twice_spin()
    }

    fn sun_value(trees: &BlockFusionTrees<Vec<i64>>, index: &[usize], seed: u64) -> f64 {
        hashed_value(&format!("{:?}{index:?}", twice(trees, sun_spin)), seed)
    }

    fn su2_value(trees: &BlockFusionTrees<SU2Irrep>, index: &[usize], seed: u64) -> f64 {
        hashed_value(&format!("{:?}{index:?}", twice(trees, su2_spin)), seed)
    }

    fn entries<R: TypedSectorAdmission>(
        t: &TensorMap<R, f64>,
        spin: impl Fn(&R::Sector) -> usize + Copy,
    ) -> BTreeMap<String, Vec<f64>>
    where
        R::Mode: TypedTensorModeDispatch<R>,
    {
        t.subblocks()
            .unwrap()
            .map(|(trees, view)| {
                let shape = view.shape().to_vec();
                let len = shape.iter().product::<usize>();
                let values = (0..len)
                    .map(|mut linear| {
                        let index: Vec<usize> = shape
                            .iter()
                            .map(|&extent| {
                                let i = linear % extent;
                                linear /= extent;
                                i
                            })
                            .collect();
                        *view.get(&index).unwrap()
                    })
                    .collect();
                (format!("{:?}", twice(&trees, spin)), values)
            })
            .collect()
    }

    #[test]
    fn checked_su2_staged_contraction_matches_multiplicity_free_and_dense() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let degeneracies = [(0usize, 2usize), (1, 2), (2, 1)];

        let mf_v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            degeneracies.map(|(s, n)| (SU2Irrep::from_twice_spin(s), n)),
        )
        .unwrap();
        let mf_w = mf_v.try_dual().unwrap();
        let (a, b) = operands(&runtime, &mf_v, &mf_w, su2_value);
        let mf = a.contract(&b, &spec()).unwrap();

        // Physical dense oracle of the multiplicity-free result.
        let (pa, pb, pc) = (
            a.to_physical_dense().unwrap(),
            b.to_physical_dense().unwrap(),
            mf.to_physical_dense().unwrap(),
        );
        let at = |shape: &[usize], index: [usize; 4]| {
            index
                .iter()
                .zip(shape)
                .rev()
                .fold(0, |linear, (&i, &extent)| linear * extent + i)
        };
        let mut want = vec![0.0; pc.data.len()];
        let (sa, sb) = (&pa.shape, &pb.shape);
        for a0 in 0..sa[0] {
            for a3 in 0..sa[3] {
                for b0 in 0..sb[0] {
                    for b2 in 0..sb[2] {
                        let mut sum = 0.0;
                        for x in 0..sa[2] {
                            for y in 0..sa[1] {
                                sum += pa.data[at(sa, [a0, y, x, a3])]
                                    * pb.data[at(sb, [b0, x, b2, y])];
                            }
                        }
                        want[at(&pc.shape, [b0, a0, a3, b2])] = sum;
                    }
                }
            }
        }
        assert_eq!(pc.shape, vec![sb[0], sa[0], sa[3], sb[2]]);
        assert_close("SU(2) MF physical", &pc.data, &want, sa[1] * sa[2]);
        assert!(want.iter().any(|v| v.abs() > 0.5));

        let rule = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap(),
        );
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            degeneracies.map(|(s, n)| (vec![s as i64], n)),
        )
        .unwrap();
        let w = v.try_dual().unwrap();
        let (a, b) = operands(&runtime, &v, &w, sun_value);
        assert!(block_product_case("SU(2)", &a, &b, true) >= 2);
        let checked = a.contract(&b, &spec()).unwrap();

        let checked = entries(&checked, sun_spin);
        let mf = entries(&mf, su2_spin);
        assert_eq!(
            checked.keys().collect::<Vec<_>>(),
            mf.keys().collect::<Vec<_>>()
        );
        for (key, want) in &mf {
            assert_close(
                &format!("SU(2) checked vs MF {key}"),
                &checked[key],
                want,
                16,
            );
        }
    }
}
