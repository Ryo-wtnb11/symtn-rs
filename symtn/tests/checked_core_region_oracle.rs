//! Checked Generic contraction whose staged operands and output both need a
//! tree transform, so its core GEMM runs on uncommitted staged structures
//! (the coupled-region core plan), against oracles that share no
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

#[cfg(feature = "cgc-gen")]
use crate::su2::{su2_irrep, su2_rule};
use std::fmt::Debug;
use std::sync::Arc;
#[cfg(feature = "cgc-gen")]
use symtn::sector::LieIrrep;

use symtn::sector::TypedSectorAdmission;
use symtn::typed::{
    BlockFusionTrees, ContractSpec, CoupledBlock, GradedSpace, Runtime, TensorMap,
    TypedTensorConstructionDispatch, TypedTensorContractDispatch, TypedTensorModeDispatch,
    TypedTensorTransformDispatch,
};

use crate::toy_rules;

use toy_rules::{GenericLabel, GenericToy};

/// The dispatch a provider needs for the fixtures below.
pub(crate) trait Mode<R: TypedSectorAdmission, D: symtn::typed::TensorScalar = f64>:
    TypedTensorModeDispatch<R>
    + TypedTensorConstructionDispatch<R, D>
    + TypedTensorContractDispatch<R, D>
    + TypedTensorTransformDispatch<R, D>
{
}

impl<R: TypedSectorAdmission, D: symtn::typed::TensorScalar, M> Mode<R, D> for M where
    M: TypedTensorModeDispatch<R>
        + TypedTensorConstructionDispatch<R, D>
        + TypedTensorContractDispatch<R, D>
        + TypedTensorTransformDispatch<R, D>
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

fn operands<R: TypedSectorAdmission, D: symtn::typed::TensorScalar>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    w: &GradedSpace<R>,
    value: impl Fn(&BlockFusionTrees<R::Sector>, &[usize], u64) -> D,
) -> (TensorMap<R, D>, TensorMap<R, D>)
where
    R::Mode: Mode<R, D>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let a = TensorMap::from_subblock_fn(engine, runtime, [v, w], [v, w], |t, i| value(t, i, 1))
        .unwrap();
    let b = TensorMap::from_subblock_fn(engine, runtime, [w, v], [v, w], |t, i| value(t, i, 2))
        .unwrap();
    (a, b)
}

fn read<R, D: symtn::typed::TensorScalar>(block: &CoupledBlock<'_, R, D>) -> Vec<D> {
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let a_p = a.permute(engine, &[0, 3], &[2, 1]).unwrap();
    let b_p = b.permute(engine, &[1, 3], &[0, 2]).unwrap();
    let core = a_p.compose(engine, &b_p).unwrap();
    if check_steps {
        let c = a.contract(engine, b, &spec()).unwrap();
        let steps = core.permute(engine, &CODOMAIN, &DOMAIN).unwrap();
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
    let runtime = Runtime::builder().build().unwrap();
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
    use symtn::sector::{CheckedGenericAdmissionMode, LieGroupFusionRule};

    #[test]
    fn su3_staged_contraction_matches_block_products() {
        let runtime = Runtime::builder().build().unwrap();
        let rule = Arc::new(
            LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
                fusion_symbols::lie::GroupId::su(3).unwrap(),
            )
            .unwrap(),
        );
        // `8 ⊗ 8 → 8` has outer multiplicity two.
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (su_irrep(&[0i64, 0]), 2),
                (su_irrep(&[1, 1]), 2),
                (su_irrep(&[1, 0]), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(rule, [(su_irrep(&[1i64, 1]), 1), (su_irrep(&[0, 1]), 2)])
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

    fn sun_spin(sector: &LieIrrep) -> usize {
        sector.dynkin().next().unwrap() as usize
    }

    use crate::legs::su2_spin;

    fn sun_value(trees: &BlockFusionTrees<LieIrrep>, index: &[usize], seed: u64) -> f64 {
        hashed_value(&format!("{:?}{index:?}", twice(trees, sun_spin)), seed)
    }

    fn su2_value(
        trees: &BlockFusionTrees<symtn_core::LieIrrep>,
        index: &[usize],
        seed: u64,
    ) -> f64 {
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
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let degeneracies = [(0usize, 2usize), (1, 2), (2, 1)];

        let mf_v = GradedSpace::try_new(
            Arc::new(su2_rule()),
            degeneracies.map(|(s, n)| (su2_irrep(s), n)),
        )
        .unwrap();
        let mf_w = mf_v.try_dual().unwrap();
        let (a, b) = operands(&runtime, &mf_v, &mf_w, su2_value);
        let mf = a.contract(engine, &b, &spec()).unwrap();

        // Physical dense oracle of the multiplicity-free result.
        let (pa, pb, pc) = (
            a.to_physical_dense(engine).unwrap(),
            b.to_physical_dense(engine).unwrap(),
            mf.to_physical_dense(engine).unwrap(),
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
            LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
                fusion_symbols::lie::GroupId::su(2).unwrap(),
            )
            .unwrap(),
        );
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            degeneracies.map(|(s, n)| (su_irrep(&[s as i64]), n)),
        )
        .unwrap();
        let w = v.try_dual().unwrap();
        let (a, b) = operands(&runtime, &v, &w, sun_value);
        assert!(block_product_case("SU(2)", &a, &b, true) >= 2);
        let checked = a.contract(engine, &b, &spec()).unwrap();

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

#[cfg(feature = "cgc-gen")]
fn su_irrep(labels: &[i64]) -> symtn::sector::LieIrrep {
    fusion_symbols::lie::LieGroup::new(fusion_symbols::lie::GroupId::su(labels.len() + 1).unwrap())
        .unwrap()
        .irrep(labels)
        .unwrap()
}

#[cfg(feature = "cgc-gen")]
pub(crate) mod fermionic {
    use super::*;
    use crate::numerics;
    use symtn::sector::{
        product_sector, CheckedGenericAdmissionMode, FermionParityFusionRule, FermionParityIrrep,
        LieGroupFusionRule, MultiplicityFreeAdmissionMode, ProductFusionRule,
    };
    use symtn::typed::{Complex64, Direction, TensorScalar, TypedTensorTwistDispatch};

    pub(crate) trait FermionMode<R: TypedSectorAdmission, D: TensorScalar>:
        Mode<R, D> + TypedTensorTwistDispatch<R, D>
    {
    }
    impl<R: TypedSectorAdmission, D: TensorScalar, M> FermionMode<R, D> for M where
        M: Mode<R, D> + TypedTensorTwistDispatch<R, D>
    {
    }

    pub(crate) type Su3 =
        ProductFusionRule<LieGroupFusionRule<CheckedGenericAdmissionMode>, FermionParityFusionRule>;

    pub(crate) fn su3_space() -> GradedSpace<Su3> {
        let su3 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            fusion_symbols::lie::GroupId::su(3).unwrap(),
        )
        .unwrap();
        GradedSpace::try_new(
            Arc::new(ProductFusionRule::try_new(su3, FermionParityFusionRule::new()).unwrap()),
            [
                (
                    product_sector(su_irrep(&[0, 0]), FermionParityIrrep::EVEN),
                    1,
                ),
                (
                    product_sector(su_irrep(&[1, 1]), FermionParityIrrep::ODD),
                    2,
                ),
                (
                    product_sector(su_irrep(&[1, 1]), FermionParityIrrep::EVEN),
                    1,
                ),
                (
                    product_sector(su_irrep(&[1, 0]), FermionParityIrrep::ODD),
                    1,
                ),
                (
                    product_sector(su_irrep(&[0, 1]), FermionParityIrrep::ODD),
                    2,
                ),
            ],
        )
        .unwrap()
    }

    pub(crate) fn filled<R, D>(
        engine: &mut dyn symtn::typed::Engine,
        runtime: &Runtime,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
        seed: u64,
        value: impl Fn(f64, f64) -> D,
    ) -> TensorMap<R, D>
    where
        R: TypedSectorAdmission,
        R::Mode: Mode<R, D>,
        D: TensorScalar,
    {
        TensorMap::from_subblock_fn(
            engine,
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |t, i| value(label_value(t, i, seed), label_value(t, i, seed + 97)),
        )
        .unwrap()
    }

    // Every entry is bilinear in the original payloads, including recoupling.
    pub(crate) fn compare<R, D>(
        what: &str,
        got: &TensorMap<R, D>,
        want: &TensorMap<R, D>,
        terms: usize,
    ) where
        R: TypedSectorAdmission,
        R::Mode: Mode<R, D>,
        D: TensorScalar + numerics::Numeric,
    {
        assert_eq!(got.codomain(), want.codomain(), "{what}: codomain");
        assert_eq!(got.domain(), want.domain(), "{what}: domain");
        numerics::assert_slices_close(
            what,
            got.dense_data().unwrap(),
            want.dense_data().unwrap(),
            terms,
        );
    }

    fn canonical<D>(value: impl Fn(f64, f64) -> D + Copy)
    where
        D: TensorScalar + numerics::Numeric + std::ops::AddAssign,
    {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let v = su3_space();
        let dual = v.try_dual().unwrap();
        for uniform in [false, true] {
            let middle = if uniform { [&dual, &dual] } else { [&v, &dual] };
            let a = filled(engine, &runtime, &[&v, &v], &middle, 3, value);
            let b = filled(engine, &runtime, &middle, &[&v], 7, value);
            let spec = ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2],
            };
            let got = a.contract(engine, &b, &spec).unwrap();
            let untwisted = a.compose(engine, &b).unwrap();
            let mut differs = false;
            let mut multiplicity = false;
            let mut nonuniform = false;
            for (sector, block) in got.blocks().unwrap() {
                let (left, right) = (a.block(&sector).unwrap(), b.block(&sector).unwrap());
                let mut theta = vec![1.0; right.rows()];
                for (tree, range) in right.row_trees().unwrap() {
                    let sign = tree
                        .uncoupled()
                        .iter()
                        .zip(middle)
                        .filter(|(_, leg)| leg.is_dual())
                        .fold(1.0, |sign, (s, _)| {
                            sign * if *s.right() == FermionParityIrrep::ODD {
                                -1.0
                            } else {
                                1.0
                            }
                        });
                    theta[range].fill(sign);
                    multiplicity |= tree.vertices().iter().any(|mu| mu.get() == 2);
                }
                nonuniform |= theta.windows(2).any(|signs| signs[0] != signs[1]);
                let (rows, inner, cols) = (left.rows(), left.cols(), right.cols());
                let mut expected = vec![D::from_real(0.0); rows * cols];
                for j in 0..cols {
                    for (k, &theta_k) in theta.iter().enumerate().take(inner) {
                        for i in 0..rows {
                            expected[i + rows * j] += left.get(i, k).unwrap()
                                * D::from_real(theta_k)
                                * right.get(k, j).unwrap();
                        }
                    }
                }
                numerics::assert_slices_close(
                    "odd dual block product",
                    &read(&block),
                    &expected,
                    inner,
                );
                let control = read(&untwisted.block(&sector).unwrap());
                differs |= expected
                    .iter()
                    .zip(control)
                    .any(|(a, b)| (a.wide() - b.wide()).norm() > 1e-6);
            }
            assert!(
                multiplicity,
                "the octet channel must include its second vertex"
            );
            assert!(differs, "removing theta must change the hand oracle");
            assert_eq!(
                nonuniform, !uniform,
                "the public row labels distinguish the two fixtures"
            );
        }
    }

    #[test]
    fn checked_fermionic_uniform_and_nonuniform_twists_match_label_block_products() {
        // What: multiplicity-two blocks carry the product of the odd dual
        // contracted labels, including non-self-dual 3/3bar, on real and complex data.
        canonical(|x, _| x);
        canonical(Complex64::new);
    }

    pub(crate) fn reference_rows<R, D>(
        v: &GradedSpace<R>,
        value: impl Fn(f64, f64) -> D + Copy,
    ) -> Vec<(TensorMap<R, D>, usize)>
    where
        R: TypedSectorAdmission,
        R::Mode: FermionMode<R, D> + symtn::typed::TypedAdjointSpace<R>,
        D: TensorScalar + numerics::Numeric,
    {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let dual = v.try_dual().unwrap();
        let spaces = [v, &dual];
        // TensorKit copyA/copyB cases followed by mixed-side, reordered contractions.
        #[allow(clippy::type_complexity)]
        let geometries: [(
            &[usize],
            &[usize],
            &[usize],
            &[usize],
            &[usize],
            &[usize],
            &[usize],
            bool,
        ); 6] = [
            (
                &[0, 0],
                &[0],
                &[1],
                &[0, 0],
                &[1],
                &[0],
                &[0, 1, 2, 3],
                true,
            ),
            (
                &[0],
                &[0, 1],
                &[0, 1],
                &[0, 0],
                &[1, 2],
                &[0, 1],
                &[0, 1, 2],
                true,
            ),
            (
                &[0, 0],
                &[1],
                &[0],
                &[0, 0],
                &[2],
                &[1],
                &[0, 1, 2, 3],
                false,
            ),
            (
                &[0, 0],
                &[0],
                &[0, 1],
                &[0, 0],
                &[1, 0],
                &[3, 1],
                &[0, 1, 2],
                true,
            ),
            (
                &[0, 0],
                &[0, 0],
                &[0, 1],
                &[0, 0],
                &[1, 0],
                &[3, 1],
                &[2, 0, 3, 1],
                true,
            ),
            (
                &[0, 1],
                &[0, 0],
                &[0, 1],
                &[0, 1],
                &[1, 0],
                &[3, 1],
                &[2, 0, 3, 1],
                true,
            ),
        ];
        let mut rows = Vec::new();
        for (ac, ad, bc, bd, ca, cb, output, twist_a) in geometries {
            let legs = |indices: &[usize]| indices.iter().map(|&i| spaces[i]).collect::<Vec<_>>();
            let a = filled(engine, &runtime, &legs(ac), &legs(ad), 11, value);
            let b = filled(engine, &runtime, &legs(bc), &legs(bd), 17, value);
            let parent_a = a.adjoint(engine).unwrap().materialize(engine).unwrap();
            let parent_b = b.adjoint(engine).unwrap().materialize(engine).unwrap();
            let lazy_a = parent_a.adjoint(engine).unwrap();
            let lazy_b = parent_b.adjoint(engine).unwrap();
            for (a, b) in [(&a, &b), (&lazy_a, &b), (&a, &lazy_b), (&lazy_a, &lazy_b)] {
                let oa = (0..a.rank())
                    .filter(|i| !ca.contains(i))
                    .collect::<Vec<_>>();
                let ob = (0..b.rank())
                    .filter(|i| !cb.contains(i))
                    .collect::<Vec<_>>();
                let mut ap = a.permute(engine, &oa, ca).unwrap();
                let mut bp = b.permute(engine, cb, &ob).unwrap();
                if twist_a {
                    // An external domain index has the dual of the stored
                    // domain leg: TensorKit's !isdual selects stored dual legs.
                    let axes = ap
                        .domain()
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| l.is_dual())
                        .map(|(i, _)| ap.codomain_rank() + i)
                        .collect::<Vec<_>>();
                    ap = ap.twist(engine, &axes, Direction::Forward).unwrap();
                } else {
                    let axes = bp
                        .codomain()
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| l.is_dual())
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>();
                    bp = bp.twist(engine, &axes, Direction::Forward).unwrap();
                }
                let split = oa.len();
                let steps = ap
                    .compose(engine, &bp)
                    .unwrap()
                    .permute(engine, &output[..split], &output[split..])
                    .unwrap();
                let spec = ContractSpec {
                    lhs: ca,
                    rhs: cb,
                    codomain: &output[..split],
                    domain: &output[split..],
                };
                let got = a.contract(engine, b, &spec).unwrap();
                let terms = a
                    .dense_data()
                    .map_or_else(|_| parent_a.dense_data().unwrap().len(), |d| d.len())
                    * b.dense_data()
                        .map_or_else(|_| parent_b.dense_data().unwrap().len(), |d| d.len());
                compare(
                    "blas_contract steps including lazy operands",
                    &got,
                    &steps,
                    terms,
                );
                rows.push((got, terms));
            }
        }
        rows
    }

    #[test]
    fn checked_fermionic_general_and_lazy_contraction_match_blas_contract_steps() {
        // What: both twist placements and logical adjoints on either or both
        // inputs give TensorKit's permute/twist/compose/permute value.
        let v = su3_space();
        reference_rows(&v, |x, _| x);
        reference_rows(&v, Complex64::new);
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn su2_spaces() -> (
        GradedSpace<
            ProductFusionRule<
                FermionParityFusionRule,
                LieGroupFusionRule<MultiplicityFreeAdmissionMode>,
            >,
        >,
        GradedSpace<
            ProductFusionRule<
                FermionParityFusionRule,
                LieGroupFusionRule<CheckedGenericAdmissionMode>,
            >,
        >,
    ) {
        let labels = [
            (product_sector(FermionParityIrrep::EVEN, su2_irrep(0)), 1),
            (product_sector(FermionParityIrrep::ODD, su2_irrep(1)), 2),
            (product_sector(FermionParityIrrep::EVEN, su2_irrep(2)), 1),
        ];
        let checked = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            fusion_symbols::lie::GroupId::su(2).unwrap(),
        )
        .unwrap();
        (
            GradedSpace::try_new(
                Arc::new(
                    ProductFusionRule::try_new(FermionParityFusionRule::new(), su2_rule()).unwrap(),
                ),
                labels.clone(),
            )
            .unwrap(),
            GradedSpace::try_new(
                Arc::new(
                    ProductFusionRule::try_new(FermionParityFusionRule::new(), checked).unwrap(),
                ),
                labels,
            )
            .unwrap(),
        )
    }

    pub(crate) fn label_entries<R, D>(
        tensor: &TensorMap<R, D>,
    ) -> std::collections::BTreeMap<String, Vec<D>>
    where
        R: TypedSectorAdmission,
        R::Mode: Mode<R, D>,
        D: TensorScalar,
    {
        tensor
            .subblocks()
            .unwrap()
            .map(|(t, b)| {
                let len = b.shape().iter().product();
                let values = (0..len)
                    .map(|mut linear| {
                        let index = b
                            .shape()
                            .iter()
                            .map(|&extent| {
                                let i = linear % extent;
                                linear /= extent;
                                i
                            })
                            .collect::<Vec<_>>();
                        *b.get(&index).unwrap()
                    })
                    .collect();
                (format!("{t:?}"), values)
            })
            .collect()
    }

    fn cross_mode<D>(value: impl Fn(f64, f64) -> D + Copy)
    where
        D: TensorScalar + numerics::Numeric,
    {
        let (mf, checked) = su2_spaces();
        let mf = reference_rows(&mf, value);
        let checked = reference_rows(&checked, value);
        for ((mf, terms), (checked, _)) in mf.iter().zip(&checked) {
            let (mf, checked) = (label_entries(mf), label_entries(checked));
            assert_eq!(
                mf.keys().collect::<Vec<_>>(),
                checked.keys().collect::<Vec<_>>()
            );
            for (key, expected) in mf {
                // Bilinear comparison bound includes all source entries.
                numerics::assert_slices_close(&key, &checked[&key], &expected, *terms);
            }
        }
    }

    #[test]
    fn fermionic_contraction_agrees_across_modes_by_labels() {
        // What: fZ2 x SU(2) agrees across admission modes for the same real
        // and complex values and the same ordinary/lazy contraction geometries.
        cross_mode(|x, _| x);
        cross_mode(Complex64::new);
    }
}
