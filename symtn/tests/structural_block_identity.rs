//! `isomorphism(W ← V)` and `isometry(W ← V)` between different spaces follow
//! TensorKit's `one!` convention (`src/tensors/linalg.jl`): every coupled
//! sector block is the (partial) identity `δ_ij` over the whole block, across
//! fusion-tree boundaries, with rows and columns in TensorKit's
//! `fusiontrees` order.
//!
//! The order is part of the result: a partial identity selects the leading
//! trees, and an isomorphism between different spaces pairs row and column
//! trees by position. symtn stores a block's trees in TensorKit's nesting
//! but with `SectorId` sector order (see `CoupledBlock`), so the main oracle is the physical dense tensor
//! (`to_physical_dense`, TensorKit's `convert(Array, t)` layout) against
//! literals checked against TensorKit 0.17.1. The block-identity checks pin
//! the block structure for fixtures whose storage order is TensorKit's.

use std::fmt::Debug;
use std::sync::Arc;

use symtn::sector::{
    ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep, TypedSectorAdmission, U1FusionRule,
    U1Irrep,
};
use symtn::typed::{Complex64, GradedSpace, PhysicalDense, Runtime, TensorMap, TensorScalar};

/// Asserts that `dense` has `shape` and is zero except at the 1-based
/// column-major `nonzeros`, as TensorKit prints `convert(Array, t)`.
fn assert_dense<D>(
    what: &str,
    dense: &PhysicalDense<D>,
    shape: &[usize],
    nonzeros: &[(&[usize], f64)],
) where
    D: Copy + Into<Complex64>,
{
    assert_eq!(dense.shape, shape, "{what}: shape");
    let mut want = vec![0.0; shape.iter().product()];
    for &(index, value) in nonzeros {
        let (mut flat, mut stride) = (0, 1);
        for (&i, &n) in index.iter().zip(shape) {
            flat += (i - 1) * stride;
            stride *= n;
        }
        want[flat] = value;
    }
    for (flat, (&got, want)) in dense.data.iter().zip(want).enumerate() {
        let got: Complex64 = got.into();
        assert!(
            (got - want).norm() < 1e-12,
            "{what}: entry {flat}: got {got}, want {want}"
        );
    }
}

/// Asserts that `t` stores exactly the coupled blocks `expected` as
/// `(sector, rows, cols)` and that each is the leading-diagonal identity.
fn assert_identity_blocks<R, D>(
    what: &str,
    t: &TensorMap<R, D>,
    expected: &[(R::Sector, usize, usize)],
) where
    R: TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
    R::Sector: PartialEq + Debug,
    D: TensorScalar + PartialEq + Debug,
{
    let blocks: Vec<_> = t.blocks().unwrap().collect();
    assert_eq!(blocks.len(), expected.len(), "{what}: stored sectors");
    for (sector, block) in blocks {
        let &(_, rows, cols) = expected
            .iter()
            .find(|(s, _, _)| *s == sector)
            .unwrap_or_else(|| panic!("{what}: unexpected sector {sector:?}"));
        assert_eq!(
            (block.rows(), block.cols()),
            (rows, cols),
            "{what}: {sector:?}"
        );
        for i in 0..rows {
            for j in 0..cols {
                let want = D::from_real(if i == j { 1.0 } else { 0.0 });
                assert_eq!(block.get(i, j), Some(want), "{what}: {sector:?} ({i},{j})");
            }
        }
    }
}

macro_rules! for_dtypes {
    ($body:ident) => {
        $body::<f64>("f64");
        $body::<Complex64>("c64");
    };
}

#[test]
fn u1_isomorphism_and_isometry_are_block_identities_between_different_spaces() {
    fn run<D: TensorScalar + PartialEq + Debug>(what: &str) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let rt = Runtime::builder().build().unwrap();
        let space = |pairs: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                pairs.iter().map(|&(q, n)| (U1Irrep::new(q), n)),
            )
            .unwrap()
        };
        let v = space(&[(0, 1), (1, 2)]);
        // V ⊗ V: q=0: 1·1; q=1: 1·2 + 2·1 = 4 (two trees); q=2: 2·2 = 4.
        let fused = space(&[(0, 1), (1, 4), (2, 4)]);
        let iso: TensorMap<_, D> = TensorMap::isomorphism(engine, &rt, [&v, &v], [&fused]).unwrap();
        let u = U1Irrep::new;
        assert_identity_blocks(
            &format!("{what} isomorphism V⊗V ← W"),
            &iso,
            &[(u(0), 1, 1), (u(1), 4, 4), (u(2), 4, 4)],
        );
        let inverse: TensorMap<_, D> =
            TensorMap::isomorphism(engine, &rt, [&fused], [&v, &v]).unwrap();
        assert_identity_blocks(
            &format!("{what} isomorphism W ← V⊗V"),
            &inverse,
            &[(u(0), 1, 1), (u(1), 4, 4), (u(2), 4, 4)],
        );
        // q=0 lives only in the codomain, so it has no stored block.
        let small = space(&[(1, 3), (2, 1)]);
        let w: TensorMap<_, D> = TensorMap::isometry(engine, &rt, [&v, &v], [&small]).unwrap();
        assert_identity_blocks(
            &format!("{what} isometry V⊗V ← W'"),
            &w,
            &[(u(1), 4, 3), (u(2), 4, 1)],
        );
    }
    for_dtypes!(run);
}

#[test]
fn su2_isomorphism_and_isometry_are_block_identities_between_different_spaces() {
    fn run<D: TensorScalar + PartialEq + Debug>(what: &str) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let rt = Runtime::builder().build().unwrap();
        let space = |pairs: &[(usize, usize)]| {
            GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                pairs
                    .iter()
                    .map(|&(s, n)| (SU2Irrep::from_twice_spin(s), n)),
            )
            .unwrap()
        };
        let j = SU2Irrep::from_twice_spin;
        let v = space(&[(0, 1), (1, 2)]);
        // V ⊗ V, twice-spin labels: 0: 0⊗0 (1) + ½⊗½ (4) = 5 (two trees);
        // ½: 0⊗½ (2) + ½⊗0 (2) = 4 (two trees); 1: ½⊗½ (4).
        let fused = space(&[(0, 5), (1, 4), (2, 4)]);
        let iso: TensorMap<_, D> = TensorMap::isomorphism(engine, &rt, [&v, &v], [&fused]).unwrap();
        assert_identity_blocks(
            &format!("{what} isomorphism V⊗V ← W"),
            &iso,
            &[(j(0), 5, 5), (j(1), 4, 4), (j(2), 4, 4)],
        );
        // ½ lives only in the codomain, so it has no stored block.
        let small = space(&[(0, 2), (2, 3)]);
        let w: TensorMap<_, D> = TensorMap::isometry(engine, &rt, [&v, &v], [&small]).unwrap();
        assert_identity_blocks(
            &format!("{what} isometry V⊗V ← W'"),
            &w,
            &[(j(0), 5, 2), (j(2), 4, 3)],
        );
    }
    for_dtypes!(run);
}

/// U(1) with negative charges, where TensorKit's sector order
/// `0, 1, -1` is not symtn's storage order. Physical index 1, 2, 3 of `V`
/// is charge 0, 1, -1; `V'` lists `V`'s order. Literals checked against
/// TensorKit 0.17.1 `convert(Array, t)`.
#[test]
fn u1_negative_charge_isometry_and_isomorphism_match_tensorkit_dense() {
    fn run<D: TensorScalar + Into<Complex64>>(what: &str) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let rt = Runtime::builder().build().unwrap();
        let space = |pairs: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                pairs.iter().map(|&(q, n)| (U1Irrep::new(q), n)),
            )
            .unwrap()
        };
        let v = space(&[(0, 1), (1, 1), (-1, 1)]);
        let w = space(&[(0, 2)]);

        // TensorKit trees of V ⊗ V at charge 0: (0,0), (-1,1), (1,-1).
        let t: TensorMap<_, D> = TensorMap::isometry(engine, &rt, [&v, &v], [&w]).unwrap();
        assert_dense(
            &format!("{what} isometry V⊗V ← U1(0=>2)"),
            &t.to_physical_dense(engine).unwrap(),
            &[3, 3, 2],
            &[(&[1, 1, 1], 1.0), (&[3, 2, 2], 1.0)],
        );

        // TensorKit trees of V ⊗ V' at charge 0: (0,0), (1,-1), (-1,1).
        let t: TensorMap<_, D> =
            TensorMap::isometry(engine, &rt, [&v, &v.try_dual().unwrap()], [&w]).unwrap();
        assert_dense(
            &format!("{what} isometry V⊗V' ← U1(0=>2)"),
            &t.to_physical_dense(engine).unwrap(),
            &[3, 3, 2],
            &[(&[1, 1, 1], 1.0), (&[2, 2, 2], 1.0)],
        );

        // W's physical order is 0 (×3), 1 (×2), -1 (×2), 2, -2.
        let fused = space(&[(0, 3), (1, 2), (-1, 2), (2, 1), (-2, 1)]);
        let t: TensorMap<_, D> = TensorMap::isomorphism(engine, &rt, [&v, &v], [&fused]).unwrap();
        assert_dense(
            &format!("{what} isomorphism V⊗V ← W"),
            &t.to_physical_dense(engine).unwrap(),
            &[3, 3, 9],
            &[
                (&[1, 1, 1], 1.0),
                (&[3, 2, 2], 1.0),
                (&[2, 3, 3], 1.0),
                (&[2, 1, 4], 1.0),
                (&[1, 2, 5], 1.0),
                (&[3, 1, 6], 1.0),
                (&[1, 3, 7], 1.0),
                (&[2, 2, 8], 1.0),
                (&[3, 3, 9], 1.0),
            ],
        );
    }
    run::<f64>("f64");
    run::<Complex64>("c64");
}

/// SU(2) with three legs: TensorKit's trees of `S⊗S⊗S` at spin ½ are
/// (½,0,0), (0,½,0), (0,0,½), (½,½,½; line 0), (½,½,½; line 1), so a
/// four-copy isometry keeps the line-0 tree and drops the line-1 tree.
/// Physical index 1 of `S` is spin 0, 2 and 3 are spin ½. Literals checked
/// against TensorKit 0.17.1 `convert(Array, t)`.
#[test]
fn su2_three_leg_isometry_matches_tensorkit_dense() {
    use std::f64::consts::FRAC_1_SQRT_2;
    fn run<D: TensorScalar + Into<Complex64>>(what: &str) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let rt = Runtime::builder().build().unwrap();
        let space = |pairs: &[(usize, usize)]| {
            GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                pairs
                    .iter()
                    .map(|&(s, n)| (SU2Irrep::from_twice_spin(s), n)),
            )
            .unwrap()
        };
        let s = space(&[(0, 1), (1, 1)]);
        let w = space(&[(1, 4)]);
        let t: TensorMap<_, D> = TensorMap::isometry(engine, &rt, [&s, &s, &s], [&w]).unwrap();
        assert_dense(
            &format!("{what} isometry S⊗S⊗S ← SU2(½=>4)"),
            &t.to_physical_dense(engine).unwrap(),
            &[3, 3, 3, 8],
            &[
                (&[2, 1, 1, 1], 1.0),
                (&[3, 1, 1, 2], 1.0),
                (&[1, 2, 1, 3], 1.0),
                (&[1, 3, 1, 4], 1.0),
                (&[1, 1, 2, 5], 1.0),
                (&[1, 1, 3, 6], 1.0),
                (&[3, 2, 2, 7], -FRAC_1_SQRT_2),
                (&[2, 3, 2, 7], FRAC_1_SQRT_2),
                (&[3, 2, 3, 8], -FRAC_1_SQRT_2),
                (&[2, 3, 3, 8], FRAC_1_SQRT_2),
            ],
        );
    }
    run::<f64>("f64");
    run::<Complex64>("c64");
}

/// Fusion multiplicity: the octet appears twice in 8 ⊗ 8, so its block rows
/// span trees that differ only in the vertex label.
/// SU(2) with three legs where symtn's storage order is not TensorKit's: at
/// spin ½, TensorKit's trees of `S⊗S⊗S` (S = 0, ½, 1) start (½,0,0),
/// (0,½,0), (1,½,0), (½,1,0), (0,0,½; line 0), (½,½,½; line 0), because the
/// line into the top vertex varies before `u₂`. A six-copy isometry keeps
/// those six trees. Physical index 1 of `S` is spin 0, 2-3 spin ½, 4-6
/// spin 1. Literals checked against TensorKit 0.17.1 `convert(Array, t)`.
#[test]
fn su2_three_leg_isometry_follows_tensorkit_inner_line_order() {
    fn run<D: TensorScalar + Into<Complex64>>(what: &str) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let rt = Runtime::builder().build().unwrap();
        let space = |pairs: &[(usize, usize)]| {
            GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                pairs
                    .iter()
                    .map(|&(s, n)| (SU2Irrep::from_twice_spin(s), n)),
            )
            .unwrap()
        };
        let s = space(&[(0, 1), (1, 1), (2, 1)]);
        let w = space(&[(1, 6)]);
        let (a, b) = ((1.0f64 / 3.0).sqrt(), (2.0f64 / 3.0).sqrt());
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let t: TensorMap<_, D> = TensorMap::isometry(engine, &rt, [&s, &s, &s], [&w]).unwrap();
        assert_dense(
            &format!("{what} isometry S⊗S⊗S ← SU2(½=>6)"),
            &t.to_physical_dense(engine).unwrap(),
            &[6, 6, 6, 12],
            &[
                (&[2, 1, 1, 1], 1.0),
                (&[3, 1, 1, 2], 1.0),
                (&[1, 2, 1, 3], 1.0),
                (&[1, 3, 1, 4], 1.0),
                (&[5, 2, 1, 5], -a),
                (&[4, 3, 1, 5], b),
                (&[6, 2, 1, 6], -b),
                (&[5, 3, 1, 6], a),
                (&[3, 4, 1, 7], -b),
                (&[2, 5, 1, 7], a),
                (&[3, 5, 1, 8], -a),
                (&[2, 6, 1, 8], b),
                (&[1, 1, 2, 9], 1.0),
                (&[1, 1, 3, 10], 1.0),
                (&[3, 2, 2, 11], -h),
                (&[2, 3, 2, 11], h),
                (&[3, 2, 3, 12], -h),
                (&[2, 3, 3, 12], h),
            ],
        );
    }
    run::<f64>("f64");
    run::<Complex64>("c64");
}

/// SU(2)⊠SU(2), where TensorKit iterates an inner line `c ⊗ dual(b)` with
/// the first factor fastest (TensorKitSectors 0.3.9 `src/product.jl` `⊗`),
/// not in product `isless` order: at coupled (½,½) the lines of Y⊗Y⊗Y run
/// (0,0), (1,0), (0,1), (1,1). Products have no physical basis, so the
/// oracle is the nonzero subblock entries, as `(coupled, codomain line,
/// domain column)` with value 1, checked against TensorKit 0.17.1
/// `t[f₁, f₂]`.
#[test]
fn su2_product_isometry_follows_tensorkit_inner_line_order() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    type Su2Su2 = ProductFusionRule<SU2FusionRule, SU2FusionRule>;
    let rt = Runtime::builder().build().unwrap();
    let p = |a: usize, b: usize| {
        ProductSector::new(SU2Irrep::from_twice_spin(a), SU2Irrep::from_twice_spin(b))
    };
    let space = |pairs: &[((usize, usize), usize)]| {
        GradedSpace::try_new(
            Arc::new(Su2Su2::try_new(SU2FusionRule, SU2FusionRule).unwrap()),
            pairs.iter().map(|&((a, b), n)| (p(a, b), n)),
        )
        .unwrap()
    };
    let y = space(&[((1, 1), 1)]);
    let w = space(&[((1, 1), 2), ((1, 3), 2), ((3, 1), 2), ((3, 3), 1)]);
    let t: TensorMap<_, f64> = TensorMap::isometry(engine, &rt, [&y, &y, &y], [&w]).unwrap();
    let mut got = Vec::new();
    for (trees, block) in t.subblocks().unwrap() {
        let columns = *block.shape().last().unwrap();
        for column in 0..columns {
            let value = *block.get(&[0, 0, 0, column]).unwrap();
            if value != 0.0 {
                assert_eq!(value, 1.0);
                let line = trees.codomain_innerlines()[0];
                got.push((*trees.coupled(), line, column));
            }
        }
    }
    got.sort();
    let mut want = vec![
        (p(1, 1), p(0, 0), 0),
        (p(1, 1), p(2, 0), 1),
        (p(1, 3), p(0, 2), 0),
        (p(1, 3), p(2, 2), 1),
        (p(3, 1), p(2, 0), 0),
        (p(3, 1), p(2, 2), 1),
        (p(3, 3), p(2, 2), 0),
    ];
    want.sort();
    assert_eq!(got, want);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_isometry_is_the_block_identity_across_multiplicity_vertices() {
    use symtn::sector::LieGroupFusionRule;
    fn run<D: TensorScalar + PartialEq + Debug>(what: &str) {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let rt = Runtime::builder().build().unwrap();
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        let (octet, singlet) = (vec![1i64, 1], vec![0i64, 0]);
        let v = GradedSpace::try_new(
            Arc::clone(&provider),
            [(octet.clone(), 2), (singlet.clone(), 1)],
        )
        .unwrap();
        // V ⊗ V: singlet: 8⊗8 (4) + 1⊗1 (1) = 5; octet: 8⊗8 twice (2·4)
        // + 8⊗1 (2) + 1⊗8 (2) = 12.
        let small =
            GradedSpace::try_new(provider, [(octet.clone(), 3), (singlet.clone(), 5)]).unwrap();
        let w: TensorMap<_, D> = TensorMap::isometry(engine, &rt, [&v, &v], [&small]).unwrap();
        assert_identity_blocks(
            &format!("{what} isometry V⊗V ← W"),
            &w,
            &[(octet, 12, 3), (singlet, 5, 5)],
        );
    }
    for_dtypes!(run);
}
