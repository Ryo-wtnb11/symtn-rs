//! Independent oracles for the whole-basis checked Generic composer (#1962).
//!
//! - Original-step composition: every source column of the checked plan
//!   equals the public per-pair checked operation on that source (TensorKit's
//!   `FSPBraidKey`/`FSPTransposeKey` pair path), resolved by key, within
//!   tolerance. It shares only the elementary move kernels with the composer.
//! - Physical dense oracle: SU(3) splitting tensors built from fusion-symbols CGCs,
//!   permuted as dense arrays and re-expanded in the destination basis.
//! - Gauge covariance with complex coefficients: a vertex-gauge-transformed
//!   SU(3) provider must give the conjugated real plan.
//! - Round trips are computed by the code under test and are supplementary.

use super::*;
use num_complex::Complex64;
use symtn_core::{
    generic_braid_tree_pair_checked, generic_permute_tree_pair_checked,
    generic_transpose_tree_pair_checked, BraidingStyleKind, CategoricalScalar, FusionStyleKind,
    LieGroupFusionRule, LieGroupFusionRuleError, SectorVec,
};

trait Near: Copy {
    fn distance(self, other: Self) -> f64;
    fn magnitude(self) -> f64;
}

impl Near for f64 {
    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl Near for Complex64 {
    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

const TOL: f64 = 1e-10;

type Column<S> = Vec<(FusionTreePairKey, S)>;
type Legs = Vec<(SectorId, bool)>;

/// Every source column of a plan, keyed by source. A Multi spec stores
/// structurally absent coefficients as zero.
fn plan_columns<S: Copy>(plan: &TreeTransformGroupPlan<S>) -> Vec<(FusionTreePairKey, Column<S>)> {
    let mut columns = Vec::new();
    for spec in plan.specs() {
        let (dst, src) = (spec.dst_keys(), spec.src_keys());
        let u = spec.recoupling_coefficients_dst_src();
        for (column, source) in src.iter().enumerate() {
            columns.push((
                source.clone(),
                dst.iter()
                    .enumerate()
                    .map(|(row, key)| (key.clone(), u[row * src.len() + column]))
                    .collect(),
            ));
        }
    }
    columns
}

fn coefficient<S: Near + num_traits::Zero>(column: &Column<S>, key: &FusionTreePairKey) -> S {
    column
        .iter()
        .find(|(destination, _)| destination == key)
        .map_or_else(S::zero, |(_, value)| *value)
}

fn per_pair<P>(
    provider: &P,
    operation: &TreeTransformOperation,
    source: &FusionTreePairKey,
) -> Column<P::Scalar>
where
    P: CheckedGenericRigidSymbols,
{
    let (cp, dp) = (
        operation.codomain_permutation(),
        operation.domain_permutation(),
    );
    match operation.kind() {
        TreeTransformOperationKind::Permute => {
            generic_permute_tree_pair_checked(provider, source, cp, dp)
        }
        TreeTransformOperationKind::Braid => generic_braid_tree_pair_checked(
            provider,
            source,
            cp,
            dp,
            operation.codomain_levels(),
            operation.domain_levels(),
        ),
        TreeTransformOperationKind::Transpose => {
            generic_transpose_tree_pair_checked(provider, source, cp, dp)
        }
    }
    .unwrap_or_else(|_| panic!("per-pair oracle failed for {operation:?}"))
}

/// Original-step oracle plus the structural contract: full key-resolved
/// columns, Single/Multi classification, unique destinations ordered
/// source-major (each group's first source reaches a prefix of its
/// destinations). Returns the number of Multi specs.
fn assert_matches_per_pair<P>(
    provider: &P,
    operation: &TreeTransformOperation,
    structure: &BlockStructure,
) -> usize
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: Near + CategoricalScalar + num_traits::Zero,
{
    let plan = build_checked_generic_tree_pair_transform_group_plan(
        provider,
        operation.clone(),
        structure,
    )
    .unwrap();
    let columns = plan_columns(&plan);
    assert_eq!(columns.len(), structure.block_count(), "{operation:?}");
    let mut oracle_by_source = Vec::new();
    for (source, column) in &columns {
        let oracle = per_pair(provider, operation, source);
        for (destination, expected) in &oracle {
            assert!(
                column.iter().any(|(key, _)| key == destination),
                "{operation:?}: oracle destination missing from the plan column"
            );
            let actual = coefficient(column, destination);
            assert!(
                actual.distance(*expected) <= TOL,
                "{operation:?}: {actual:?} != {expected:?}",
                actual = actual.magnitude(),
                expected = expected.magnitude()
            );
        }
        for (destination, value) in column {
            if !oracle.iter().any(|(key, _)| key == destination) {
                assert!(value.magnitude() <= TOL, "{operation:?}: extra destination");
            }
        }
        oracle_by_source.push((source.clone(), oracle));
    }

    let mut multis = 0;
    for group in structure.fusion_tree_group_slice() {
        let keys = group
            .block_indices()
            .iter()
            .map(|&index| match structure.block(index).unwrap().key() {
                BlockKey::FusionTree(key) => key.clone(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let oracle = keys
            .iter()
            .map(|key| {
                &oracle_by_source
                    .iter()
                    .find(|(source, _)| source == key)
                    .unwrap()
                    .1
            })
            .collect::<Vec<_>>();
        let singleton = oracle.iter().all(|rows| rows.len() == 1);
        let mut targets = oracle
            .iter()
            .filter(|rows| rows.len() == 1)
            .map(|rows| rows[0].0.clone())
            .collect::<Vec<_>>();
        let injective = {
            let before = targets.len();
            targets.sort_by_key(|key| format!("{key:?}"));
            targets.dedup();
            before == targets.len()
        };
        let specs = plan
            .specs()
            .iter()
            .filter(|spec| spec.src_keys().iter().any(|key| keys.contains(key)))
            .collect::<Vec<_>>();
        if singleton && injective {
            assert_eq!(
                specs.len(),
                keys.len(),
                "{operation:?}: expected Single specs"
            );
            assert!(specs
                .iter()
                .all(|spec| spec.src_keys().len() == 1 && spec.dst_keys().len() == 1));
        } else {
            multis += 1;
            assert_eq!(specs.len(), 1, "{operation:?}: expected one Multi spec");
            let spec = specs[0];
            assert_eq!(spec.src_keys(), keys.as_slice());
            let dst = spec.dst_keys();
            for (index, key) in dst.iter().enumerate() {
                assert!(!dst[..index].contains(key), "duplicate destination");
            }
            // Source-major first appearance: the first source's reached
            // destinations come first (#1962 changes only the order within
            // one source, which is visible through `specs()`).
            let first = &oracle[0];
            assert!(dst[..first.len()]
                .iter()
                .all(|key| first.iter().any(|(reached, _)| reached == key)));
        }
    }
    multis
}

fn sun_structure(
    rule: &LieGroupFusionRule,
    codomain: &[(SectorId, bool)],
    domain: &[(SectorId, bool)],
) -> Arc<BlockStructure> {
    FusionTreeHomSpace::new(
        FusionProductSpace::new(
            codomain
                .iter()
                .map(|&(sector, dual)| SectorLeg::new([(sector, 1)], dual)),
        ),
        FusionProductSpace::new(
            domain
                .iter()
                .map(|&(sector, dual)| SectorLeg::new([(sector, 1)], dual)),
        ),
    )
    .coupled_subblock_structure_from_leg_degeneracies_generic_checked(rule)
    .unwrap()
}

/// Every planar transpose of a `codomain_rank|domain_rank` tensor: each
/// cyclic rotation of the linearized legs (codomain, then reversed domain)
/// into every target split, both rotation directions and empty sides
/// included.
fn all_transposes(codomain_rank: usize, domain_rank: usize) -> Vec<TreeTransformOperation> {
    let total = codomain_rank + domain_rank;
    let linear = (0..codomain_rank)
        .chain((codomain_rank..total).rev())
        .collect::<Vec<_>>();
    let mut operations = Vec::new();
    for shift in 0..total {
        let rotated = (0..total)
            .map(|index| linear[(index + shift) % total])
            .collect::<Vec<_>>();
        for split in 0..=total {
            let codomain = rotated[..split].to_vec();
            let domain = rotated[split..].iter().rev().copied().collect::<Vec<_>>();
            operations.push(TreeTransformOperation::transpose(codomain, domain));
        }
    }
    operations
}

fn sun3() -> (LieGroupFusionRule, SectorId, SectorId, SectorId) {
    let rule = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let vacuum = rule.encode_dynkin(&[0, 0]).unwrap();
    let three = rule.encode_dynkin(&[1, 0]).unwrap();
    let adjoint = rule.encode_dynkin(&[1, 1]).unwrap();
    (rule, vacuum, three, adjoint)
}

/// What: over multiplicity (SU(3) adjoint, `N_{8,8}^8 = 2`; SU(4)
/// adjoint), dual and non-self-dual legs (SU(3) 3 / 3̄), permutations,
/// level-carrying braids, and every transpose cycle in both directions
/// including empty-codomain and empty-domain orders, every checked plan
/// column equals the original-step per-pair composition, and repeated
/// destinations produce Multi specs.
#[test]
fn checked_composer_matches_original_step_composition_on_sun() {
    let (rule, _, three, adjoint) = sun3();
    let a = (adjoint, false);
    let su4 = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(4).unwrap()).unwrap();
    let su4_adjoint = (su4.encode_dynkin(&[1, 0, 1]).unwrap(), false);
    let mut multis = 0;
    let cases: Vec<(&LieGroupFusionRule, Legs, Legs)> = vec![
        (&rule, vec![a, a], vec![a]),
        (&rule, vec![a, a, a, a], vec![]),
        (&rule, vec![a, a], vec![a, a]),
        (&rule, vec![a, a, a], vec![a]),
        (
            &rule,
            vec![(three, false), (three, true), (three, false)],
            vec![(three, true)],
        ),
        (
            &rule,
            vec![(three, true), (three, false)],
            vec![(three, false), (three, true)],
        ),
        (&su4, vec![su4_adjoint, su4_adjoint], vec![su4_adjoint]),
    ];
    for (provider, codomain, domain) in &cases {
        let structure = sun_structure(provider, codomain, domain);
        let (p, q) = (codomain.len(), domain.len());
        let total = p + q;
        let mut operations = all_transposes(p, q);
        let codomain_reversed = (0..p).rev().collect::<Vec<_>>();
        let domain_identity = (p..total).collect::<Vec<_>>();
        operations.push(TreeTransformOperation::permute(
            codomain_reversed.clone(),
            domain_identity.clone(),
        ));
        operations.push(TreeTransformOperation::permute(
            (0..p).collect::<Vec<_>>(),
            (p..total).rev().collect::<Vec<_>>(),
        ));
        // A cross-side permutation with a repartition.
        operations.push(TreeTransformOperation::permute(
            (1..total).collect::<Vec<_>>(),
            [0],
        ));
        operations.push(TreeTransformOperation::braid(
            codomain_reversed,
            domain_identity,
            (0..p).rev().collect::<Vec<_>>(),
            (p..total).collect::<Vec<_>>(),
        ));
        for operation in &operations {
            multis += assert_matches_per_pair(*provider, operation, &structure);
        }
    }
    assert!(multis > 0, "the matrix must exercise grouped Multi specs");
}

/// What: anyonic levels and inverse braids follow the per-pair oracle.
#[test]
fn checked_composer_matches_original_step_composition_on_anyonic_levels() {
    let rule = AnyonicGenericRule;
    let provider = CheckedPlanSpy::new(&rule);
    let pairs = dense_generic_source_pairs(&DenseGenericRule);
    let structure = packed_fixture_structure(
        3,
        pairs
            .iter()
            .cloned()
            .map(BlockKey::from)
            .map(|key| (key, vec![1usize; 3])),
    )
    .unwrap();
    for (codomain_levels, domain_levels) in [([0, 1], [2]), ([1, 0], [2]), ([2, 0], [1])] {
        for (codomain, domain) in [([1, 0], [2]), ([0, 2], [1]), ([2, 1], [0])] {
            let operation =
                TreeTransformOperation::braid(codomain, domain, codomain_levels, domain_levels);
            assert_matches_per_pair(&provider, &operation, &structure);
        }
    }
    for operation in all_transposes(2, 1) {
        assert_matches_per_pair(&provider, &operation, &structure);
    }
}

/// What: an identity operation emits coefficient exactly one without any
/// F, R or dimension query.
#[test]
fn checked_composer_identity_is_exact_without_symbol_queries() {
    let rule = AnyonicGenericRule;
    let pairs = dense_generic_source_pairs(&DenseGenericRule);
    let structure = packed_fixture_structure(
        3,
        pairs
            .iter()
            .cloned()
            .map(BlockKey::from)
            .map(|key| (key, vec![1usize; 3])),
    )
    .unwrap();
    for operation in [
        TreeTransformOperation::braid([0, 1], [2], [1, 0], [2]),
        TreeTransformOperation::transpose([0, 1], [2]),
    ] {
        let provider = CheckedPlanSpy::new(&rule);
        let plan =
            build_checked_generic_tree_pair_transform_group_plan(&provider, operation, &structure)
                .unwrap();
        assert_eq!(plan.specs().len(), pairs.len());
        for spec in plan.specs() {
            assert_eq!(spec.dst_keys(), spec.src_keys());
            assert_eq!(spec.recoupling_coefficients_dst_src(), &[1.0]);
        }
        for call in [
            CheckedPlanCall::F,
            CheckedPlanCall::R,
            CheckedPlanCall::SqrtDim,
            CheckedPlanCall::InvSqrtDim,
            CheckedPlanCall::FrobeniusSchur,
        ] {
            assert_eq!(provider.call_count(call), 0, "{call:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Physical dense oracle.
//
// Convention (TensorKit `fusiontensor`, SUNRepresentations.jl v0.4.0 gauge as
// generated by `fusion_symbols::category::GroupRepresentations::cgc`): the vertex `a ⊗ b → c` with one-based
// multiplicity label `μ` is the CGC slice `C[m_a, m_b, m_c, μ - 1]`. A
// splitting tree `a₁ … aₙ ← c` is the left-associated contraction
// `C^{a₁a₂}_{e₁,μ₁} C^{e₁a₃}_{e₂,μ₂} … C^{e_{n-2}aₙ}_{c,μ_{n-1}}`; one leg is
// `δ(m₁, m_c)`, no leg is the scalar one. A pair `(f₁, f₂)` is
// `T[m_cod, m_dom] = Σ_{m_c} X_{f₁}[m_cod, m_c] · X_{f₂}[m_dom, m_c]` (the CGCs
// are real). A permutation acts as `Y[i…] = T[p(i)…]` (TensorKit
// `permute`). symtn's SU(N) F/R symbols are themselves derived from the same
// fusion-symbols CGCs (`symtn-sectors/src/lie.rs`), so this oracle shares their gauge
// but not F, R, `GenericK`, the composer or plan assembly. Only non-dual legs
// and within-side or all-codomain permutations are claimed: dual legs and
// bends need the cup/cap (Z, Frobenius-Schur) conventions.
// ---------------------------------------------------------------------------

struct DenseTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn sun_irrep(
    category: &fusion_symbols::lie::LieGroup,
    rule: &LieGroupFusionRule,
    sector: SectorId,
) -> fusion_symbols::lie::LieIrrep {
    category
        .irrep(&rule.decode_dynkin(sector).unwrap(), &[])
        .unwrap()
}

fn cgc_dense(rule: &LieGroupFusionRule, a: SectorId, b: SectorId, c: SectorId) -> DenseTensor {
    use fusion_symbols::{
        category::{CgcBlock, GroupRepresentations},
        lie::LieGroup,
    };
    let category = LieGroup::new(rule.group_id().clone()).unwrap();
    let cgc = category
        .cgc(
            &sun_irrep(&category, rule, a),
            &sun_irrep(&category, rule, b),
            &sun_irrep(&category, rule, c),
        )
        .unwrap();
    let dims = cgc.dims();
    let mut data = Vec::with_capacity(dims.iter().product());
    for m1 in 0..dims[0] {
        for m2 in 0..dims[1] {
            for m3 in 0..dims[2] {
                for mu in 0..dims[3] {
                    data.push(cgc.at([m1, m2, m3, mu]));
                }
            }
        }
    }
    DenseTensor {
        shape: dims.to_vec(),
        data,
    }
}

fn sector_dim(rule: &LieGroupFusionRule, sector: SectorId) -> usize {
    cgc_dense(rule, sector, CheckedGenericFusion::vacuum(rule), sector).shape[0]
}

/// `X[m₁ … mₙ, m_c]` of one splitting tree.
fn tree_tensor(rule: &LieGroupFusionRule, tree: &FusionTreeKey) -> DenseTensor {
    let uncoupled = tree.uncoupled();
    assert!(tree.is_dual().iter().all(|dual| !dual));
    match uncoupled.len() {
        0 => DenseTensor {
            shape: vec![1],
            data: vec![1.0],
        },
        1 => {
            let d = sector_dim(rule, uncoupled[0]);
            let mut data = vec![0.0; d * d];
            for m in 0..d {
                data[m * d + m] = 1.0;
            }
            DenseTensor {
                shape: vec![d, d],
                data,
            }
        }
        n => {
            let outs = tree
                .innerlines()
                .iter()
                .copied()
                .chain(std::iter::once(tree.coupled()))
                .collect::<Vec<_>>();
            let mut current: Option<DenseTensor> = None;
            let mut left = uncoupled[0];
            for vertex in 0..n - 1 {
                let right = uncoupled[vertex + 1];
                let out = outs[vertex];
                let cgc = cgc_dense(rule, left, right, out);
                let mu = tree.vertices()[vertex].get() - 1;
                let [dl, dr, dout, nmu] = [cgc.shape[0], cgc.shape[1], cgc.shape[2], cgc.shape[3]];
                let slice = |ml: usize, mr: usize, mo: usize| {
                    cgc.data[((ml * dr + mr) * dout + mo) * nmu + mu]
                };
                current = Some(match current {
                    None => {
                        let mut data = vec![0.0; dl * dr * dout];
                        for ml in 0..dl {
                            for mr in 0..dr {
                                for mo in 0..dout {
                                    data[(ml * dr + mr) * dout + mo] = slice(ml, mr, mo);
                                }
                            }
                        }
                        DenseTensor {
                            shape: vec![dl, dr, dout],
                            data,
                        }
                    }
                    Some(previous) => {
                        let outer: usize =
                            previous.shape[..previous.shape.len() - 1].iter().product();
                        let mut data = vec![0.0; outer * dr * dout];
                        for o in 0..outer {
                            for ml in 0..dl {
                                let value = previous.data[o * dl + ml];
                                if value == 0.0 {
                                    continue;
                                }
                                for mr in 0..dr {
                                    for mo in 0..dout {
                                        data[(o * dr + mr) * dout + mo] +=
                                            value * slice(ml, mr, mo);
                                    }
                                }
                            }
                        }
                        let mut shape = previous.shape[..previous.shape.len() - 1].to_vec();
                        shape.extend([dr, dout]);
                        DenseTensor { shape, data }
                    }
                });
                left = out;
            }
            current.unwrap()
        }
    }
}

fn pair_tensor(rule: &LieGroupFusionRule, pair: &FusionTreePairKey) -> DenseTensor {
    let codomain = tree_tensor(rule, pair.codomain_tree());
    let domain = tree_tensor(rule, pair.domain_tree());
    let dc = *codomain.shape.last().unwrap();
    assert_eq!(dc, *domain.shape.last().unwrap());
    let (nc, nd) = (codomain.data.len() / dc, domain.data.len() / dc);
    let mut data = vec![0.0; nc * nd];
    for i in 0..nc {
        for j in 0..nd {
            data[i * nd + j] = (0..dc)
                .map(|m| codomain.data[i * dc + m] * domain.data[j * dc + m])
                .sum();
        }
    }
    let mut shape = codomain.shape[..codomain.shape.len() - 1].to_vec();
    shape.extend_from_slice(&domain.shape[..domain.shape.len() - 1]);
    DenseTensor { shape, data }
}

fn permute_dense(tensor: &DenseTensor, permutation: &[usize]) -> DenseTensor {
    let rank = tensor.shape.len();
    let shape = permutation
        .iter()
        .map(|&axis| tensor.shape[axis])
        .collect::<Vec<_>>();
    let mut in_strides = vec![1; rank];
    for axis in (0..rank.saturating_sub(1)).rev() {
        in_strides[axis] = in_strides[axis + 1] * tensor.shape[axis + 1];
    }
    let mut data = vec![0.0; tensor.data.len()];
    let mut index = vec![0; rank];
    for value in data.iter_mut() {
        let source: usize = (0..rank)
            .map(|axis| index[axis] * in_strides[permutation[axis]])
            .sum();
        *value = tensor.data[source];
        for axis in (0..rank).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    DenseTensor { shape, data }
}

/// What: on non-dual SU(3) adjoint legs (multiplicity two at `8 ⊗ 8 → 8`),
/// `Σ_dst U[dst, src] T_dst` equals the dense permuted source tensor for
/// permute and braid at ranks 2|0, 2|1, 3|1, 4|0 and within-side 2|2.
#[test]
fn checked_composer_matches_sun_dense_cgc_oracle() {
    let (rule, _, _, adjoint) = sun3();
    let a = (adjoint, false);
    let cases: Vec<(Legs, Legs, Vec<TreeTransformOperation>)> = vec![
        (
            vec![a, a],
            vec![],
            vec![TreeTransformOperation::permute([1, 0], [])],
        ),
        (
            vec![a, a],
            vec![a],
            vec![
                TreeTransformOperation::permute([1, 0], [2]),
                TreeTransformOperation::braid([1, 0], [2], [1, 0], [2]),
            ],
        ),
        (
            vec![a, a, a],
            vec![a],
            vec![
                TreeTransformOperation::permute([2, 0, 1], [3]),
                TreeTransformOperation::braid([1, 2, 0], [3], [0, 1, 2], [3]),
            ],
        ),
        (
            vec![a, a, a, a],
            vec![],
            vec![
                TreeTransformOperation::permute([1, 2, 0, 3], []),
                TreeTransformOperation::permute([3, 1, 0, 2], []),
            ],
        ),
        (
            vec![a, a],
            vec![a, a],
            vec![
                TreeTransformOperation::permute([1, 0], [2, 3]),
                TreeTransformOperation::permute([0, 1], [3, 2]),
                TreeTransformOperation::permute([1, 0], [3, 2]),
            ],
        ),
    ];
    for (codomain, domain, operations) in cases {
        let structure = sun_structure(&rule, &codomain, &domain);
        let sources = (0..structure.block_count())
            .map(|index| match structure.block(index).unwrap().key() {
                BlockKey::FusionTree(key) => key.clone(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        for operation in operations {
            let permutation = operation
                .codomain_permutation()
                .iter()
                .chain(operation.domain_permutation())
                .copied()
                .collect::<Vec<_>>();
            let plan = build_checked_generic_tree_pair_transform_group_plan(
                &rule,
                operation.clone(),
                &structure,
            )
            .unwrap();
            let columns = plan_columns(&plan);
            for source in &sources {
                let expected = permute_dense(&pair_tensor(&rule, source), &permutation);
                let column = &columns.iter().find(|(key, _)| key == source).unwrap().1;
                let mut actual = vec![0.0; expected.data.len()];
                for (destination, value) in column {
                    let basis = pair_tensor(&rule, destination);
                    assert_eq!(basis.shape, expected.shape);
                    for (out, b) in actual.iter_mut().zip(&basis.data) {
                        *out += value * b;
                    }
                }
                let residual = actual
                    .iter()
                    .zip(&expected.data)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                // Orthonormal CGCs give `‖T_f‖² = dim(coupled)`.
                let norm2 = expected.data.iter().map(|x| x * x).sum::<f64>();
                assert!(norm2 > 0.5, "nonzero source tensor");
                assert!(residual <= 1e-10, "{operation:?}: residual {residual}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Complex vertex-gauge covariance.
//
// Rescale every basis vertex `(8, 8 → 8)` with one-based label `μ` by a unit
// phase `φ(μ)`; every other vertex, in particular each one touching the
// vacuum, keeps `φ = 1`. A splitting tree rescales by `D(f) = Π_v φ_v`, a pair
// basis tensor `X_{f₁} X_{f₂}†` by `D(f₁, f₂) = D(f₁)·conj(D(f₂))`. With
// TensorKit's F-move `X_left(μ,ν) = Σ F[μ,ν,κ,λ] X_right(κ,λ)` (vertices
// `(a,b,e,μ)`, `(e,c,d,ν)` | `(b,c,f,κ)`, `(a,f,d,λ)`) and R-move
// `swap X^{ab}_{c,μ} = Σ_ν R[μ,ν] X^{ba}_{c,ν}`, the transformed symbols are
// `F' = φ_abe(μ) φ_ecd(ν) F conj(φ_bcf(κ)) conj(φ_afd(λ))` and
// `R' = φ_abc(μ) R conj(φ_bac(ν))`; the plan must transform as
// `U'[dst, src] = conj(D(dst)) · U[dst, src] · D(src)`.
// ---------------------------------------------------------------------------

struct GaugedSun {
    inner: LieGroupFusionRule,
    adjoint: SectorId,
    phases: [Complex64; 2],
}

impl GaugedSun {
    fn phase(&self, a: SectorId, b: SectorId, c: SectorId, mu: usize) -> Complex64 {
        if [a, b, c] == [self.adjoint; 3] {
            self.phases[mu]
        } else {
            Complex64::new(1.0, 0.0)
        }
    }

    fn tree_gauge(&self, tree: &FusionTreeKey) -> Complex64 {
        let uncoupled = tree.uncoupled();
        if uncoupled.len() < 2 {
            return Complex64::new(1.0, 0.0);
        }
        let outs = tree
            .innerlines()
            .iter()
            .copied()
            .chain(std::iter::once(tree.coupled()))
            .collect::<Vec<_>>();
        let mut left = uncoupled[0];
        let mut gauge = Complex64::new(1.0, 0.0);
        for (vertex, &out) in outs.iter().enumerate() {
            gauge *= self.phase(
                left,
                uncoupled[vertex + 1],
                out,
                tree.vertices()[vertex].get() - 1,
            );
            left = out;
        }
        gauge
    }

    fn pair_gauge(&self, pair: &FusionTreePairKey) -> Complex64 {
        self.tree_gauge(pair.codomain_tree()) * self.tree_gauge(pair.domain_tree()).conj()
    }
}

impl symtn_core::testing::Provider for GaugedSun {}

impl CheckedGenericFusion for GaugedSun {
    type Error = LieGroupFusionRuleError;
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }
    fn try_dual(&self, a: SectorId) -> Result<SectorId, Self::Error> {
        self.inner.try_dual(a)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        self.inner.try_fusion_channels(a, b)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.inner.try_fusion_channels_in_table(a, b)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.inner.try_nsymbol(a, b, c)
    }
}

impl CheckedGenericRigidSymbols for GaugedSun {
    type Scalar = Complex64;
    fn try_dim_scalar(&self, a: SectorId) -> Result<Complex64, Self::Error> {
        self.inner.try_dim_scalar(a).map(Complex64::from)
    }
    fn try_sqrt_dim_scalar(&self, a: SectorId) -> Result<Complex64, Self::Error> {
        self.inner.try_sqrt_dim_scalar(a).map(Complex64::from)
    }
    fn try_inv_sqrt_dim_scalar(&self, a: SectorId) -> Result<Complex64, Self::Error> {
        self.inner.try_inv_sqrt_dim_scalar(a).map(Complex64::from)
    }
    fn try_frobenius_schur_phase_scalar(&self, a: SectorId) -> Result<Complex64, Self::Error> {
        self.inner
            .try_frobenius_schur_phase_scalar(a)
            .map(Complex64::from)
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Complex64>, Self::Error> {
        let block = self.inner.try_f_symbol_generic(a, b, c, d, e, f)?;
        let (n_mu, n_nu, n_kappa, n_lambda) = block.shape();
        let mut data = Vec::with_capacity(block.data().len());
        for mu in 0..n_mu {
            for nu in 0..n_nu {
                for kappa in 0..n_kappa {
                    for lambda in 0..n_lambda {
                        data.push(
                            self.phase(a, b, e, mu)
                                * self.phase(e, c, d, nu)
                                * *block.get(mu, nu, kappa, lambda)
                                * self.phase(b, c, f, kappa).conj()
                                * self.phase(a, f, d, lambda).conj(),
                        );
                    }
                }
            }
        }
        Ok(GenericFArray::new(data, (n_mu, n_nu, n_kappa, n_lambda)))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Complex64>, Self::Error> {
        let block = self.inner.try_r_symbol_generic(a, b, c)?;
        let (rows, cols) = block.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for mu in 0..rows {
            for nu in 0..cols {
                data.push(
                    self.phase(a, b, c, mu) * *block.get(mu, nu) * self.phase(b, a, c, nu).conj(),
                );
            }
        }
        Ok(GenericRMatrix::new(data, rows, cols))
    }
}

/// What: complex coefficients through the whole checked composer, including
/// bends, folds and both cycle directions, transform covariantly under a
/// nontrivial vertex gauge of the multiplicity-two SU(3) vertex.
#[test]
fn checked_composer_is_covariant_under_complex_vertex_gauge() {
    let (rule, _, _, adjoint) = sun3();
    let gauged = GaugedSun {
        inner: LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        adjoint,
        phases: [
            Complex64::from_polar(1.0, 0.7),
            Complex64::from_polar(1.0, -1.9),
        ],
    };
    let a = (adjoint, false);
    for (codomain, domain) in [
        (vec![a, a], vec![a]),
        (vec![a, a], vec![a, a]),
        (vec![a, a, a, a], vec![]),
    ] {
        let structure = sun_structure(&rule, &codomain, &domain);
        let (p, q) = (codomain.len(), domain.len());
        let mut operations = all_transposes(p, q);
        operations.push(TreeTransformOperation::permute(
            (0..p).rev().collect::<Vec<_>>(),
            (p..p + q).collect::<Vec<_>>(),
        ));
        operations.push(TreeTransformOperation::permute(
            (1..p + q).collect::<Vec<_>>(),
            [0],
        ));
        for operation in operations {
            let real = plan_columns(
                &build_checked_generic_tree_pair_transform_group_plan(
                    &rule,
                    operation.clone(),
                    &structure,
                )
                .unwrap(),
            );
            let complex = plan_columns(
                &build_checked_generic_tree_pair_transform_group_plan(
                    &gauged,
                    operation.clone(),
                    &structure,
                )
                .unwrap(),
            );
            assert_eq!(real.len(), complex.len());
            for (source, column) in &complex {
                let real_column = &real.iter().find(|(key, _)| key == source).unwrap().1;
                for destination in column
                    .iter()
                    .map(|(key, _)| key)
                    .chain(real_column.iter().map(|(key, _)| key))
                {
                    let expected = gauged.pair_gauge(destination).conj()
                        * coefficient(real_column, destination)
                        * gauged.pair_gauge(source);
                    let actual = coefficient(column, destination);
                    assert!(
                        actual.distance(expected) <= TOL,
                        "{operation:?}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}

/// Supplementary (computed by the code under test): a transpose followed by
/// its inverse is the identity on every source.
#[test]
fn checked_composer_transpose_round_trips_are_identity() {
    let (rule, _, three, adjoint) = sun3();
    let a = (adjoint, false);
    for (codomain, domain) in [
        (vec![a, a], vec![a, a]),
        (vec![(three, false), (three, true)], vec![(three, false)]),
    ] {
        let structure = sun_structure(&rule, &codomain, &domain);
        let (p, q) = (codomain.len(), domain.len());
        for forward in all_transposes(p, q) {
            let (cod, dom) = (
                forward.codomain_permutation().to_vec(),
                forward.domain_permutation().to_vec(),
            );
            let linear = cod.iter().chain(dom.iter()).copied().collect::<Vec<_>>();
            let mut inverse = vec![0; linear.len()];
            for (position, &axis) in linear.iter().enumerate() {
                inverse[axis] = position;
            }
            let backward =
                TreeTransformOperation::transpose(inverse[..p].to_vec(), inverse[p..].to_vec());
            let first = build_checked_generic_tree_pair_transform_group_plan(
                &rule,
                forward.clone(),
                &structure,
            )
            .unwrap();
            let middle = packed_fixture_structure(
                p + q,
                first
                    .specs()
                    .iter()
                    .flat_map(|spec| spec.dst_keys().iter().cloned())
                    .map(|key| (key, vec![1; p + q])),
            )
            .unwrap();
            let second =
                build_checked_generic_tree_pair_transform_group_plan(&rule, backward, &middle)
                    .unwrap();
            let (first, second) = (plan_columns(&first), plan_columns(&second));
            for (source, column) in &first {
                let mut composed: Vec<(FusionTreePairKey, f64)> = Vec::new();
                for (middle_key, value) in column {
                    let back = &second.iter().find(|(key, _)| key == middle_key).unwrap().1;
                    for (destination, back_value) in back {
                        match composed.iter_mut().find(|(key, _)| key == destination) {
                            Some((_, total)) => *total += value * back_value,
                            None => composed.push((destination.clone(), value * back_value)),
                        }
                    }
                }
                for (destination, value) in composed {
                    let expected = if &destination == source { 1.0 } else { 0.0 };
                    assert!((value - expected).abs() <= 1e-10, "{forward:?}");
                }
            }
        }
    }
}

mod contract_oracles;
