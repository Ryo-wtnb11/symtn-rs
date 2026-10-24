//! Independent oracles for checked Generic contraction through the shared
//! planner (#1860), one per route the ladder takes: the Core rung (both
//! orientations), TensorKit's `copyC`, and the `DynamicTree` artifact with a
//! transformed lhs only (the rhs-sorted candidate), a transformed rhs, a
//! `RhsLhs` winner and an output transform.
//!
//! - Physical dense oracle (non-dual legs): every operand and the result
//!   are expanded with SU(3) fusion-symbols CGC splitting tensors (outer
//!   multiplicity two on `8 ⊗ 8 → 8`), the operands are contracted as dense
//!   arrays, and the result must equal the expanded output. Legs carry
//!   several sectors with degeneracies, so the results span several
//!   coupled sectors, recoupling groups and repeated (Multi) destinations.
//! - Reference-step composition (dual legs): TensorKit's `blas_contract!`
//!   steps run as separate checked primitives (permute each operand to core
//!   form, compose, permute the output) and must equal the contraction.

use super::*;
use crate::contract::{BoundDynamicFusionMapSpace, TensorContractFusionExecutionContext};
use symtn_operations::{OutputAxisOrder, TensorContractSpec};

type Leg = Vec<(SectorId, usize)>;
type Bound = BoundDynamicFusionMapSpace<LieGroupFusionRule>;

trait Value: Copy + std::fmt::Debug {
    fn make(index: usize, seed: f64) -> Self;
    fn widen(self) -> Complex64;
}

impl Value for f64 {
    fn make(index: usize, seed: f64) -> Self {
        ((index * 7 + 3) % 17) as f64 * 0.125 - 1.0 + seed
    }
    fn widen(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Value for Complex64 {
    fn make(index: usize, seed: f64) -> Self {
        Complex64::new(
            f64::make(index, seed),
            ((index * 5 + 1) % 11) as f64 * 0.25 - 1.25,
        )
    }
    fn widen(self) -> Complex64 {
        self
    }
}

fn space(
    rule: &Arc<LieGroupFusionRule>,
    codomain: &[(&Leg, bool)],
    domain: &[(&Leg, bool)],
) -> Bound {
    let legs = |legs: &[(&Leg, bool)]| {
        FusionProductSpace::new(
            legs.iter()
                .map(|(leg, dual)| SectorLeg::new(leg.iter().copied(), *dual))
                .collect::<Vec<_>>(),
        )
    };
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(rule),
        FusionTreeHomSpace::new(legs(codomain), legs(domain)),
    )
    .unwrap()
}

fn payload<D: Value>(space: &Bound, seed: f64) -> Vec<D> {
    (0..space.space().required_len().unwrap())
        .map(|index| D::make(index, seed))
        .collect()
}

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// The multi-indices of `shape` in row-major order.
fn indices(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    let total = shape.iter().product::<usize>();
    (0..total).map(move |mut linear| {
        let mut index = vec![0; shape.len()];
        for axis in (0..shape.len()).rev() {
            index[axis] = linear % shape[axis];
            linear /= shape[axis];
        }
        index
    })
}

/// The physical dense tensor (row-major over the legs, each leg ordered by
/// sector, then degeneracy, then multiplet) of a non-dual checked tensor.
fn expand<D: Value>(
    rule: &LieGroupFusionRule,
    legs: &[&Leg],
    space: &Bound,
    data: &[D],
) -> Vec<Complex64> {
    let segment = |leg: &Leg, sector: SectorId| {
        let mut start = 0;
        for &(candidate, degeneracy) in leg {
            if candidate == sector {
                return start;
            }
            start += sector_dim(rule, candidate) * degeneracy;
        }
        unreachable!("block sector lies on its leg")
    };
    let shape = legs
        .iter()
        .map(|leg| {
            leg.iter()
                .map(|&(sector, degeneracy)| sector_dim(rule, sector) * degeneracy)
                .sum::<usize>()
        })
        .collect::<Vec<_>>();
    let dense_strides = strides(&shape);
    let mut dense = vec![Complex64::new(0.0, 0.0); shape.iter().product()];
    let structure = space.space().structure();
    for block in 0..structure.block_count() {
        let block = structure.block(block).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            unreachable!("checked spaces are fusion-tree keyed")
        };
        let basis = pair_tensor(rule, key);
        let sectors = key
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(key.domain_tree().uncoupled())
            .copied()
            .collect::<Vec<_>>();
        let starts = sectors
            .iter()
            .zip(legs)
            .map(|(&sector, leg)| segment(leg, sector))
            .collect::<Vec<_>>();
        let dims = sectors
            .iter()
            .map(|&sector| sector_dim(rule, sector))
            .collect::<Vec<_>>();
        for degeneracy in indices(block.shape()) {
            let at = block.offset()
                + degeneracy
                    .iter()
                    .zip(block.strides())
                    .map(|(k, stride)| k * stride)
                    .sum::<usize>();
            let value = data[at].widen();
            for (multiplet, &coefficient) in indices(&dims).zip(&basis.data) {
                if coefficient == 0.0 {
                    continue;
                }
                let position = (0..legs.len())
                    .map(|axis| {
                        (starts[axis] + degeneracy[axis] * dims[axis] + multiplet[axis])
                            * dense_strides[axis]
                    })
                    .sum::<usize>();
                dense[position] += value * coefficient;
            }
        }
    }
    dense
}

/// `out[i] = Σ_c lhs[open_l, c] rhs[c, open_r]` over row-major dense arrays,
/// then permuted to `output`.
#[allow(clippy::too_many_arguments)]
fn dense_contract(
    lhs: &[Complex64],
    lhs_shape: &[usize],
    rhs: &[Complex64],
    rhs_shape: &[usize],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output: &[usize],
) -> Vec<Complex64> {
    let lhs_open = (0..lhs_shape.len())
        .filter(|axis| !lhs_axes.contains(axis))
        .collect::<Vec<_>>();
    let rhs_open = (0..rhs_shape.len())
        .filter(|axis| !rhs_axes.contains(axis))
        .collect::<Vec<_>>();
    let open_shape = lhs_open
        .iter()
        .map(|&axis| lhs_shape[axis])
        .chain(rhs_open.iter().map(|&axis| rhs_shape[axis]))
        .collect::<Vec<_>>();
    let contracted_shape = lhs_axes
        .iter()
        .map(|&axis| lhs_shape[axis])
        .collect::<Vec<_>>();
    let (lhs_strides, rhs_strides) = (strides(lhs_shape), strides(rhs_shape));
    let out_shape = output
        .iter()
        .map(|&axis| open_shape[axis])
        .collect::<Vec<_>>();
    let out_strides = strides(&out_shape);
    let mut out = vec![Complex64::new(0.0, 0.0); out_shape.iter().product()];
    for open in indices(&open_shape) {
        let (lhs_open_index, rhs_open_index) = open.split_at(lhs_open.len());
        let lhs_base = lhs_open
            .iter()
            .zip(lhs_open_index)
            .map(|(&axis, &i)| i * lhs_strides[axis])
            .sum::<usize>();
        let rhs_base = rhs_open
            .iter()
            .zip(rhs_open_index)
            .map(|(&axis, &i)| i * rhs_strides[axis])
            .sum::<usize>();
        let mut sum = Complex64::new(0.0, 0.0);
        for contracted in indices(&contracted_shape) {
            let l = lhs_base
                + lhs_axes
                    .iter()
                    .zip(&contracted)
                    .map(|(&axis, &i)| i * lhs_strides[axis])
                    .sum::<usize>();
            let r = rhs_base
                + rhs_axes
                    .iter()
                    .zip(&contracted)
                    .map(|(&axis, &i)| i * rhs_strides[axis])
                    .sum::<usize>();
            sum += lhs[l] * rhs[r];
        }
        let position = output
            .iter()
            .enumerate()
            .map(|(slot, &axis)| open[axis] * out_strides[slot])
            .sum::<usize>();
        out[position] = sum;
    }
    out
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    let scale = expected.iter().map(|v| v.norm()).fold(1.0, f64::max);
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).norm())
        .fold(0.0, f64::max);
    assert!(residual <= 1e-12 * scale, "{what}: residual {residual}");
    assert!(
        expected.iter().any(|v| v.norm() > 0.5),
        "{what}: nonzero oracle"
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Route {
    Core,
    CopyC,
    DynamicTree,
}

struct Case {
    name: &'static str,
    lhs: [Vec<usize>; 2],
    rhs: [Vec<usize>; 2],
    lhs_axes: &'static [usize],
    rhs_axes: &'static [usize],
    output: &'static [usize],
    codomain_rank: usize,
    route: Route,
    orientation: Option<crate::contract::FusionContractOrientation>,
    /// Completed transformers a cold call builds.
    transforms: usize,
}

fn run_case<D>(rule: &Arc<LieGroupFusionRule>, legs: &[Leg], case: &Case)
where
    D: Value
        + crate::DenseRecouplingScalar
        + crate::RecouplingCoefficientAction<f64>
        + crate::ConjugateValue
        + num_traits::Zero
        + crate::ZeroBytes,
{
    let pick = |indices: &[usize]| {
        indices
            .iter()
            .map(|&index| (&legs[index], false))
            .collect::<Vec<_>>()
    };
    let lhs = space(rule, &pick(&case.lhs[0]), &pick(&case.lhs[1]));
    let rhs = space(rule, &pick(&case.rhs[0]), &pick(&case.rhs[1]));
    let (lhs_data, rhs_data) = (payload::<D>(&lhs, 0.0), payload::<D>(&rhs, 0.5));
    let mut context = TensorContractFusionExecutionContext::<D, RuleIdentity>::default();
    crate::probe::take_completed_transformer_activity();
    let (output, data) = context
        .tensorcontract_checked_generic_in(
            direct_side(&lhs, &lhs_data),
            direct_side(&rhs, &rhs_data),
            entry_axes(TensorContractSpec::new(
                case.lhs_axes,
                case.rhs_axes,
                OutputAxisOrder::Axes(case.output),
            )),
            case.codomain_rank,
        )
        .unwrap();
    let route = if context.last_resolution_is_core() {
        Route::Core
    } else if context.last_resolution_orientation().is_some() {
        Route::DynamicTree
    } else {
        Route::CopyC
    };
    let what = format!("{} {}", case.name, std::any::type_name::<D>());
    assert_eq!(route, case.route, "{what}");
    if let Some(orientation) = case.orientation {
        assert_eq!(
            context.last_resolution_orientation(),
            Some(orientation),
            "{what}"
        );
    }
    let built = crate::probe::take_completed_transformer_activity().builds;
    assert!(built <= case.transforms, "{what}: {built} transforms");

    let all = |side: &[Vec<usize>; 2]| {
        side[0]
            .iter()
            .chain(&side[1])
            .map(|&index| &legs[index])
            .collect::<Vec<_>>()
    };
    let (lhs_legs, rhs_legs) = (all(&case.lhs), all(&case.rhs));
    let shape = |legs: &[&Leg]| {
        legs.iter()
            .map(|leg| {
                leg.iter()
                    .map(|&(sector, degeneracy)| sector_dim(rule, sector) * degeneracy)
                    .sum::<usize>()
            })
            .collect::<Vec<_>>()
    };
    let expected = dense_contract(
        &expand(rule, &lhs_legs, &lhs, &lhs_data),
        &shape(&lhs_legs),
        &expand(rule, &rhs_legs, &rhs, &rhs_data),
        &shape(&rhs_legs),
        case.lhs_axes,
        case.rhs_axes,
        case.output,
    );
    let open_legs = lhs_legs
        .iter()
        .enumerate()
        .filter(|(axis, _)| !case.lhs_axes.contains(axis))
        .map(|(_, leg)| *leg)
        .chain(
            rhs_legs
                .iter()
                .enumerate()
                .filter(|(axis, _)| !case.rhs_axes.contains(axis))
                .map(|(_, leg)| *leg),
        )
        .collect::<Vec<_>>();
    let output_legs = case
        .output
        .iter()
        .map(|&axis| open_legs[axis])
        .collect::<Vec<_>>();
    let actual = expand(rule, &output_legs, &output, &data);
    assert_close(&actual, &expected, &what);
}

/// What: on SU(3) with outer multiplicity, each route of the checked
/// contraction ladder equals the dense contraction of the CGC-expanded
/// operands, for real and complex payloads, over legs with several sectors
/// and degeneracies (several coupled sectors and recoupling groups; the
/// output transforms accumulate into repeated destinations).
#[test]
fn checked_contraction_routes_match_sun_dense_cgc_oracle() {
    use crate::contract::FusionContractOrientation::{LhsRhs, RhsLhs};
    let (rule, vacuum, _, adjoint) = sun3();
    let rule = Arc::new(rule);
    // Leg 0: 1 ⊕ 2·8; leg 1: 8 (a smaller operand leg).
    let legs = [vec![(vacuum, 1), (adjoint, 2)], vec![(adjoint, 1)]];
    let square = || [vec![0, 0], vec![0, 0]];
    let cases = [
        Case {
            name: "core",
            lhs: square(),
            rhs: square(),
            lhs_axes: &[2, 3],
            rhs_axes: &[0, 1],
            output: &[0, 1, 2, 3],
            codomain_rank: 2,
            route: Route::Core,
            orientation: None,
            transforms: 0,
        },
        Case {
            name: "core B·A",
            lhs: square(),
            rhs: square(),
            lhs_axes: &[0, 1],
            rhs_axes: &[2, 3],
            output: &[2, 3, 0, 1],
            codomain_rank: 2,
            route: Route::Core,
            orientation: Some(RhsLhs),
            transforms: 0,
        },
        Case {
            name: "copyC",
            lhs: square(),
            rhs: square(),
            lhs_axes: &[2, 3],
            rhs_axes: &[0, 1],
            output: &[1, 0, 2, 3],
            codomain_rank: 2,
            route: Route::CopyC,
            orientation: None,
            transforms: 1,
        },
        Case {
            name: "lhs-sorted, rhs transformed",
            lhs: square(),
            rhs: square(),
            lhs_axes: &[3, 2],
            rhs_axes: &[0, 1],
            output: &[0, 1, 2, 3],
            codomain_rank: 2,
            route: Route::DynamicTree,
            orientation: Some(LhsRhs),
            transforms: 1,
        },
        Case {
            name: "rhs-sorted, lhs transformed",
            lhs: [vec![1, 1], vec![1, 1]],
            rhs: [vec![1, 1], vec![0, 0]],
            lhs_axes: &[3, 2],
            rhs_axes: &[0, 1],
            output: &[0, 1, 2, 3],
            codomain_rank: 2,
            route: Route::DynamicTree,
            orientation: Some(LhsRhs),
            transforms: 1,
        },
        Case {
            name: "B·A winner",
            lhs: square(),
            rhs: square(),
            lhs_axes: &[1, 0],
            rhs_axes: &[2, 3],
            output: &[2, 3, 0, 1],
            codomain_rank: 2,
            route: Route::DynamicTree,
            orientation: Some(RhsLhs),
            transforms: 1,
        },
        Case {
            name: "output transform",
            lhs: square(),
            rhs: square(),
            lhs_axes: &[3, 1],
            rhs_axes: &[0, 3],
            output: &[2, 0, 3, 1],
            codomain_rank: 2,
            route: Route::DynamicTree,
            orientation: None,
            transforms: 3,
        },
    ];
    for case in &cases {
        run_case::<f64>(&rule, &legs, case);
        run_case::<Complex64>(&rule, &legs, case);
    }
}

/// What: with dual contracted and open legs, the checked contraction equals
/// TensorKit `blas_contract!`'s steps run as separate checked primitives:
/// `permute(A, open | contracted)`, `permute(B, contracted | open)`, their
/// composition (`mul!`), then the output permute.
#[test]
fn checked_contraction_with_dual_legs_matches_reference_step_composition() {
    let (rule, vacuum, three, adjoint) = sun3();
    let rule = Arc::new(rule);
    let v: Leg = vec![(vacuum, 1), (three, 2), (adjoint, 1)];
    // `V*`: the dual sectors on a dual leg.
    let dual_v: Leg = v
        .iter()
        .map(|&(sector, degeneracy)| {
            (
                CheckedGenericFusion::try_dual(&*rule, sector).unwrap(),
                degeneracy,
            )
        })
        .collect();
    // `(name, lhs codomain | domain, rhs codomain | domain, lhs axes, rhs
    // axes, output, codomain rank)`; `true` marks a dual leg.
    type DualCase = (
        &'static str,
        [&'static [bool]; 2],
        [&'static [bool]; 2],
        &'static [usize],
        &'static [usize],
        &'static [usize],
        usize,
    );
    let cases: [DualCase; 3] = [
        (
            "dual contracted pair",
            [&[false, true], &[false]],
            [&[false, false], &[true]],
            &[1, 2],
            &[0, 1],
            &[0, 1],
            1,
        ),
        (
            "dual open legs, output transform",
            [&[false, true], &[true]],
            [&[true, true], &[false]],
            &[2],
            &[1],
            &[2, 0, 1, 3],
            2,
        ),
        (
            "dual core",
            [&[false, true], &[true]],
            [&[true], &[false, true]],
            &[2],
            &[0],
            &[0, 1, 2, 3],
            2,
        ),
    ];
    for (name, lhs_duals, rhs_duals, lhs_axes, rhs_axes, output, codomain_rank) in cases {
        let side = |duals: &[bool]| {
            duals
                .iter()
                .map(|&dual| if dual { (&dual_v, true) } else { (&v, false) })
                .collect::<Vec<_>>()
        };
        let lhs = space(&rule, &side(lhs_duals[0]), &side(lhs_duals[1]));
        let rhs = space(&rule, &side(rhs_duals[0]), &side(rhs_duals[1]));
        let lhs_data = payload::<Complex64>(&lhs, 0.0);
        let rhs_data = payload::<Complex64>(&rhs, 0.5);
        let mut context =
            TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
        let (output_space, data) = context
            .tensorcontract_checked_generic_in(
                direct_side(&lhs, &lhs_data),
                direct_side(&rhs, &rhs_data),
                entry_axes(TensorContractSpec::new(
                    lhs_axes,
                    rhs_axes,
                    OutputAxisOrder::Axes(output),
                )),
                codomain_rank,
            )
            .unwrap();

        let open = |rank: usize, axes: &[usize]| {
            (0..rank)
                .filter(|axis| !axes.contains(axis))
                .collect::<Vec<_>>()
        };
        let one = Complex64::new(1.0, 0.0);
        let lhs_open = open(lhs.space().rank(), lhs_axes);
        let (lhs_core, lhs_core_data) = crate::TreeTransformExecutionContext::<
            Complex64,
            RuleIdentity,
            f64,
        >::default()
        .tree_transform_owned_checked_generic_in(
            &lhs,
            None,
            &lhs_data,
            &TreeTransformOperation::permute(lhs_open.iter().copied(), lhs_axes.iter().copied()),
            one,
        )
        .unwrap();
        let rhs_open = open(rhs.space().rank(), rhs_axes);
        let (rhs_core, rhs_core_data) = crate::TreeTransformExecutionContext::<
            Complex64,
            RuleIdentity,
            f64,
        >::default()
        .tree_transform_owned_checked_generic_in(
            &rhs,
            None,
            &rhs_data,
            &TreeTransformOperation::permute(rhs_axes.iter().copied(), rhs_open.iter().copied()),
            one,
        )
        .unwrap();
        let (core, core_data) = context
            .tensorcompose_checked_generic_in(
                direct_side(&lhs_core, &lhs_core_data),
                direct_side(&rhs_core, &rhs_core_data),
            )
            .unwrap();
        let (codomain, domain) = output.split_at(codomain_rank);
        let (expected_space, expected) = crate::TreeTransformExecutionContext::<
            Complex64,
            RuleIdentity,
            f64,
        >::default()
        .tree_transform_owned_checked_generic_in(
            &core,
            None,
            &core_data,
            &TreeTransformOperation::permute(codomain.iter().copied(), domain.iter().copied()),
            one,
        )
        .unwrap();
        assert_eq!(
            output_space.space().homspace(),
            expected_space.space().homspace(),
            "{name}"
        );
        assert_eq!(
            output_space.space().structure().as_ref(),
            expected_space.space().structure().as_ref(),
            "{name}"
        );
        assert_close(&data, &expected, name);
    }
}
