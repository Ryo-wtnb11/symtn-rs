//! Planarity of a pairwise contraction: TensorKit 0.17.1 `@planar`'s binary
//! step (`src/planar/analyzers.jl`, `preprocessors.jl:_check_planarity`) and
//! `planarcontract!`'s leg reordering (`planaroperations.jl:reorder_indices`).
//!
//! A tensor's cyclic boundary is its codomain axes in order, then its domain
//! axes reversed (TensorKit `(codind..., reverse(domind)...)`, the convention
//! of `TensorMap::transpose`). A contraction is planar when each operand's
//! contracted axes are one arc of its boundary, the pairing walks the two
//! arcs in opposite directions, and the requested output boundary is a
//! rotation of the boundary the two open arcs leave.

use symtn_operations::{AxisVec, OutputAxisOrder, TensorContractSpec};

use super::structure::TensorContractAxisPlan;
use crate::OperationError;

/// The transposes of an admitted planar contraction, TensorKit
/// `reorder_indices`: the left operand onto `(open | contracted)`, the right
/// onto `(contracted | open)`, both along their boundaries with the
/// contracted axes paired in order, and the core result
/// `left open ← right open` onto the requested `(codomain, domain)`.
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarContractOrders {
    pub(crate) lhs: [AxisVec; 2],
    pub(crate) rhs: [AxisVec; 2],
    pub(crate) output: [AxisVec; 2],
}

/// `codomain ++ reverse(domain)` of a `nout`-codomain, rank-`rank` tensor.
fn boundary(nout: usize, rank: usize) -> AxisVec {
    (0..nout).chain((nout..rank).rev()).collect()
}

/// `boundary`'s inverse: the boundary position of every axis.
fn positions(boundary: &[usize]) -> AxisVec {
    let mut positions = AxisVec::from_elem(0, boundary.len());
    for (position, &axis) in boundary.iter().enumerate() {
        positions[axis] = position;
    }
    positions
}

fn is_identity(order: &[AxisVec; 2], nout: usize, rank: usize) -> bool {
    order[0].iter().copied().eq(0..nout) && order[1].iter().copied().eq(nout..rank)
}

/// Whether `cycle` read from some start is `sequence`.
fn is_rotation(cycle: &[usize], sequence: &[usize]) -> bool {
    let n = cycle.len();
    n == sequence.len()
        && (n == 0 || (0..n).any(|start| (0..n).all(|i| cycle[(start + i) % n] == sequence[i])))
}

impl PlanarContractOrders {
    /// Admits `spec` between a `lhs = (nout, rank)` and a `rhs = (nout, rank)`
    /// operand (their logical shapes) as a planar contraction whose output is
    /// `output_axes` split after `codomain_rank`, in `TensorMap::contract`'s
    /// leg roles (open legs numbered left-ascending then right-ascending).
    ///
    /// A malformed spec reports exactly the ordinary contraction's error. A
    /// contracted set that is not one boundary arc is
    /// [`OperationError::InvalidAxisSet`] of that side; a pairing that does
    /// not reverse orientation is `InvalidAxisSet` of `"rhs"`; an output that
    /// is not a rotation of the open boundary is
    /// [`OperationError::InvalidPermutation`] of `output_axes`. TensorKit
    /// asserts the same conditions (`reorder_indices`) and `@planar` throws
    /// `ArgumentError`.
    ///
    /// Where the boundary leaves a choice (a contraction over a whole
    /// boundary, or none), the orders with the most identity transposes are
    /// chosen, so a composition-shaped spec is exactly a composition.
    pub fn validate(
        (lhs_nout, lhs_rank): (usize, usize),
        (rhs_nout, rhs_rank): (usize, usize),
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Self, OperationError> {
        TensorContractAxisPlan::compile_unobserved(
            lhs_rank,
            rhs_rank,
            output_axes.len(),
            TensorContractSpec::new(lhs_axes, rhs_axes, OutputAxisOrder::from_axes(output_axes)),
        )?;
        let contracted = lhs_axes.len();
        let (lhs_boundary, rhs_boundary) =
            (boundary(lhs_nout, lhs_rank), boundary(rhs_nout, rhs_rank));
        let rhs_position = positions(&rhs_boundary);
        let mut partner = AxisVec::from_elem(usize::MAX, lhs_rank);
        for (&lhs, &rhs) in lhs_axes.iter().zip(rhs_axes) {
            partner[lhs] = rhs;
        }

        // P1: the left contracted axes are one arc. Its start is unique
        // unless the arc is empty or the whole boundary.
        let starts: AxisVec = if contracted == 0 || contracted == lhs_rank {
            (0..lhs_rank.max(1)).collect()
        } else {
            let on_arc = |position: usize| partner[lhs_boundary[position % lhs_rank]] != usize::MAX;
            let starts: AxisVec = (0..lhs_rank)
                .filter(|&p| on_arc(p) && !on_arc(p + lhs_rank - 1))
                .collect();
            if starts.len() != 1 {
                return Err(OperationError::InvalidAxisSet {
                    tensor: "lhs",
                    axes: lhs_axes.to_vec(),
                    rank: lhs_rank,
                });
            }
            starts
        };
        // P2: walking the left arc forward visits the partners backward
        // along the right boundary, which also makes them one arc there.
        // `q0` is the partner position of the arc's first axis.
        let reversed = |start: usize| -> Option<usize> {
            if contracted == 0 {
                return Some(0);
            }
            let partner_position =
                |i: usize| rhs_position[partner[lhs_boundary[(start + i) % lhs_rank]]];
            let q0 = partner_position(0);
            (1..contracted)
                .all(|i| partner_position(i) == (q0 + rhs_rank * contracted - i) % rhs_rank)
                .then_some(q0)
        };
        let starts: smallvec::SmallVec<[(usize, usize); 8]> = starts
            .iter()
            .filter_map(|&start| reversed(start).map(|q0| (start, q0)))
            .collect();
        if starts.is_empty() {
            return Err(OperationError::InvalidAxisSet {
                tensor: "rhs",
                axes: rhs_axes.to_vec(),
                rank: rhs_rank,
            });
        }

        // P3: the requested boundary is a rotation of the open arcs'.
        let open_rank = output_axes.len();
        let mut label = AxisVec::from_elem(usize::MAX, lhs_rank + rhs_rank);
        let mut next = 0;
        for axis in 0..lhs_rank {
            if partner[axis] == usize::MAX {
                label[axis] = next;
                next += 1;
            }
        }
        for axis in 0..rhs_rank {
            if !rhs_axes.contains(&axis) {
                label[lhs_rank + axis] = next;
                next += 1;
            }
        }
        let (codomain, domain) = output_axes.split_at(codomain_rank);
        let requested: AxisVec = codomain
            .iter()
            .chain(domain.iter().rev())
            .copied()
            .collect();
        let rhs_open_starts: AxisVec = if contracted == 0 {
            (0..rhs_rank.max(1)).collect()
        } else {
            AxisVec::new()
        };
        let mut best: Option<(usize, Self)> = None;
        for &(start, q0) in &starts {
            let rhs_starts: &[usize] = if contracted == 0 {
                &rhs_open_starts
            } else {
                &[q0 + 1]
            };
            for &rhs_start in rhs_starts {
                let lhs_open: AxisVec = (0..lhs_rank - contracted)
                    .map(|i| lhs_boundary[(start + contracted + i) % lhs_rank])
                    .collect();
                let rhs_open: AxisVec = (0..rhs_rank - contracted)
                    .map(|i| rhs_boundary[(rhs_start + i) % rhs_rank])
                    .collect();
                let open_boundary: AxisVec = lhs_open
                    .iter()
                    .map(|&axis| label[axis])
                    .chain(rhs_open.iter().map(|&axis| label[lhs_rank + axis]))
                    .collect();
                if !is_rotation(&requested, &open_boundary) {
                    continue;
                }
                let lhs_contracted: AxisVec = (0..contracted)
                    .rev()
                    .map(|i| lhs_boundary[(start + i) % lhs_rank])
                    .collect();
                let rhs_contracted: AxisVec =
                    lhs_contracted.iter().map(|&axis| partner[axis]).collect();
                let rhs_open: AxisVec = rhs_open.into_iter().rev().collect();
                let mut core = AxisVec::from_elem(0, open_rank);
                for (position, &semantic) in lhs_open
                    .iter()
                    .map(|&axis| label[axis])
                    .chain(rhs_open.iter().map(|&axis| label[lhs_rank + axis]))
                    .collect::<AxisVec>()
                    .iter()
                    .enumerate()
                {
                    core[semantic] = position;
                }
                let lhs_open_rank = lhs_open.len();
                let orders = Self {
                    lhs: [lhs_open, lhs_contracted],
                    rhs: [rhs_contracted, rhs_open],
                    output: [
                        codomain.iter().map(|&semantic| core[semantic]).collect(),
                        domain.iter().map(|&semantic| core[semantic]).collect(),
                    ],
                };
                let identities = usize::from(is_identity(&orders.lhs, lhs_nout, lhs_rank))
                    + usize::from(is_identity(&orders.rhs, rhs_nout, rhs_rank))
                    + usize::from(
                        codomain_rank == lhs_open_rank
                            && is_identity(&orders.output, codomain_rank, open_rank),
                    );
                if best.as_ref().is_none_or(|(score, _)| identities > *score) {
                    best = Some((identities, orders));
                }
            }
        }
        best.map(|(_, orders)| orders)
            .ok_or_else(|| OperationError::InvalidPermutation {
                axes: output_axes.to_vec(),
                rank: open_rank,
            })
    }

    /// Whether every transpose is the identity: the spec is the composition
    /// `lhs.domain` against `rhs.codomain` (TensorKit `mul!`), which
    /// `planarcontract!` runs without copying either operand.
    pub(crate) fn is_composition(&self, lhs: (usize, usize), rhs: (usize, usize)) -> bool {
        let open = self.lhs[0].len();
        is_identity(&self.lhs, lhs.0, lhs.1)
            && is_identity(&self.rhs, rhs.0, rhs.1)
            && is_identity(&self.output, open, open + self.rhs[1].len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orders(
        lhs: (usize, usize),
        rhs: (usize, usize),
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<[[Vec<usize>; 2]; 3], OperationError> {
        let output: Vec<usize> = codomain.iter().chain(domain).copied().collect();
        PlanarContractOrders::validate(lhs, rhs, lhs_axes, rhs_axes, &output, codomain.len()).map(
            |orders| {
                [orders.lhs, orders.rhs, orders.output]
                    .map(|[c, d]| [c.into_iter().collect(), d.into_iter().collect()])
            },
        )
    }

    #[test]
    fn tensorkit_reorder_indices_examples() {
        // `@planar C[b d; h f] := A[a b; c d] * B[c f; a h]`: the O4 steps
        // `transpose(A, ((2,4),(1,3)))`, `transpose(B, ((3,1),(4,2)))`.
        assert_eq!(
            orders((2, 4), (2, 4), &[0, 2], &[2, 0], &[0, 1], &[3, 2]).unwrap(),
            [
                [vec![1, 3], vec![0, 2]],
                [vec![2, 0], vec![3, 1]],
                [vec![0, 1], vec![2, 3]]
            ]
        );
        // `gc.jl`: `@planar θ′[l s1 s2; r] := H[s1 s2; t1 t2] * θ[l t1 t2; r]`
        // keeps H, transposes θ to ((2,3),(1,4)) and the result to
        // ((3,1,2),(4,)).
        assert_eq!(
            orders((2, 4), (3, 4), &[2, 3], &[1, 2], &[2, 0, 1], &[3]).unwrap(),
            [
                [vec![0, 1], vec![2, 3]],
                [vec![1, 2], vec![0, 3]],
                [vec![2, 0, 1], vec![3]]
            ]
        );
    }

    #[test]
    fn composition_shapes_are_identity_transposes() {
        let mul =
            PlanarContractOrders::validate((2, 4), (2, 4), &[2, 3], &[0, 1], &[0, 1, 2, 3], 2)
                .unwrap();
        assert!(mul.is_composition((2, 4), (2, 4)));
        // A full contraction of matching boundaries picks the rotation
        // where both operands are read in place.
        let full =
            PlanarContractOrders::validate((0, 2), (2, 2), &[0, 1], &[0, 1], &[], 0).unwrap();
        assert!(full.is_composition((0, 2), (2, 2)));
        // An outer product with nothing contracted: `(a ←) ⊗ (← b)`.
        let outer = PlanarContractOrders::validate((1, 1), (0, 1), &[], &[], &[0, 1], 1).unwrap();
        assert!(outer.is_composition((1, 1), (0, 1)));
    }

    #[test]
    fn non_planar_specs_are_rejected_by_condition() {
        // Same orientation: walking A's arc forward walks B's forward too.
        assert!(matches!(
            orders((2, 4), (2, 4), &[0, 2], &[0, 2], &[0, 1], &[2, 3]),
            Err(OperationError::InvalidAxisSet { tensor: "rhs", .. })
        ));
        // A's contracted axes a, d are not adjacent on (a b d c).
        assert!(matches!(
            orders((2, 4), (2, 4), &[0, 3], &[1, 0], &[0, 1], &[2, 3]),
            Err(OperationError::InvalidAxisSet { tensor: "lhs", .. })
        ));
        // The output swaps two of A's open legs: not a rotation.
        assert!(matches!(
            orders((2, 4), (2, 4), &[0, 2], &[2, 0], &[1, 0], &[3, 2]),
            Err(OperationError::InvalidPermutation { .. })
        ));
        // Malformed specs keep the ordinary contraction's errors.
        assert!(matches!(
            orders((2, 4), (2, 4), &[0, 0], &[1, 2], &[0, 1], &[2, 3]),
            Err(OperationError::InvalidAxisSet { tensor: "lhs", .. })
        ));
        assert!(matches!(
            orders((2, 4), (2, 4), &[0], &[1, 2], &[0, 1], &[2, 3]),
            Err(OperationError::ContractAxisCountMismatch { .. })
        ));
    }
}
