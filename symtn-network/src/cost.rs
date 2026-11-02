use std::collections::BTreeMap;

use crate::error::{ContractError, Result};
use crate::ir::{HyperEdge, NetworkIR};
use crate::labels::TemporaryLabel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTensorInfo {
    shape: Vec<usize>,
}

impl DenseTensorInfo {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseCostModel {
    label_dims: BTreeMap<TemporaryLabel, usize>,
}

impl DenseCostModel {
    pub fn from_network(ir: &NetworkIR, tensors: &[DenseTensorInfo]) -> Result<Self> {
        if tensors.len() != ir.tensors().len() {
            return Err(ContractError::TensorCountMismatch {
                expected: ir.tensors().len(),
                actual: tensors.len(),
            });
        }

        let mut label_dims = BTreeMap::<TemporaryLabel, usize>::new();
        for tensor in ir.tensors() {
            let tensor_info = &tensors[tensor.id().index()];
            if tensor_info.shape().len() != tensor.rank() {
                return Err(ContractError::RankMismatch {
                    tensor: tensor.id().index(),
                    expected: tensor.rank(),
                    actual: tensor_info.shape().len(),
                });
            }
            for (axis, label) in tensor.labels().iter().enumerate() {
                let dim = tensor_info.shape()[axis];
                match label_dims.get(label) {
                    Some(&expected) if expected != dim => {
                        return Err(ContractError::DimensionMismatch {
                            label: label.to_string(),
                            expected,
                            actual: dim,
                        });
                    }
                    Some(_) => {}
                    None => {
                        label_dims.insert(label.clone(), dim);
                    }
                }
            }
        }

        Ok(Self { label_dims })
    }

    pub fn dim(&self, label: &TemporaryLabel) -> Option<usize> {
        self.label_dims.get(label).copied()
    }

    /// A copy of this cost model with the given labels' dimensions forced to 1.
    ///
    /// Used by dynamic slicing to reflect already-sliced indices: a sliced index
    /// contributes a factor of 1 to intermediate sizes and contraction costs.
    pub fn with_unit_dims(&self, labels: &[TemporaryLabel]) -> DenseCostModel {
        let mut label_dims = self.label_dims.clone();
        for label in labels {
            if let Some(dim) = label_dims.get_mut(label) {
                *dim = 1;
            }
        }
        DenseCostModel { label_dims }
    }

    pub fn edge_dim(&self, edge: &HyperEdge) -> usize {
        self.dim(edge.label()).unwrap_or(1)
    }

    /// Number of scalar elements in a tensor with these legs (the product of the
    /// leg dimensions). Used as the dense FLOP/size proxy by `pair_cost`.
    ///
    /// The product is accumulated with [`usize::saturating_mul`] so a large
    /// network whose true size exceeds `usize::MAX` *saturates* to `usize::MAX`
    /// rather than silently wrapping to a small value — a wrapped tiny cost would
    /// otherwise make the greedy optimizer treat a huge contraction as cheap.
    pub fn tensor_size(&self, labels: &[TemporaryLabel]) -> usize {
        labels.iter().fold(1usize, |acc, label| {
            acc.saturating_mul(self.dim(label).unwrap_or(1))
        })
    }

    pub fn pair_cost(&self, lhs: &[TemporaryLabel], rhs: &[TemporaryLabel]) -> usize {
        let mut labels = lhs.to_vec();
        for label in rhs {
            if !labels.contains(label) {
                labels.push(label.clone());
            }
        }
        self.tensor_size(&labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::test_ir;

    fn label(s: &str) -> TemporaryLabel {
        TemporaryLabel::new(s)
    }

    #[test]
    fn with_unit_dims_forces_only_the_named_known_labels_to_one() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["a", "c"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let legs = [label("a"), label("b"), label("c")];
        assert_eq!(cost.tensor_size(&legs), 24);

        let unit = cost.with_unit_dims(&[label("b"), label("absent")]);
        assert_eq!(unit.dim(&label("a")), Some(2));
        assert_eq!(unit.dim(&label("b")), Some(1));
        assert_eq!(unit.dim(&label("c")), Some(4));
        assert_eq!(unit.dim(&label("absent")), None);
        assert_eq!(unit.tensor_size(&legs), 8);
        assert_eq!(
            cost.dim(&label("b")),
            Some(3),
            "the source model is unchanged"
        );
    }

    /// `tensor_size` (and thus `pair_cost`) must SATURATE on overflow instead of
    /// wrapping. With five legs of dimension 2^14 the true element count is
    /// 2^70, far beyond `usize::MAX` (2^64-1 on this target): a plain `.product()`
    /// would wrap to a tiny value and mislead the optimizer; the saturating
    /// fold must clamp at `usize::MAX`.
    #[test]
    fn dense_tensor_size_saturates_on_overflow() {
        let big = 1usize << 14; // 16384
                                // Five distinct legs, each dim 2^14 -> product 2^70 > usize::MAX.
        let ir = test_ir(&[&["a", "b", "c", "d", "e"]], &["a", "b", "c", "d", "e"]);
        let infos = vec![DenseTensorInfo::new(vec![big, big, big, big, big])];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();

        let legs = [label("a"), label("b"), label("c"), label("d"), label("e")];
        assert_eq!(
            cost.tensor_size(&legs),
            usize::MAX,
            "overflowing tensor size must saturate, not wrap"
        );

        // A non-overflowing product is still exact (no behavior change).
        assert_eq!(cost.tensor_size(&[label("a"), label("b")]), big * big);
    }

    /// `pair_cost` builds on `tensor_size`, so it must saturate too rather than
    /// wrap to a small (and misleading) number.
    #[test]
    fn dense_pair_cost_saturates_on_overflow() {
        let big = 1usize << 14;
        let ir = test_ir(&[&["a", "b", "c"], &["c", "d", "e"]], &["a", "b", "d", "e"]);
        let infos = vec![
            DenseTensorInfo::new(vec![big, big, big]),
            DenseTensorInfo::new(vec![big, big, big]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        // Union of {a,b,c} and {c,d,e} = {a,b,c,d,e}, five legs of 2^14 -> 2^70.
        let lhs = [label("a"), label("b"), label("c")];
        let rhs = [label("c"), label("d"), label("e")];
        assert_eq!(cost.pair_cost(&lhs, &rhs), usize::MAX);
    }
}
