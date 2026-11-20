use super::*;

/// Proof that one exact [`BlockStructure`] is categorically valid for `rule`.
///
/// This proof is deliberately LOCAL: it covers tree shape, fusion and
/// vertex-label admissibility under the provider-owned fusion style, and
/// structure rank for provider-domain sector IDs in this borrowed structure. It is neither a
/// fusion-space construction proof nor a firewall for arbitrary
/// numeric sector IDs.
///
/// Why not make this crate-private: `symtn-tensors` consumes the proof across
/// the crate boundary while the type remains hidden from generated public API
/// documentation.
#[doc(hidden)]
pub struct LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R> {
    pub(crate) rule: &'rule R,
    structure: &'structure BlockStructure,
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    pub fn try_new(
        rule: &'rule R,
        structure: &'structure BlockStructure,
    ) -> Result<Self, CoreError> {
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                });
            };
            let key_rank =
                key.codomain_tree().uncoupled().len() + key.domain_tree().uncoupled().len();
            if key_rank != structure.rank() {
                return Err(CoreError::StructureRankMismatch {
                    expected: structure.rank(),
                    actual: key_rank,
                });
            }
            key.validate_for_rule(rule)?;
        }
        Ok(Self { rule, structure })
    }

    #[inline]
    pub fn rule(&self) -> &'rule R {
        self.rule
    }

    #[inline]
    pub fn structure(&self) -> &'structure BlockStructure {
        self.structure
    }

    #[doc(hidden)]
    pub fn fusion_tree_pair_key(
        &self,
        index: usize,
    ) -> Result<Option<&'structure FusionTreePairKey>, CoreError> {
        Ok(match self.structure.block(index)?.key() {
            BlockKey::FusionTree(key) => Some(key),
            key => {
                return Err(CoreError::ExpectedFusionTreePairKey { actual: key.kind() });
            }
        })
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    pub fn execute_unique_rigid_for_block_index(
        &self,
        index: usize,
        operation: &PreparedTreePairOperation<'_>,
        scratch: &mut UniqueTreePairScratch,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError> {
        if self.rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: self.rule.fusion_style(),
            });
        }
        let key = self.required_fusion_tree_pair_key(index)?;
        operation.execute_unique_rigid_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key,
            },
            scratch,
        )
    }

    /// [`Self::execute_unique_rigid_for_block_index`] with the destination
    /// left in `scratch` ([`UniqueTreePairScratch::pair_view`]): a caller
    /// reusing one scratch for every block allocates nothing per block.
    pub fn execute_unique_rigid_for_block_index_into(
        &self,
        index: usize,
        operation: &PreparedTreePairOperation<'_>,
        scratch: &mut UniqueTreePairScratch,
    ) -> Result<R::Scalar, CoreError> {
        if self.rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: self.rule.fusion_style(),
            });
        }
        let key = self.required_fusion_tree_pair_key(index)?;
        operation.execute_unique_rigid_proven_into(
            ValidatedFusionTreePair {
                rule: self.rule,
                key,
            },
            scratch,
        )
    }

    /// Why this seam keeps a release check although the `symtn-tensors`
    /// gate already admitted the style: for an identity plan it is the only
    /// guard, since the identity rows never consult the rule (#1950).
    pub fn execute_multiplicity_free_for_block_index(
        &self,
        index: usize,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        validate_multiplicity_free_execution_style(self.rule)?;
        let key = self.required_fusion_tree_pair_key(index)?;
        operation.execute_multiplicity_free_proven(ValidatedFusionTreePair {
            rule: self.rule,
            key,
        })
    }

    #[expect(
        clippy::type_complexity,
        reason = "the public block API returns source-major transform rows as nested vectors"
    )]
    #[cfg(test)]
    pub(crate) fn execute_multiplicity_free_braid_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        operation.validate_block_preflight(self.rule, PreparedTreePairFamily::BraidLike)?;
        if operation.is_identity() {
            let indices = indices.into_iter();
            let (lower, upper) = indices.size_hint();
            let mut rows = Vec::with_capacity(upper.unwrap_or(lower));
            for index in indices {
                let source = self.required_fusion_tree_pair_key(index)?;
                operation.validate_source_split(
                    source.codomain_tree().uncoupled().len(),
                    source.domain_tree().uncoupled().len(),
                )?;
                rows.push(vec![(source.clone(), R::Scalar::one())]);
            }
            return Ok(rows);
        }
        let batch = ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_braid_tree_pair_block_proven(batch, &operation)
    }

    #[cfg(test)]
    pub(crate) fn execute_multiplicity_free_braid_ordered_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        self.execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(indices, &operation)
    }

    /// Style-admitted seam of the `symtn-tensors` tree-transform miss path:
    /// the caller has already rejected the rule's fusion style with its own
    /// `OperationError::UnsupportedFusionStyle` gate, so this checks the
    /// style only in debug builds (#1950). Why not check again: the gate
    /// owns the observable error, and a second query only repeats it per
    /// block or block group.
    #[doc(hidden)]
    pub fn execute_multiplicity_free_braid_ordered_for_block_indices_borrowed<I>(
        &self,
        indices: I,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        debug_assert!(self.rule.fusion_style().is_multiplicity_free());
        operation.validate_block_preflight(self.rule, PreparedTreePairFamily::BraidLike)?;
        let batch = ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_braid_tree_pair_block_ordered_proven(batch, operation)
    }

    #[expect(
        clippy::type_complexity,
        reason = "the public block API returns source-major transform rows as nested vectors"
    )]
    #[cfg(test)]
    pub(crate) fn execute_multiplicity_free_transpose_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        operation.validate_block_preflight(self.rule, PreparedTreePairFamily::Transpose)?;
        if operation.is_identity() {
            let indices = indices.into_iter();
            let (lower, upper) = indices.size_hint();
            let mut rows = Vec::with_capacity(upper.unwrap_or(lower));
            for index in indices {
                let source = self.required_fusion_tree_pair_key(index)?;
                operation.validate_source_split(
                    source.codomain_tree().uncoupled().len(),
                    source.domain_tree().uncoupled().len(),
                )?;
                rows.push(vec![(source.clone(), R::Scalar::one())]);
            }
            return Ok(rows);
        }
        let batch = ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_transpose_tree_pair_block_proven(batch, &operation)
    }

    #[cfg(test)]
    pub(crate) fn execute_multiplicity_free_transpose_ordered_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        self.execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
            indices, &operation,
        )
    }

    /// Style-admitted seam of the `symtn-tensors` tree-transform miss path:
    /// the caller has already rejected the rule's fusion style with its own
    /// `OperationError::UnsupportedFusionStyle` gate, so this checks the
    /// style only in debug builds (#1950). Why not check again: the gate
    /// owns the observable error, and a second query only repeats it per
    /// block or block group.
    #[doc(hidden)]
    pub fn execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed<I>(
        &self,
        indices: I,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        debug_assert!(self.rule.fusion_style().is_multiplicity_free());
        operation.validate_block_preflight(self.rule, PreparedTreePairFamily::Transpose)?;
        let batch = ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_transpose_tree_pair_block_ordered_proven(batch, operation)
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    pub fn braid_codomain_rows_for_block_index(
        &self,
        block_index: usize,
        permutation: &[usize],
        levels: &[usize],
    ) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError> {
        if !self.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: self.rule.fusion_style(),
            });
        }
        let source = self.required_fusion_tree_pair_key(block_index)?;
        let tree = source.codomain_tree();
        let rank = tree.uncoupled().len();
        if levels.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: levels.len(),
            });
        }
        let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
        execute_multiplicity_free_tree_braid_proven(
            ValidatedFusionTree {
                rule: self.rule,
                key: tree,
            },
            prepared,
        )
    }

    pub fn permute_codomain_rows_for_block_index(
        &self,
        block_index: usize,
        permutation: &[usize],
    ) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError> {
        if !self.rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: self.rule.braiding_style(),
            });
        }
        if !self.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: self.rule.fusion_style(),
            });
        }
        let source = self.required_fusion_tree_pair_key(block_index)?;
        let tree = source.codomain_tree();
        let rank = tree.uncoupled().len();
        let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
        let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
        execute_multiplicity_free_tree_braid_proven(
            ValidatedFusionTree {
                rule: self.rule,
                key: tree,
            },
            prepared,
        )
    }

    #[expect(
        clippy::type_complexity,
        reason = "the public block API returns source-major transform rows as nested vectors"
    )]
    pub fn braid_codomain_rows_for_block_indices<I>(
        &self,
        indices: I,
        permutation: &[usize],
        levels: &[usize],
    ) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        let batch = ValidatedMultiplicityFreeTreeBatch::from_locally_validated(self, indices)?;
        multiplicity_free_braid_tree_block_proven(batch, permutation, levels)
    }

    #[expect(
        clippy::type_complexity,
        reason = "the public block API returns source-major transform rows as nested vectors"
    )]
    pub fn permute_codomain_rows_for_block_indices<I>(
        &self,
        indices: I,
        permutation: &[usize],
    ) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        if !self.rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: self.rule.braiding_style(),
            });
        }
        let indices = indices.into_iter().collect::<SmallVec<[usize; 8]>>();
        let rank = indices
            .first()
            .map(|&index| {
                self.required_fusion_tree_pair_key(index)
                    .map(|key| key.codomain_tree().uncoupled().len())
            })
            .transpose()?
            .unwrap_or(0);
        let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
        self.braid_codomain_rows_for_block_indices(indices, permutation, &levels)
    }
}

#[doc(hidden)]
#[cfg(any(test, feature = "testing"))]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    pub fn generic_permute_tree_pair_for_block_index(
        &self,
        block_index: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        let source = self.required_generic_fusion_tree_pair_key(block_index)?;
        generic_permute_tree_pair_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key: source,
            },
            codomain_permutation,
            domain_permutation,
        )
    }

    pub fn generic_braid_tree_pair_for_block_index(
        &self,
        block_index: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        let source = self.required_generic_fusion_tree_pair_key(block_index)?;
        generic_braid_tree_pair_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key: source,
            },
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    pub fn generic_transpose_tree_pair_for_block_index(
        &self,
        block_index: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        let source = self.required_generic_fusion_tree_pair_key(block_index)?;
        generic_transpose_tree_pair_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key: source,
            },
            codomain_permutation,
            domain_permutation,
        )
    }

    fn required_generic_fusion_tree_pair_key(
        &self,
        block_index: usize,
    ) -> Result<&FusionTreePairKey, CoreError> {
        if self.rule.fusion_style() != FusionStyleKind::Generic {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: self.rule.fusion_style(),
            });
        }
        self.required_fusion_tree_pair_key(block_index)
    }
}

impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    pub(crate) fn required_fusion_tree_pair_key(
        &self,
        index: usize,
    ) -> Result<&'structure FusionTreePairKey, CoreError> {
        self.fusion_tree_pair_key(index)?
            .ok_or_else(|| CoreError::MalformedFusionTree {
                message: "validated fusion-tree group contains a dense block",
            })
    }
}
