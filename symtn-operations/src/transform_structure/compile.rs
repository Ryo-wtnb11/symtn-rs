use super::*;

trait TreeTransformCompileSpec<T> {
    fn dst_blocks(&self) -> &[usize];
    fn src_blocks(&self) -> &[usize];
    fn coefficients(&self) -> &[T];
    /// The spec's own shared matrix storage, which a Multi block references.
    fn shared_coefficients(&self) -> Option<&Arc<[T]>>;
    fn source_axes(&self) -> Option<&[usize]>;
}

impl<T> TreeTransformCompileSpec<T> for TreeTransformBlockSpec<T> {
    fn dst_blocks(&self) -> &[usize] {
        self.dst_blocks()
    }

    fn src_blocks(&self) -> &[usize] {
        self.src_blocks()
    }

    fn coefficients(&self) -> &[T] {
        self.recoupling_coefficients_dst_src()
    }

    fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        self.shared_coefficients()
    }

    fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes()
    }
}

impl<T> TreeTransformCompileSpec<T> for ResolvedTreeTransformBlockSpec<'_, T> {
    fn dst_blocks(&self) -> &[usize] {
        self.dst_blocks()
    }

    fn src_blocks(&self) -> &[usize] {
        self.src_blocks()
    }

    fn coefficients(&self) -> &[T] {
        self.coefficients()
    }

    fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        self.shared_coefficients()
    }

    fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes()
    }
}

/// Every layout of one transform block, from `dst_layout_start` to the table
/// end, must share the first destination's (source-permuted) shape. Replay
/// packs each source and scatters each destination in its own shape order,
/// so equal element counts alone would silently reshape a `[2, 2]` source
/// into a `[4, 1]` destination (#1739).
fn validate_uniform_layout_shapes(
    layouts: &RawLayoutTable,
    dst_layout_start: usize,
) -> Result<(), OperationError> {
    let expected = layouts.shape(dst_layout_start);
    for index in dst_layout_start + 1..layouts.entry_count() {
        let shape = layouts.shape(index);
        if shape != expected {
            return Err(OperationError::ShapeMismatch {
                dst: expected.to_vec(),
                src: shape.to_vec(),
            });
        }
    }
    Ok(())
}

impl<T: Copy> TreeTransformStructure<T> {
    pub fn compile_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_structures_with_storage_conjugation(
            dst_structure,
            src_structure,
            specs,
            false,
        )
    }

    pub fn compile_structures_with_storage_conjugation(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
            storage_conjugate,
        )
    }

    /// Compiles block-index specs against shared structures; the result
    /// keeps these `Arc`s, so no structure is copied.
    #[doc(hidden)]
    pub fn compile_indexed_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(dst_structure, src_structure, specs, storage_conjugate)
    }

    pub(crate) fn compile_resolved_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[ResolvedTreeTransformBlockSpec<'_, T>],
        storage_conjugate: bool,
        coefficients: Arc<TreeTransformCoefficients<T>>,
    ) -> Result<Self, OperationError> {
        Self::compile_with_coefficients(
            dst_structure,
            src_structure,
            specs,
            storage_conjugate,
            coefficients,
        )
    }

    /// Compiles keyed (Dense, Opaque or FusionTree label) specs against
    /// shared structures; the result keeps these `Arc`s, so no structure is
    /// copied.
    pub fn compile_keyed_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformKeyBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        let mut resolved_specs = Vec::with_capacity(specs.len());
        // Why not compile spec-by-spec: grouped and keyed entry points must
        // resolve every key before rank/count/layout validation.
        for spec in specs {
            resolved_specs.push(spec.resolve(&dst_structure, &src_structure)?);
        }
        Self::compile_shared_structures(
            dst_structure,
            src_structure,
            &resolved_specs,
            storage_conjugate,
        )
    }

    fn compile_shared_structures<S>(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[S],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError>
    where
        S: TreeTransformCompileSpec<T>,
    {
        let coefficients = TreeTransformCoefficients::from_specs(specs.iter().map(|spec| {
            (
                spec.dst_blocks().len(),
                spec.src_blocks().len(),
                spec.coefficients(),
                spec.shared_coefficients(),
            )
        }))?;
        Self::compile_with_coefficients(
            dst_structure,
            src_structure,
            specs,
            storage_conjugate,
            Arc::new(coefficients),
        )
    }

    fn compile_with_coefficients<S>(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[S],
        storage_conjugate: bool,
        shared: Arc<TreeTransformCoefficients<T>>,
    ) -> Result<Self, OperationError>
    where
        S: TreeTransformCompileSpec<T>,
    {
        let rank = dst_structure.rank();
        if src_structure.rank() != rank {
            return Err(OperationError::StructureRankMismatch {
                expected: rank,
                actual: src_structure.rank(),
            });
        }
        validate_destination_layouts_injective(
            &dst_structure,
            "tree transform destination layouts overlap",
        )?;

        let mut layouts = RawLayoutTable::new(rank);
        // Every destination block gets exactly one entry (a spec's or an
        // inactive one) and every spec source one more.
        let source_entries = specs.iter().fold(0usize, |sources, spec| {
            sources.saturating_add(spec.src_blocks().len())
        });
        layouts.reserve_exact(dst_structure.block_count().saturating_add(source_entries));
        let mut blocks = Vec::with_capacity(specs.len());
        let mut single_end = 0usize;
        let mut matrix_end = shared.singles.len();
        let mut matrix = 0u32;
        let mut touched_dst_blocks = vec![false; dst_structure.block_count()];

        for spec in specs {
            let dst_blocks = spec.dst_blocks();
            let src_blocks = spec.src_blocks();
            let spec_coefficients = spec.coefficients();
            if dst_blocks.is_empty() || src_blocks.is_empty() {
                return Err(OperationError::EmptyTransformBlock);
            }
            let src_count = src_blocks.len();
            let dst_count = dst_blocks.len();
            let expected_coefficients = src_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            if spec_coefficients.len() != expected_coefficients {
                return Err(OperationError::CoefficientCountMismatch {
                    expected: expected_coefficients,
                    actual: spec_coefficients.len(),
                });
            }

            for &dst_block in dst_blocks {
                let touched = touched_dst_blocks.get_mut(dst_block).ok_or(
                    OperationError::BlockIndexOutOfBounds {
                        tensor: "dst",
                        index: dst_block,
                        count: dst_structure.block_count(),
                    },
                )?;
                if *touched {
                    return Err(OperationError::DuplicateTransformDestination { dst_block });
                }
                *touched = true;
            }

            let packed = !(src_count == 1 && dst_count == 1);
            let dst_layout_start = layouts.entry_count();
            let mut element_count = None;
            for &dst_block in dst_blocks {
                let block = dst_structure.block(dst_block)?;
                let layout_element_count =
                    layouts.push_block(block.shape(), block.strides(), block.offset())?;
                match element_count {
                    Some(expected) if expected != layout_element_count => {
                        return Err(OperationError::ElementCountMismatch {
                            expected,
                            actual: layout_element_count,
                        });
                    }
                    Some(_) => {}
                    None => element_count = Some(layout_element_count),
                }
            }

            let src_layout_start = layouts.entry_count();
            for &src_block in src_blocks {
                let block = src_structure.block(src_block)?;
                let layout_element_count = layouts.push_block_with_axes(
                    block.shape(),
                    block.strides(),
                    block.offset(),
                    spec.source_axes(),
                )?;
                match element_count {
                    Some(expected) if expected != layout_element_count => {
                        return Err(OperationError::ElementCountMismatch {
                            expected,
                            actual: layout_element_count,
                        });
                    }
                    Some(_) => {}
                    None => element_count = Some(layout_element_count),
                }
            }
            let element_count = element_count.expect("validated non-empty block");
            validate_uniform_layout_shapes(&layouts, dst_layout_start)?;

            if !packed {
                if shared_coefficient_mismatch(
                    shared.singles.get(single_end..=single_end),
                    spec_coefficients,
                ) {
                    return Err(shared_payload_mismatch());
                }
                blocks.push(TreeTransformBlock::Single {
                    dst_layout: dst_layout_start,
                    src_layout: src_layout_start,
                    coefficient: single_end,
                });
                single_end += 1;
            } else {
                if shared_coefficient_mismatch(shared.matrix(matrix), spec_coefficients) {
                    return Err(shared_payload_mismatch());
                }
                let coefficient_start = matrix_end;
                matrix_end = matrix_end
                    .checked_add(spec_coefficients.len())
                    .ok_or(OperationError::ElementCountOverflow)?;
                blocks.push(TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                    matrix,
                });
                matrix = matrix
                    .checked_add(1)
                    .ok_or(OperationError::ElementCountOverflow)?;
            }
        }
        let mut inactive_dst_layouts = Vec::new();
        for (dst_block, touched) in touched_dst_blocks.into_iter().enumerate() {
            if touched {
                continue;
            }
            let block = dst_structure.block(dst_block)?;
            inactive_dst_layouts.push(layouts.entry_count());
            layouts.push_block(block.shape(), block.strides(), block.offset())?;
        }
        blocks.sort_by(|lhs, rhs| {
            tree_transform_block_weight(rhs, layouts.entries())
                .cmp(&tree_transform_block_weight(lhs, layouts.entries()))
        });
        if single_end != shared.singles.len() || matrix_end != shared.len {
            return Err(shared_payload_mismatch());
        }
        let recoupling_plan = compile_recoupling_plan(&blocks)?;
        let mut parallel_schedule =
            compile_parallel_schedule(&blocks, &inactive_dst_layouts, &layouts, &recoupling_plan)?;
        let physical_overwrite_len = compile_physical_overwrite_coverage(
            &blocks,
            &inactive_dst_layouts,
            &layouts,
            &recoupling_plan,
            dst_structure.block_count(),
            dst_structure.storage_tiling_proven(),
            dst_structure.required_len()?,
        )?;
        // Every structural proof above read the raw table; the retained
        // table keeps only the compiled roles replay executes.
        let layouts = layouts.finish(&blocks, &inactive_dst_layouts)?;
        super::ownership::record_planned(&mut parallel_schedule, &blocks, &layouts)?;

        Ok(Self::from_replay_core(
            Arc::new(TreeTransformReplay {
                rank,
                storage_conjugate,
                identity: Arc::new(()),
                blocks,
                layouts,
                coefficients: shared,
                inactive_dst_layouts,
                physical_overwrite_len,
                recoupling_plan,
                parallel_schedule,
            }),
            dst_structure,
            src_structure,
        ))
    }
}

#[doc(hidden)]
pub fn validate_destination_layouts_injective(
    dst_structure: &BlockStructure,
    overlap_message: &'static str,
) -> Result<(), OperationError> {
    // Why-not deduplicate only the beta scale: aliased logical destination
    // blocks can also require distinct alpha contributions, so no replay order
    // can represent their outputs in one physical element.
    match validate_block_storage_injective(dst_structure) {
        Ok(()) => Ok(()),
        Err(CoreError::OverlappingBlockStorage { .. }) => {
            // Why not expose block/offset details in a new variant:
            // OperationError is a public exhaustive enum, so that would
            // break downstream matches for a validation-only diagnostic.
            Err(OperationError::InvalidArgument {
                message: overlap_message,
            })
        }
        Err(CoreError::ElementCountOverflow) => Err(OperationError::ElementCountOverflow),
        Err(error) => Err(OperationError::Core(error)),
    }
}
