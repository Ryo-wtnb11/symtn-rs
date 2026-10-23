use core::{alloc::Layout, any::TypeId};
use std::sync::{Arc, Weak};

use symtn_core::{BlockStructure, Placement};

use crate::{task_view::TreeTransformTaskView, OperationError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextIdentity(pub(crate) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageDomain(pub(crate) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AllocationIdentity(pub(crate) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageRegion {
    Empty,
    Bytes {
        domain: StorageDomain,
        allocation: AllocationIdentity,
        start: usize,
        len: usize,
    },
}

/// Immutable post-preparation storage facts for one synchronous operation.
///
/// A nonempty handle must report a checked byte region covering every byte
/// reachable through its usable capacity. All wrappers of one root allocation
/// must share `(domain, allocation)`, whose identity cannot be reused while a
/// snapshot is live. Region, placement, and context must remain stable through
/// the executor's final completion fence. Adapters unable to prove this
/// provenance are unsupported and must fail before submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageSnapshot {
    pub(crate) active_len: usize,
    pub(crate) usable_capacity: usize,
    pub(crate) placement: Placement,
    pub(crate) context: ContextIdentity,
    pub(crate) region: StorageRegion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExecutorSnapshot {
    pub(crate) placement: Placement,
    pub(crate) context: ContextIdentity,
    pub(crate) supports_strided: bool,
    pub(crate) supports_matrix: bool,
    pub(crate) scalar: TypeId,
    /// How many workers need their own host fused-index scratch row.
    ///
    /// Executor-declared, because it is a fact about the executor rather than
    /// about the task: host replay sizes one row per worker, while a device
    /// executor reads the structure's baked fused layout and declares `0`. A
    /// zero here means the workspace owes no fused-index buffer at all, which
    /// is what keeps Stage C from demanding a Host allocation of an executor
    /// that never touches host memory.
    pub(crate) fused_index_workers: usize,
}

/// Stage-A-issued checked arithmetic result. Private fields prevent later
/// operation adapters from weakening Stage C with a forged scratch length.
pub(crate) struct AdmissionRequirements {
    fused_index_len: usize,
}

impl AdmissionRequirements {
    #[cfg(test)]
    pub(crate) fn fused_index_len(&self) -> usize {
        self.fused_index_len
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CoefficientReadiness {
    pub(crate) structure_and_layout: Weak<()>,
    pub(crate) scalar: TypeId,
    pub(crate) context: ContextIdentity,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceSnapshot {
    pub(crate) packed_source: StorageSnapshot,
    pub(crate) packed_destination: StorageSnapshot,
    pub(crate) converted_coefficients: StorageSnapshot,
    pub(crate) fused_index_capacity: usize,
    pub(crate) fused_index_placement: Placement,
    pub(crate) coefficient_readiness: Option<CoefficientReadiness>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TreeTransformAdmissionError {
    Structure(&'static str),
    Length { expected: usize, actual: usize },
    ArithmeticOverflow,
    Placement(&'static str),
    Context(&'static str),
    Capability(&'static str),
    InvalidRegion(&'static str),
    Aliasing,
    WorkspaceCapacity(&'static str),
    CoefficientReadiness,
    Task(OperationError),
}

/// Failures the Host replay also detects keep the Host's typed category:
/// `validate_structures_and_lengths` reports `StructureMismatch` and
/// `ElementCountMismatch`, and `validate_workspace_requirements` /
/// `checked_fused_index_len` report `ElementCountOverflow`. Only the
/// placement, context, capability, region, aliasing, workspace and readiness
/// verdicts are opaque-executor-only and have no Host category to keep.
/// Design record: `docs/backend_policy.md#device-validation`.
impl From<TreeTransformAdmissionError> for OperationError {
    fn from(error: TreeTransformAdmissionError) -> Self {
        let message = match error {
            TreeTransformAdmissionError::Structure(tensor) => {
                return OperationError::StructureMismatch { tensor }
            }
            TreeTransformAdmissionError::Length { expected, actual } => {
                return OperationError::ElementCountMismatch { expected, actual }
            }
            TreeTransformAdmissionError::ArithmeticOverflow => {
                return OperationError::ElementCountOverflow
            }
            TreeTransformAdmissionError::Task(error) => return error,
            TreeTransformAdmissionError::Placement(_) => {
                "tree transform placement admission failed"
            }
            TreeTransformAdmissionError::Context(_) => "tree transform context admission failed",
            TreeTransformAdmissionError::Capability(_) => {
                "tree transform capability admission failed"
            }
            TreeTransformAdmissionError::InvalidRegion(_) => {
                "tree transform storage region admission failed"
            }
            TreeTransformAdmissionError::Aliasing => "tree transform storage regions overlap",
            TreeTransformAdmissionError::WorkspaceCapacity(_) => {
                "tree transform workspace capacity admission failed"
            }
            TreeTransformAdmissionError::CoefficientReadiness => {
                "tree transform coefficient readiness admission failed"
            }
        };
        OperationError::InvalidArgument { message }
    }
}

pub(crate) fn validate_stage_a<D: 'static, C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst: StorageSnapshot,
    src: StorageSnapshot,
    executor: ExecutorSnapshot,
) -> Result<AdmissionRequirements, TreeTransformAdmissionError> {
    // Stage A order: completed structure and checked arithmetic, exact lengths,
    // then executor-local placement, context, and finite capabilities.
    task.validate_structures(dst_structure, src_structure)
        .map_err(map_task_error)?;
    let requirements = task.workspace_requirements();
    for len in [
        requirements.packed_source_len,
        requirements.packed_destination_len,
        requirements.converted_coefficient_len,
    ] {
        Layout::array::<D>(len).map_err(|_| TreeTransformAdmissionError::ArithmeticOverflow)?;
    }
    let fused_index_len = executor
        .fused_index_workers
        .checked_mul(requirements.fused_index_len_per_worker)
        .ok_or(TreeTransformAdmissionError::ArithmeticOverflow)?;
    Layout::array::<usize>(fused_index_len)
        .map_err(|_| TreeTransformAdmissionError::ArithmeticOverflow)?;
    task.validate_lengths(dst.active_len, src.active_len)
        .map_err(map_task_error)?;
    for (name, storage) in [("destination", dst), ("source", src)] {
        if storage.placement != executor.placement {
            return Err(TreeTransformAdmissionError::Placement(name));
        }
        if storage.context != executor.context {
            return Err(TreeTransformAdmissionError::Context(name));
        }
    }
    if !executor.supports_strided {
        return Err(TreeTransformAdmissionError::Capability("strided"));
    }
    if !task.recoupling_plan().is_empty() && !executor.supports_matrix {
        return Err(TreeTransformAdmissionError::Capability("matrix"));
    }
    if executor.scalar != TypeId::of::<D>() {
        return Err(TreeTransformAdmissionError::Capability("scalar"));
    }
    Ok(AdmissionRequirements { fused_index_len })
}

pub(crate) fn validate_stage_c<D: 'static, C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    dst: StorageSnapshot,
    src: StorageSnapshot,
    workspace: &WorkspaceSnapshot,
    executor: ExecutorSnapshot,
    admission: &AdmissionRequirements,
) -> Result<(), TreeTransformAdmissionError> {
    let requirements = task.workspace_requirements();
    // Stage C order: immutable provenance, aliasing, workspace facts, then the
    // exact coefficient-readiness witness.
    validate_region::<D>("destination", dst)?;
    validate_region::<D>("source", src)?;
    let workspace_slots = [
        (
            "packed source",
            workspace.packed_source,
            requirements.packed_source_len,
        ),
        (
            "packed destination",
            workspace.packed_destination,
            requirements.packed_destination_len,
        ),
        (
            "converted coefficients",
            workspace.converted_coefficients,
            requirements.converted_coefficient_len,
        ),
    ];
    for (name, storage, _) in workspace_slots {
        validate_region::<D>(name, storage)?;
    }
    let regions = [
        (dst.region, true),
        (src.region, false),
        (workspace.packed_source.region, true),
        (workspace.packed_destination.region, true),
        (workspace.converted_coefficients.region, false),
    ];
    for left in 0..regions.len() {
        for right in left + 1..regions.len() {
            if (regions[left].1 || regions[right].1) && overlaps(regions[left].0, regions[right].0)?
            {
                return Err(TreeTransformAdmissionError::Aliasing);
            }
        }
    }
    for (name, storage, required) in workspace_slots {
        if storage.placement != executor.placement {
            return Err(TreeTransformAdmissionError::Placement(name));
        }
        if storage.context != executor.context {
            return Err(TreeTransformAdmissionError::Context(name));
        }
        if storage.usable_capacity < required {
            return Err(TreeTransformAdmissionError::WorkspaceCapacity(name));
        }
    }
    // Only an executor that asked for fused-index rows owes a host buffer.
    if admission.fused_index_len != 0 && workspace.fused_index_placement != Placement::Host {
        return Err(TreeTransformAdmissionError::Placement("fused indices"));
    }
    if workspace.fused_index_capacity < admission.fused_index_len {
        return Err(TreeTransformAdmissionError::WorkspaceCapacity(
            "fused indices",
        ));
    }
    if requirements.converted_coefficient_len != 0 {
        let ready = workspace.coefficient_readiness.as_ref();
        if !ready.is_some_and(|ready| {
            Weak::ptr_eq(&ready.structure_and_layout, &task.admission_identity())
                && ready.scalar == TypeId::of::<D>()
                && ready.context == executor.context
        }) {
            return Err(TreeTransformAdmissionError::CoefficientReadiness);
        }
    }
    Ok(())
}

fn validate_region<D>(
    name: &'static str,
    storage: StorageSnapshot,
) -> Result<(), TreeTransformAdmissionError> {
    if storage.active_len > storage.usable_capacity {
        return Err(TreeTransformAdmissionError::InvalidRegion(name));
    }
    let bytes = Layout::array::<D>(storage.usable_capacity)
        .map_err(|_| TreeTransformAdmissionError::ArithmeticOverflow)?
        .size();
    match storage.region {
        StorageRegion::Empty if storage.usable_capacity == 0 => Ok(()),
        StorageRegion::Empty => Err(TreeTransformAdmissionError::InvalidRegion(name)),
        StorageRegion::Bytes { start, len, .. } => {
            start
                .checked_add(len)
                .ok_or(TreeTransformAdmissionError::InvalidRegion(name))?;
            if len < bytes {
                Err(TreeTransformAdmissionError::InvalidRegion(name))
            } else {
                Ok(())
            }
        }
    }
}

fn overlaps(
    left: StorageRegion,
    right: StorageRegion,
) -> Result<bool, TreeTransformAdmissionError> {
    let (
        StorageRegion::Bytes {
            domain: ld,
            allocation: la,
            start: ls,
            len: ll,
        },
        StorageRegion::Bytes {
            domain: rd,
            allocation: ra,
            start: rs,
            len: rl,
        },
    ) = (left, right)
    else {
        return Ok(false);
    };
    let le = ls
        .checked_add(ll)
        .ok_or(TreeTransformAdmissionError::InvalidRegion("storage"))?;
    let re = rs
        .checked_add(rl)
        .ok_or(TreeTransformAdmissionError::InvalidRegion("storage"))?;
    Ok(ld == rd && la == ra && ls < re && rs < le)
}

fn map_task_error(error: OperationError) -> TreeTransformAdmissionError {
    match error {
        OperationError::StructureMismatch { tensor } => {
            TreeTransformAdmissionError::Structure(tensor)
        }
        OperationError::ElementCountMismatch { expected, actual } => {
            TreeTransformAdmissionError::Length { expected, actual }
        }
        OperationError::ElementCountOverflow => TreeTransformAdmissionError::ArithmeticOverflow,
        error => TreeTransformAdmissionError::Task(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TreeTransformBlock, TreeTransformBlockSpec, TreeTransformStructure};
    use symtn_core::BlockSpec;

    #[derive(Clone, Copy)]
    enum DestinationMode {
        Overwrite,
        Axpby(f64),
    }

    #[derive(Debug, PartialEq)]
    enum MockError {
        Admission(TreeTransformAdmissionError),
        Preparation,
        Backend,
    }

    #[derive(Clone, Copy)]
    enum AdmissionFault {
        None,
        DestinationPlacement,
        DestinationContext,
        DestinationLength,
        ActiveBeyondCapacity,
        MissingStrided,
        MissingMatrix,
        WrongScalar,
        Overlap,
        PackedSourceCapacity,
        PackedDestinationCapacity,
        CoefficientCapacity,
        FusedCapacity,
        WorkspacePlacement,
        WorkspaceContext,
        StaleReadiness,
        WrongReadinessScalar,
        WrongReadinessContext,
        ReadReadOverlap,
    }

    struct MockOpaqueExecutor {
        source: Vec<f64>,
        destination: Vec<f64>,
        source_region: StorageRegion,
        destination_region: StorageRegion,
        placement: Placement,
        context: ContextIdentity,
        fault: AdmissionFault,
        fail_preparation: bool,
        fail_after_submission: Option<usize>,
        converted_coefficients: Vec<f64>,
        coefficient_readiness: Option<CoefficientReadiness>,
        submissions: usize,
        writes: usize,
    }

    impl MockOpaqueExecutor {
        fn new(source: Vec<f64>, destination: Vec<f64>) -> Self {
            Self {
                source_region: region(1, 2, 0, source.capacity() * size_of::<f64>()),
                destination_region: region(1, 1, 0, destination.capacity() * size_of::<f64>()),
                source,
                destination,
                placement: Placement::Cuda(3),
                context: ContextIdentity(7),
                fault: AdmissionFault::None,
                fail_preparation: false,
                fail_after_submission: None,
                converted_coefficients: Vec::new(),
                coefficient_readiness: None,
                submissions: 0,
                writes: 0,
            }
        }

        fn execute(
            &mut self,
            transform: &TreeTransformStructure<f64>,
            structure: &Arc<BlockStructure>,
            alpha: f64,
            mode: DestinationMode,
            workers: usize,
        ) -> Result<(), MockError> {
            let task = transform
                .task_view()
                .map_err(|error| MockError::Admission(TreeTransformAdmissionError::Task(error)))?;
            let mut executor = ExecutorSnapshot {
                placement: self.placement,
                context: self.context,
                supports_strided: true,
                supports_matrix: true,
                scalar: TypeId::of::<f64>(),
                fused_index_workers: workers.max(1),
            };
            let mut dst = self.storage_snapshot(
                self.destination.len(),
                self.destination.capacity(),
                self.destination_region,
            );
            let src = self.storage_snapshot(
                self.source.len(),
                self.source.capacity(),
                self.source_region,
            );
            match self.fault {
                AdmissionFault::DestinationPlacement => dst.placement = Placement::Host,
                AdmissionFault::DestinationContext => dst.context = ContextIdentity(99),
                AdmissionFault::DestinationLength => dst.active_len -= 1,
                AdmissionFault::ActiveBeyondCapacity => dst.usable_capacity -= 1,
                AdmissionFault::MissingStrided => executor.supports_strided = false,
                AdmissionFault::MissingMatrix => executor.supports_matrix = false,
                AdmissionFault::WrongScalar => executor.scalar = TypeId::of::<f32>(),
                _ => {}
            }
            let admission =
                validate_stage_a::<f64, _>(task, structure, structure, dst, src, executor)
                    .map_err(MockError::Admission)?;

            // Stage B invalidates readiness before any allocation/conversion.
            let mut workspace =
                self.prepare_workspace(task, executor, admission.fused_index_len())?;
            self.inject_stage_c_fault(&mut workspace, &mut dst, src);
            validate_stage_c::<f64, _>(task, dst, src, &workspace, executor, &admission)
                .map_err(MockError::Admission)?;

            for (block_index, block) in task.blocks().iter().enumerate() {
                self.submissions += 1;
                self.execute_block(task, block_index, block, alpha, mode);
                if self.fail_after_submission == Some(self.submissions) {
                    return Err(MockError::Backend);
                }
            }
            Ok(())
        }

        fn storage_snapshot(
            &self,
            active_len: usize,
            usable_capacity: usize,
            region: StorageRegion,
        ) -> StorageSnapshot {
            StorageSnapshot {
                active_len,
                usable_capacity,
                placement: self.placement,
                context: self.context,
                region,
            }
        }

        fn prepare_workspace<C: Copy + Into<f64>>(
            &mut self,
            task: TreeTransformTaskView<'_, C>,
            executor: ExecutorSnapshot,
            fused: usize,
        ) -> Result<WorkspaceSnapshot, MockError> {
            self.coefficient_readiness = None;
            if self.fail_preparation {
                return Err(MockError::Preparation);
            }
            let required = task.workspace_requirements();
            self.converted_coefficients.clear();
            self.converted_coefficients
                .reserve(required.converted_coefficient_len);
            for (block_index, _) in task.recoupling_plan().entries() {
                let TreeTransformBlock::Multi { .. } = task.blocks()[block_index] else {
                    unreachable!("completed recoupling plan references Multi blocks")
                };
                self.converted_coefficients.extend(
                    task.block_matrix(block_index)
                        .expect("completed Multi block has a matrix")
                        .iter()
                        .copied()
                        .map(Into::into),
                );
            }
            if required.converted_coefficient_len != 0 {
                self.coefficient_readiness = Some(CoefficientReadiness {
                    structure_and_layout: task.admission_identity(),
                    scalar: TypeId::of::<f64>(),
                    context: executor.context,
                });
            }
            let storage = |allocation, capacity| StorageSnapshot {
                active_len: capacity,
                usable_capacity: capacity,
                placement: executor.placement,
                context: executor.context,
                region: if capacity == 0 {
                    StorageRegion::Empty
                } else {
                    region(42, allocation, 0, capacity * size_of::<f64>())
                },
            };
            Ok(WorkspaceSnapshot {
                packed_source: storage(3, required.packed_source_len),
                packed_destination: storage(4, required.packed_destination_len),
                converted_coefficients: storage(5, required.converted_coefficient_len),
                fused_index_capacity: fused,
                fused_index_placement: Placement::Host,
                coefficient_readiness: self.coefficient_readiness.clone(),
            })
        }

        fn inject_stage_c_fault(
            &self,
            workspace: &mut WorkspaceSnapshot,
            dst: &mut StorageSnapshot,
            src: StorageSnapshot,
        ) {
            match self.fault {
                AdmissionFault::Overlap => dst.region = src.region,
                AdmissionFault::PackedSourceCapacity => {
                    workspace.packed_source.usable_capacity =
                        workspace.packed_source.usable_capacity.saturating_sub(1);
                    workspace.packed_source.active_len = workspace.packed_source.usable_capacity;
                }
                AdmissionFault::PackedDestinationCapacity => {
                    workspace.packed_destination.usable_capacity = workspace
                        .packed_destination
                        .usable_capacity
                        .saturating_sub(1);
                    workspace.packed_destination.active_len =
                        workspace.packed_destination.usable_capacity;
                }
                AdmissionFault::CoefficientCapacity => {
                    workspace.converted_coefficients.usable_capacity = workspace
                        .converted_coefficients
                        .usable_capacity
                        .saturating_sub(1);
                    workspace.converted_coefficients.active_len =
                        workspace.converted_coefficients.usable_capacity;
                }
                AdmissionFault::FusedCapacity => {
                    workspace.fused_index_capacity =
                        workspace.fused_index_capacity.saturating_sub(1)
                }
                AdmissionFault::WorkspacePlacement => {
                    workspace.packed_source.placement = Placement::Host
                }
                AdmissionFault::WorkspaceContext => {
                    workspace.packed_source.context = ContextIdentity(99)
                }
                AdmissionFault::StaleReadiness => {
                    workspace.coefficient_readiness = Some(CoefficientReadiness {
                        structure_and_layout: Weak::new(),
                        scalar: TypeId::of::<f64>(),
                        context: self.context,
                    })
                }
                AdmissionFault::WrongReadinessScalar => {
                    workspace.coefficient_readiness.as_mut().unwrap().scalar = TypeId::of::<f32>()
                }
                AdmissionFault::WrongReadinessContext => {
                    workspace.coefficient_readiness.as_mut().unwrap().context = ContextIdentity(99)
                }
                AdmissionFault::ReadReadOverlap => {
                    workspace.converted_coefficients.region = src.region
                }
                _ => {}
            }
        }

        fn execute_block<C: Copy + Into<f64>>(
            &mut self,
            task: TreeTransformTaskView<'_, C>,
            block_index: usize,
            block: &TreeTransformBlock,
            alpha: f64,
            mode: DestinationMode,
        ) {
            match *block {
                TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    coefficient,
                } => {
                    let dst = task.layouts().entry(dst_layout);
                    let src = task.layouts().entry(src_layout);
                    let coefficient = task.single_coefficients()[coefficient].into();
                    for element in 0..dst.element_count {
                        self.write(
                            dst.offset as usize + element,
                            alpha * coefficient * self.source[src.offset as usize + element],
                            mode,
                        );
                    }
                }
                TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start: _,
                    element_count,
                    matrix: _,
                } => {
                    let mut prepared_start = 0;
                    for (prepared_block, _) in task.recoupling_plan().entries() {
                        let TreeTransformBlock::Multi {
                            dst_count,
                            src_count,
                            ..
                        } = task.blocks()[prepared_block]
                        else {
                            unreachable!("completed recoupling plan references Multi blocks")
                        };
                        if prepared_block == block_index {
                            break;
                        }
                        prepared_start += dst_count * src_count;
                    }
                    for destination in 0..dst_count {
                        let dst = task.layouts().entry(dst_layout_start + destination);
                        for element in 0..element_count {
                            let mut value = 0.0;
                            for source in 0..src_count {
                                let src = task.layouts().entry(src_layout_start + source);
                                value += self.source[src.offset as usize + element]
                                    * self.converted_coefficients
                                        [prepared_start + destination * src_count + source];
                            }
                            self.write(dst.offset as usize + element, alpha * value, mode);
                        }
                    }
                }
            }
        }

        fn write(&mut self, index: usize, value: f64, mode: DestinationMode) {
            self.destination[index] = match mode {
                DestinationMode::Overwrite => value,
                DestinationMode::Axpby(beta) => value + beta * self.destination[index],
            };
            self.writes += 1;
        }
    }

    fn region(domain: u64, allocation: u64, start: usize, len: usize) -> StorageRegion {
        StorageRegion::Bytes {
            domain: StorageDomain(domain),
            allocation: AllocationIdentity(allocation),
            start,
            len,
        }
    }

    fn scalar_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, 3.0),
            ],
        )
        .unwrap();
        (structure, transform)
    }

    fn multi_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![0.0, 1.0, 1.0, 0.0],
            )],
        )
        .unwrap();
        (structure, transform)
    }

    #[test]
    fn half_open_regions_obey_domain_and_empty_rules() {
        assert!(!overlaps(region(1, 1, 0, 8), region(1, 1, 8, 8)).unwrap());
        assert!(overlaps(region(1, 1, 0, 9), region(1, 1, 8, 8)).unwrap());
        assert!(!overlaps(region(1, 1, 0, 9), region(2, 1, 0, 9)).unwrap());
        assert!(!overlaps(StorageRegion::Empty, region(1, 1, 0, 9)).unwrap());
        assert_eq!(
            overlaps(region(1, 1, usize::MAX, 2), region(1, 1, 0, 1)),
            Err(TreeTransformAdmissionError::InvalidRegion("storage"))
        );
    }

    #[test]
    fn nonempty_capacity_requires_a_large_enough_checked_region() {
        let snapshot = StorageSnapshot {
            active_len: 2,
            usable_capacity: 2,
            placement: Placement::Host,
            context: ContextIdentity(1),
            region: StorageRegion::Empty,
        };
        assert_eq!(
            validate_region::<u64>("source", snapshot),
            Err(TreeTransformAdmissionError::InvalidRegion("source"))
        );
        assert_eq!(
            validate_region::<u64>(
                "source",
                StorageSnapshot {
                    active_len: 3,
                    region: region(1, 1, 0, 16),
                    ..snapshot
                }
            ),
            Err(TreeTransformAdmissionError::InvalidRegion("source"))
        );
        assert_eq!(
            validate_region::<u64>(
                "source",
                StorageSnapshot {
                    region: region(1, 1, 0, 15),
                    ..snapshot
                }
            ),
            Err(TreeTransformAdmissionError::InvalidRegion("source"))
        );
        validate_region::<u64>(
            "source",
            StorageSnapshot {
                region: region(1, 1, 0, 16),
                ..snapshot
            },
        )
        .unwrap();
    }

    #[test]
    fn a_fused_index_buffer_is_owed_only_by_an_executor_that_asked_for_rows() {
        // What: the fused-index scratch is an executor-declared fact. An
        // executor that declares no workers (a device executor reads the
        // structure's baked fused layout instead) owes no Host buffer, while
        // one that declares workers still must provide Host rows.
        let (structure, transform) = scalar_fixture();
        let task = transform.task_view().unwrap();
        let context = ContextIdentity(11);
        let device = Placement::Cuda(0);
        let executor = |placement, workers| ExecutorSnapshot {
            placement,
            context,
            supports_strided: true,
            supports_matrix: true,
            scalar: TypeId::of::<f64>(),
            fused_index_workers: workers,
        };
        let storage = |placement, allocation, capacity: usize| StorageSnapshot {
            active_len: capacity,
            usable_capacity: capacity,
            placement,
            context,
            region: if capacity == 0 {
                StorageRegion::Empty
            } else {
                region(1, allocation, 0, capacity * size_of::<f64>())
            },
        };
        let workspace = |placement, capacity| WorkspaceSnapshot {
            packed_source: storage(placement, 3, 0),
            packed_destination: storage(placement, 4, 0),
            converted_coefficients: storage(placement, 5, 0),
            fused_index_capacity: capacity,
            fused_index_placement: placement,
            coefficient_readiness: None,
        };

        // Zero workers: no rows required, and the workspace may report the
        // executor's own placement for a buffer that does not exist.
        let device_executor = executor(device, 0);
        let dst = storage(device, 1, 4);
        let src = storage(device, 2, 4);
        let admission =
            validate_stage_a::<f64, _>(task, &structure, &structure, dst, src, device_executor)
                .unwrap();
        assert_eq!(admission.fused_index_len(), 0);
        validate_stage_c::<f64, _>(
            task,
            dst,
            src,
            &workspace(device, 0),
            device_executor,
            &admission,
        )
        .unwrap();

        // One worker: rows are required and they must live on the host.
        let host_executor = executor(Placement::Host, 1);
        let dst = storage(Placement::Host, 1, 4);
        let src = storage(Placement::Host, 2, 4);
        let admission =
            validate_stage_a::<f64, _>(task, &structure, &structure, dst, src, host_executor)
                .unwrap();
        assert_eq!(admission.fused_index_len(), 1);
        validate_stage_c::<f64, _>(
            task,
            dst,
            src,
            &workspace(Placement::Host, 1),
            host_executor,
            &admission,
        )
        .unwrap();
        let mut elsewhere = workspace(Placement::Host, 1);
        elsewhere.fused_index_placement = device;
        assert_eq!(
            validate_stage_c::<f64, _>(task, dst, src, &elsewhere, host_executor, &admission),
            Err(TreeTransformAdmissionError::Placement("fused indices"))
        );
    }

    #[test]
    fn stage_a_and_c_validate_completed_multi_task() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
        let transform = crate::TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0_f64, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        let task = transform.task_view().unwrap();
        let context = ContextIdentity(7);
        let executor = ExecutorSnapshot {
            placement: Placement::Host,
            context,
            supports_strided: true,
            supports_matrix: true,
            scalar: TypeId::of::<f64>(),
            fused_index_workers: 3,
        };
        let storage = |allocation, capacity| StorageSnapshot {
            active_len: capacity,
            usable_capacity: capacity,
            placement: Placement::Host,
            context,
            region: region(1, allocation, 0, capacity * size_of::<f64>()),
        };
        let dst = storage(1, 4);
        let src = storage(2, 4);
        let admission =
            validate_stage_a::<f64, _>(task, &structure, &structure, dst, src, executor).unwrap();
        assert_eq!(admission.fused_index_len(), 3);
        let structure_and_layout = task.admission_identity();
        let workspace = WorkspaceSnapshot {
            packed_source: storage(3, 4),
            packed_destination: storage(4, 4),
            converted_coefficients: storage(5, 4),
            fused_index_capacity: admission.fused_index_len(),
            fused_index_placement: Placement::Host,
            coefficient_readiness: Some(CoefficientReadiness {
                structure_and_layout,
                scalar: TypeId::of::<f64>(),
                context,
            }),
        };
        validate_stage_c::<f64, _>(task, dst, src, &workspace, executor, &admission).unwrap();
        let mut too_small = workspace.clone();
        too_small.fused_index_capacity = admission.fused_index_len() - 1;
        assert_eq!(
            validate_stage_c::<f64, _>(task, dst, src, &too_small, executor, &admission),
            Err(TreeTransformAdmissionError::WorkspaceCapacity(
                "fused indices"
            ))
        );
        assert_eq!(
            validate_stage_c::<f64, _>(
                task,
                dst,
                src,
                &WorkspaceSnapshot {
                    packed_destination: StorageSnapshot {
                        region: region(1, 1, 8, 32),
                        ..workspace.packed_destination
                    },
                    ..workspace
                },
                executor,
                &admission,
            ),
            Err(TreeTransformAdmissionError::Aliasing)
        );
        assert_eq!(
            OperationError::from(TreeTransformAdmissionError::Capability("matrix")),
            OperationError::InvalidArgument {
                message: "tree transform capability admission failed"
            }
        );
    }

    /// #1763: every Stage A failure the Host replay also detects surfaces with
    /// the Host's `OperationError`, in Stage A order, before any device work.
    #[test]
    fn host_shared_admission_failures_keep_the_host_category() {
        let strided = Arc::new(
            BlockStructure::from_blocks_with_rank(
                2,
                vec![BlockSpec::new(vec![2, 2], vec![1, 3], 0).unwrap()],
            )
            .unwrap(),
        );
        let other = Arc::new(BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &strided,
            &strided,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let task = transform.task_view().unwrap();
        assert!(task.workspace_requirements().fused_index_len_per_worker > 1);
        let executor = |workers| ExecutorSnapshot {
            placement: Placement::Cuda(0),
            context: ContextIdentity(7),
            supports_strided: true,
            supports_matrix: true,
            scalar: TypeId::of::<f64>(),
            fused_index_workers: workers,
        };
        let storage = |allocation, active_len| StorageSnapshot {
            active_len,
            usable_capacity: active_len,
            placement: Placement::Cuda(0),
            context: ContextIdentity(7),
            region: region(1, allocation, 0, active_len * size_of::<f64>()),
        };
        // (case, dst structure, dst len, workers, stage-A placement fault, expected)
        let cases = [
            (
                "structure",
                &other,
                5,
                0,
                false,
                OperationError::StructureMismatch { tensor: "dst" },
            ),
            (
                "structure before length and placement",
                &other,
                4,
                0,
                true,
                OperationError::StructureMismatch { tensor: "dst" },
            ),
            (
                "length",
                &strided,
                4,
                0,
                false,
                OperationError::ElementCountMismatch {
                    expected: 5,
                    actual: 4,
                },
            ),
            (
                "length before placement",
                &strided,
                4,
                0,
                true,
                OperationError::ElementCountMismatch {
                    expected: 5,
                    actual: 4,
                },
            ),
            (
                "fused-index overflow",
                &strided,
                5,
                usize::MAX,
                false,
                OperationError::ElementCountOverflow,
            ),
        ];
        for (case, dst_structure, dst_len, workers, misplaced, expected) in cases {
            let mut dst = storage(1, dst_len);
            if misplaced {
                dst.placement = Placement::Host;
            }
            let error = validate_stage_a::<f64, _>(
                task,
                dst_structure,
                &strided,
                dst,
                storage(2, 5),
                executor(workers),
            )
            .err()
            .unwrap_or_else(|| panic!("{case}: admitted"));
            assert_eq!(OperationError::from(error), expected, "{case}");
            if workers == 0 {
                // The Host replay's own entry check reports the same error.
                assert_eq!(
                    task.validate_structures_and_lengths(dst_structure, &strided, dst_len, 5),
                    Err(expected),
                    "{case}: host"
                );
            }
        }

        // A task error passes through unchanged, as the Host returns it.
        for error in [
            OperationError::EmptyTransformBlock,
            OperationError::BlockCountMismatch { dst: 1, src: 2 },
        ] {
            assert_eq!(
                OperationError::from(TreeTransformAdmissionError::Task(error.clone())),
                error
            );
        }
        // Executor-only verdicts have no Host category and stay InvalidArgument.
        for (error, message) in [
            (
                TreeTransformAdmissionError::Placement("source"),
                "tree transform placement admission failed",
            ),
            (
                TreeTransformAdmissionError::Context("source"),
                "tree transform context admission failed",
            ),
            (
                TreeTransformAdmissionError::Capability("matrix"),
                "tree transform capability admission failed",
            ),
            (
                TreeTransformAdmissionError::Aliasing,
                "tree transform storage regions overlap",
            ),
        ] {
            assert_eq!(
                OperationError::from(error),
                OperationError::InvalidArgument { message }
            );
        }
    }

    #[test]
    fn mock_executes_scalar_and_matrix_overwrite_and_axpby_deterministically() {
        let (structure, scalar) = scalar_fixture();
        let mut executor = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![f64::NAN; 4]);
        executor
            .execute(&scalar, &structure, 1.0, DestinationMode::Overwrite, 1)
            .unwrap();
        assert_eq!(executor.destination, [2.0, 4.0, 9.0, 12.0]);
        assert_eq!((executor.submissions, executor.writes), (2, 4));

        let (structure, matrix) = multi_fixture();
        let mut executor = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![10.0; 4]);
        assert_eq!(executor.placement, Placement::Cuda(3));
        executor
            .execute(&matrix, &structure, 2.0, DestinationMode::Axpby(0.5), 2)
            .unwrap();
        assert_eq!(executor.converted_coefficients, [0.0, 1.0, 1.0, 0.0]);
        assert!(executor.coefficient_readiness.is_some());
        assert_eq!(executor.destination, [11.0, 13.0, 7.0, 9.0]);
        assert_eq!((executor.submissions, executor.writes), (1, 4));
    }

    #[test]
    fn every_mock_admission_and_preparation_failure_is_pre_submit_atomic() {
        let (structure, transform) = multi_fixture();
        for fault in [
            AdmissionFault::DestinationPlacement,
            AdmissionFault::DestinationContext,
            AdmissionFault::DestinationLength,
            AdmissionFault::ActiveBeyondCapacity,
            AdmissionFault::MissingStrided,
            AdmissionFault::MissingMatrix,
            AdmissionFault::WrongScalar,
            AdmissionFault::Overlap,
            AdmissionFault::PackedSourceCapacity,
            AdmissionFault::PackedDestinationCapacity,
            AdmissionFault::CoefficientCapacity,
            AdmissionFault::FusedCapacity,
            AdmissionFault::WorkspacePlacement,
            AdmissionFault::WorkspaceContext,
            AdmissionFault::StaleReadiness,
            AdmissionFault::WrongReadinessScalar,
            AdmissionFault::WrongReadinessContext,
        ] {
            let mut executor = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![10.0; 4]);
            executor.fault = fault;
            assert!(matches!(
                executor.execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 2),
                Err(MockError::Admission(_))
            ));
            assert_eq!((executor.submissions, executor.writes), (0, 0));
            assert_eq!(executor.destination, [10.0; 4]);
        }

        let mut executor = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![10.0; 4]);
        executor.coefficient_readiness = Some(CoefficientReadiness {
            structure_and_layout: transform.task_view().unwrap().admission_identity(),
            scalar: TypeId::of::<f64>(),
            context: executor.context,
        });
        executor.fail_preparation = true;
        assert_eq!(
            executor.execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 1),
            Err(MockError::Preparation)
        );
        assert_eq!((executor.submissions, executor.writes), (0, 0));
        assert_eq!(executor.destination, [10.0; 4]);
        assert!(executor.coefficient_readiness.is_none());
    }

    #[test]
    fn mock_accepts_disjoint_shared_regions_domains_and_read_read_overlap() {
        let (structure, transform) = multi_fixture();
        let mut disjoint = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.0; 4]);
        disjoint.destination_region = region(1, 1, 0, 32);
        disjoint.source_region = region(1, 1, 32, 32);
        disjoint
            .execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 1)
            .unwrap();

        let mut domains = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.0; 4]);
        domains.destination_region = region(1, 1, 0, 32);
        domains.source_region = region(2, 1, 0, 32);
        domains
            .execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 1)
            .unwrap();

        let mut read_overlap = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.0; 4]);
        read_overlap.fault = AdmissionFault::ReadReadOverlap;
        read_overlap
            .execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 1)
            .unwrap();
    }

    #[test]
    fn mock_accepts_empty_storage() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let mut executor = MockOpaqueExecutor::new(Vec::new(), Vec::new());
        executor.source_region = StorageRegion::Empty;
        executor.destination_region = StorageRegion::Empty;
        executor
            .execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 1)
            .unwrap();
        assert_eq!((executor.submissions, executor.writes), (1, 0));
    }

    #[test]
    fn worker_overflow_is_pre_submit_and_post_submit_failure_does_not_roll_back() {
        let strided = Arc::new(
            BlockStructure::from_blocks_with_rank(
                2,
                vec![BlockSpec::new(vec![2, 2], vec![1, 3], 0).unwrap()],
            )
            .unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &strided,
            &strided,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let mut overflow = MockOpaqueExecutor::new(vec![1.0; 5], vec![0.0; 5]);
        assert_eq!(
            overflow.execute(
                &transform,
                &strided,
                1.0,
                DestinationMode::Overwrite,
                usize::MAX,
            ),
            Err(MockError::Admission(
                TreeTransformAdmissionError::ArithmeticOverflow
            ))
        );
        assert_eq!((overflow.submissions, overflow.writes), (0, 0));

        let (structure, transform) = scalar_fixture();
        let mut partial = MockOpaqueExecutor::new(vec![1.0, 2.0, 3.0, 4.0], vec![10.0; 4]);
        partial.fail_after_submission = Some(1);
        assert_eq!(
            partial.execute(&transform, &structure, 1.0, DestinationMode::Overwrite, 1,),
            Err(MockError::Backend)
        );
        assert_eq!(partial.destination, [2.0, 4.0, 10.0, 10.0]);
        assert_eq!((partial.submissions, partial.writes), (1, 2));
    }
}
