//! The recording writer, the forwarding executor and the fixtures the lane
//! replay tests (native, barrier and member) share.

use super::*;
use crate::kernel_adapter::movement::{Store, Write};
use crate::transform_structure::TreeTransformDirectItem;
use std::sync::Mutex;
use std::thread::ThreadId;
use symtn_dense::CpuEngine;
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

use super::super::writers::{crop_role, inactive_role, DestinationWriter, ReplayParts};

/// One destination element as the recording writer leaves it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Mark {
    pub(super) writes: u32,
    pub(super) worker: Option<ThreadId>,
}

/// A movement store that records each write's thread instead of a value.
enum Record {}

impl Store<f64> for Record {
    type Slot = Mark;

    fn put(slot: &mut Mark, _: f64) {
        slot.writes += 1;
        slot.worker = Some(std::thread::current().id());
    }
}

/// A writer that addresses exactly the elements the real writers address,
/// whole or cropped to a band, and records who wrote each.
pub(super) struct Recorder<'a> {
    pub(super) parts: ReplayParts<'a, f64>,
}

impl DestinationWriter<Mark, f64> for Recorder<'_> {
    type Kernels = ();

    fn kernels(&self) {}

    fn overwrites(&self) -> bool {
        true
    }

    fn direct(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        src: &[f64],
        dst: &mut [Mark],
        dst_start: isize,
        band: Option<super::super::writers::Band>,
        _: crate::host_pool::Exec<'_>,
        _: &crate::kernel_adapter::sealed::LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let mut cropped = None;
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let (role, shift) = inactive_role(layouts, layout, band, &mut cropped)?;
                span::<Record, _, f64>(
                    dst,
                    &[0.0],
                    role,
                    layouts.entry(layout).offset - dst_start + shift,
                    0,
                    &mut index[..role.dims().len()],
                    &Write(|value: f64| value),
                );
            }
            TreeTransformDirectItem::Single { block, .. } => {
                let (dst_layout, src_layout, _) = scheduled_single(self.parts.blocks, block)?;
                let role = layouts.role(dst_layout)?;
                let (role, dst_shift, src_shift) = match band {
                    None => (role, 0, 0),
                    Some(band) => crop_role(role, band, &mut cropped)?,
                };
                span::<Record, _, f64>(
                    dst,
                    src,
                    role,
                    layouts.entry(dst_layout).offset - dst_start + dst_shift,
                    layouts.entry(src_layout).offset + src_shift,
                    &mut index[..role.dims().len()],
                    &Write(|value: f64| value),
                );
            }
        }
        Ok(())
    }

    fn scatter(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        dst_layout: usize,
        packed: &[f64],
        packed_offset: usize,
        dst: &mut [Mark],
        dst_start: isize,
        band: Option<super::super::writers::Band>,
        _: crate::host_pool::Exec<'_>,
        _: &crate::kernel_adapter::sealed::LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let mut cropped = None;
        let role = layouts.role(dst_layout)?;
        let (role, dst_shift, src_shift) = match band {
            None => (role, 0, 0),
            Some(band) => crop_role(role, band, &mut cropped)?,
        };
        span::<Record, _, f64>(
            dst,
            packed,
            role,
            layouts.entry(dst_layout).offset - dst_start + dst_shift,
            offset_to_isize(packed_offset)? + src_shift,
            &mut index[..role.dims().len()],
            &Write(|value: f64| value),
        );
        Ok(())
    }
}

/// Destination element addresses of a block `(offset, dims, strides)`.
pub(super) fn addresses(offset: usize, dims: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut out = vec![offset];
    for (&dim, &stride) in dims.iter().zip(strides) {
        out = out
            .iter()
            .flat_map(|&base| (0..dim).map(move |i| base + i * stride))
            .collect();
    }
    out
}

/// Columns of the interleaved bandable blocks.
pub(super) const COLUMNS: usize = 32;

/// Two `[2, COLUMNS]` blocks of row stride 4 interleave in one atom
/// bandable by its columns, a Single and a Multi group's first column; the
/// group's second column, `[2, COLUMNS]` of row stride 2, fuses to one
/// contiguous atom; with `injective`, a Single over the `[3, 2]` / `[2, 3]`
/// layout, which no band certificate can split, forms a second component.
/// Returns the destination blocks `(offset, dims, strides)`, the structures
/// and the transform.
#[allow(clippy::type_complexity)]
pub(super) fn interleaved_bandable(
    injective: bool,
) -> (
    Vec<(usize, [usize; 2], [usize; 2])>,
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let mut blocks = vec![
        (0, [2, COLUMNS], [1, 4]),
        (2, [2, COLUMNS], [1, 4]),
        (4 * COLUMNS, [2, COLUMNS], [1, 2]),
    ];
    if injective {
        blocks.push((6 * COLUMNS, [3, 2], [2, 3]));
    }
    let dst = Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            blocks
                .iter()
                .enumerate()
                .map(|(key, &(offset, dims, strides))| {
                    BlockSpec::with_key(
                        BlockKey::ordinal(key),
                        dims.to_vec(),
                        strides.to_vec(),
                        offset,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap(),
    );
    let src = Arc::new(
        BlockStructure::packed_column_major(2, blocks.iter().map(|&(_, dims, _)| dims.to_vec()))
            .unwrap(),
    );
    let mut specs = vec![
        TreeTransformBlockSpec::single(0, 0, -2.0),
        TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0]),
    ];
    if injective {
        specs.push(TreeTransformBlockSpec::single(3, 3, 1.0));
    }
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    (blocks, dst, src, transform)
}

/// The one-writer contract of an [`interleaved_bandable`]`(true)` replay
/// on four workers: every addressed element of `blocks` is written exactly
/// once and nothing else is; a band of the interleaved atom is never split,
/// its bands go to more than one worker, and the injective `[3, 2]` /
/// `[2, 3]` block is written whole by one worker.
#[allow(clippy::type_complexity)]
pub(super) fn assert_one_writer_per_element(
    blocks: &[(usize, [usize; 2], [usize; 2])],
    marks: &[Mark],
) {
    let len = marks.len();
    let mut addressed = vec![false; len];
    for &(offset, dims, strides) in blocks {
        for at in addresses(offset, &dims, &strides) {
            addressed[at] = true;
        }
    }
    for (at, mark) in marks.iter().enumerate() {
        assert_eq!(mark.writes, u32::from(addressed[at]), "element {at}");
    }
    // Atom `[0, 4 COLUMNS)`: band `k` is elements `4k..4k + 4`.
    let mut atom_workers = Vec::new();
    for band in 0..COLUMNS {
        let workers = (4 * band..4 * band + 4)
            .map(|at| marks[at].worker)
            .collect::<Vec<_>>();
        assert!(workers.iter().all(|w| *w == workers[0]), "band {band}");
        atom_workers.push(workers[0]);
    }
    atom_workers.sort_unstable_by_key(|w| format!("{w:?}"));
    atom_workers.dedup();
    assert!(atom_workers.len() > 1, "the bandable atom is not banded");
    let injective = addresses(6 * COLUMNS, &[3, 2], &[2, 3])
        .into_iter()
        .map(|at| marks[at].worker)
        .collect::<Vec<_>>();
    assert!(injective.iter().all(|w| w.is_some() && *w == injective[0]));
}

/// An eager-policy engine of `threads` workers under `strategy`.
pub(super) fn engine(threads: usize, strategy: CpuBatchStrategy) -> CpuEngine {
    crate::host_pool::eager_test_engine(threads).with_batch_policy(CpuBatchPolicy::new(strategy))
}

/// The error a test executor injects for recoupling group `group`.
pub(super) fn injected(group: usize) -> DenseError {
    DenseError::Unsupported {
        op: "spy",
        message: format!("group {group}"),
    }
}

/// A custom executor on the default batch entry forwarding to the built-in
/// executor, without lane GEMMs unless `lanes`: each GEMM (single product
/// or lane GEMM) records its label and fails when that label is in `fail`.
/// The label is the coefficients' first value (the fixture's group label)
/// plus ten times the member, read as the packed source's first value over
/// [`MEMBER_SPAN`] (the fixtures give member `b` the source values
/// `b MEMBER_SPAN + 1, ...`).
#[derive(Default)]
pub(super) struct Forwarding {
    pub(super) inner: DefaultDenseExecutor,
    pub(super) lanes: bool,
    pub(super) fail: Vec<usize>,
    pub(super) attempts: Mutex<Vec<usize>>,
}

/// The source value span of one member in the failure fixtures.
pub(super) const MEMBER_SPAN: f64 = 1000.0;

impl Forwarding {
    /// The labels attempted, in submission order (lane GEMMs in completion
    /// order).
    pub(super) fn attempts(&self) -> Vec<usize> {
        self.attempts.lock().unwrap().clone()
    }

    /// Records the GEMM over `lhs` and `rhs` and fails it when its label is
    /// in `fail`.
    fn attempt(&self, lhs: Option<f64>, rhs: Option<f64>) -> Result<(), DenseError> {
        let Some(label) = rhs else {
            return Ok(());
        };
        let member = lhs.map_or(0, |value| (value / MEMBER_SPAN) as usize);
        let label = label as usize + 10 * member;
        self.attempts.lock().unwrap().push(label);
        if self.fail.contains(&label) {
            return Err(injected(label));
        }
        Ok(())
    }
}

fn first(read: &DenseRead<'_>) -> Option<f64> {
    match read {
        DenseRead::F64(view) => view.data().get(view.offset()).copied(),
        _ => None,
    }
}

impl DenseExecutor for Forwarding {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(engine, input)
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(engine, input)
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(engine, input)
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner
            .dot_general_into(engine, output, lhs, rhs, config)
    }

    fn matmul_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.attempt(first(&lhs), first(&rhs))?;
        self.inner
            .matmul_axpby_into(engine, output, lhs, rhs, alpha, beta)
    }

    fn supports_lane_gemms(&self) -> bool {
        self.lanes
    }

    fn with_lane_gemms(
        &mut self,
        engine: &mut dyn Engine,
        dtype: symtn_dense::DenseDType,
        jobs: &[symtn_dense::lanes::LaneGemmShape],
        body: &mut symtn_dense::lanes::LaneGemmBody<'_>,
    ) -> Result<(), DenseError> {
        let Self {
            inner,
            fail,
            attempts,
            ..
        } = self;
        let spy = Forwarding {
            fail: core::mem::take(fail),
            attempts: Mutex::new(core::mem::take(attempts.get_mut().unwrap())),
            ..Forwarding::default()
        };
        let entered = inner.with_lane_gemms(engine, dtype, jobs, &mut |exec, gemms| {
            body(exec, &Labelled { gemms, spy: &spy })
        });
        *fail = spy.fail;
        *attempts.get_mut().unwrap() = spy.attempts.into_inner().unwrap();
        entered
    }
}

/// Lane GEMMs labelled and failed as [`Forwarding`]'s products are.
struct Labelled<'a> {
    gemms: &'a dyn symtn_dense::lanes::LaneGemms,
    spy: &'a Forwarding,
}

impl symtn_dense::lanes::LaneGemms for Labelled<'_> {
    fn run(
        &self,
        exec: symtn_dense::lanes::Exec<'_>,
        job: usize,
        io: symtn_dense::lanes::LaneGemmIo<'_>,
    ) -> Result<(), DenseError> {
        if let symtn_dense::lanes::LaneGemmIo::F64 { lhs, rhs, .. } = &io {
            self.spy
                .attempt(lhs.first().copied(), rhs.first().copied())?;
        }
        self.gemms.run(exec, job, io)
    }

    fn estimate(&self, job: usize) -> symtn_dense::lanes::Estimate {
        self.gemms.estimate(job)
    }
}

/// Two Multi groups `1` and `2` share the interleaved atom `[0, 63]` (one
/// component of two GEMMs) and group `3` owns `[64, 95]` (a component of
/// one); each group's label is its coefficients' first value. Blocks of 32
/// elements keep every stage's estimate nonzero, so forced lanes spread the
/// components.
pub(super) fn two_lane_fixture() -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let dst = Arc::new(custom_structure(vec![
        block(0, 32, 2, 0),
        block(1, 32, 2, 1),
        block(2, 32, 1, 64),
    ]));
    let src = Arc::new(BlockStructure::packed_column_major(1, vec![vec![32]; 6]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::multi(vec![0], vec![0, 1], vec![1.0, -1.0]),
            TreeTransformBlockSpec::multi(vec![1], vec![2, 3], vec![2.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![2], vec![4, 5], vec![3.0, 1.0]),
        ],
    )
    .unwrap();
    (dst, src, transform)
}

/// The group message of an injected error.
pub(super) fn group_of(error: OperationError) -> String {
    match error {
        OperationError::Dense(DenseError::Unsupported { op: "spy", message }) => message,
        other => panic!("not an injected error: {other:?}"),
    }
}
