use super::*;

/// One sub-slot's share of one atom of a component stage: the whole atom,
/// or a run of its bands.
pub(super) struct Piece<'a, X> {
    slot: u32,
    sub: u32,
    band: Option<Band>,
    /// First destination element of `data`.
    start: isize,
    data: &'a mut [X],
}

/// One band sub-slot's pieces, traversal index and outcome.
pub(super) struct SubSlot<'c, 'd, X> {
    pieces: &'c mut [Piece<'d, X>],
    index: &'c mut [usize],
    result: Result<(), OperationError>,
}

pub(super) type SubCell<'c, 'd, X> = RwLock<Option<SubSlot<'c, 'd, X>>>;

/// One pack range's writer, columns, traversal index and outcome.
pub(super) struct PackSlot<'c, X> {
    writer: PackColumnWriter<'c, X>,
    columns: &'c [TreeTransformPackReplay],
    /// Packed offset of the writer's first slot.
    start: usize,
    index: &'c mut [usize],
    result: Result<(), OperationError>,
}

pub(super) type PackCell<'c, X> = RwLock<Option<PackSlot<'c, X>>>;

/// The parts a movement fork opens once its bands or pack ranges have capped
/// the selected width to `parts`: under `Auto`, one (inline) when the
/// stage's `estimate` does not pay for a section of `parts` participants on
/// `token`, the token and thread that fork; a forced policy keeps its parts.
fn fork_parts(
    token: &tenferro_cpu::threadteam::Exec<'_>,
    policy: &lanes::BatchPolicy,
    estimate: lanes::Estimate,
    parts: usize,
) -> usize {
    match core::num::NonZeroUsize::new(parts) {
        Some(width)
            if parts > 1
                && policy.strategy == tenferro_cpu::CpuBatchStrategy::Auto
                && !token.pays(estimate.serial, estimate.sections.max(1), width) =>
        {
            1
        }
        _ => parts,
    }
}

impl<W, D, C> Replay<'_, W, D, C>
where
    W: Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    /// Whether an overwrite of component `component` runs whole on its one
    /// planned Single on `exec`, never banded: a group-free component whose
    /// one item is a Single with a compiled plan that [`admitted_plan`]
    /// admits on `exec` through `gate` (the same check its writer then
    /// applies).
    fn whole_single(&self, component: usize, exec: &Exec<'_>, gate: &LeafGate) -> bool {
        let schedule = self.schedule;
        if !schedule.component_group_list(component).is_empty() {
            return false;
        }
        let [TreeTransformDirectItem::Single { block, .. }] = *schedule.component_direct(component)
        else {
            return false;
        };
        let layouts = self.parts.layouts;
        scheduled_single(self.parts.blocks, block)
            .and_then(|(dst_layout, _, _)| {
                let elements = layouts.entry(dst_layout).element_count;
                layouts
                    .role(dst_layout)
                    .map(|role| admitted_plan(role, elements, exec, gate).is_some())
            })
            .unwrap_or(false)
    }

    /// Movement stage `movement` of component `component` on `exec`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn movement<T>(
        &self,
        component: usize,
        movement: Movement<'_, D>,
        atoms: &mut [&mut [T]],
        pieces: &mut Vec<Piece<'static, u8>>,
        subs: &mut Vec<SubCell<'static, 'static, u8>>,
        index: &mut [usize],
        estimate: lanes::Estimate,
        exec: lanes::Exec<'_>,
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
    {
        let schedule = self.schedule;
        let empty = match movement {
            Movement::Direct { .. } => schedule.component_direct(component).is_empty(),
            Movement::Scatter { group, .. } => schedule.scatter_groups[group].columns.is_empty(),
        };
        if empty {
            return Ok(());
        }
        let policy = *exec.batch_policy();
        let token = exec.into_token();
        let width = token.budget().get();
        // The stage's gate on its own thread; each forked sub-slot below
        // builds its own, as a gate is not `Sync`.
        let gate = LeafGate::new(self.parts.layouts.max_planned_elements());
        let bands = if self.writer.overwrites() && self.whole_single(component, &token, &gate) {
            1
        } else {
            let bands = width.min(schedule.components[component].units).max(1);
            fork_parts(&token, &policy, estimate, bands)
        };
        let mut list: Vec<Piece<'_, T>> = recycle(core::mem::take(pieces));
        let result = self
            .band(component, bands, atoms, &mut list)
            .and_then(|()| {
                if bands == 1 {
                    return self.write_pieces(
                        component,
                        movement,
                        &mut list,
                        &mut index[..self.rank],
                        token,
                        Some(&gate),
                    );
                }
                let mut cells: Vec<SubCell<'_, '_, T>> = recycle(core::mem::take(subs));
                let mut rest = &mut list[..];
                let mut indices = &mut index[..bands * self.rank];
                for sub in 0..bands {
                    let own = rest
                        .iter()
                        .take_while(|piece| piece.sub as usize == sub)
                        .count();
                    let (own, tail) = core::mem::take(&mut rest).split_at_mut(own);
                    rest = tail;
                    let (traversal, tail) = core::mem::take(&mut indices).split_at_mut(self.rank);
                    indices = tail;
                    cells.push(RwLock::new(Some(SubSlot {
                        pieces: own,
                        index: traversal,
                        result: Ok(()),
                    })));
                }
                token.fork(bands, |sub, child| {
                    let mut cell = cells[sub].write().unwrap_or_else(PoisonError::into_inner);
                    if let Some(SubSlot {
                        pieces,
                        index,
                        result,
                    }) = cell.as_mut()
                    {
                        *result =
                            self.write_pieces(component, movement, pieces, index, child, None);
                    }
                });
                let mut first = Ok(());
                for cell in &mut cells {
                    if let Some(slot) = cell.get_mut().unwrap_or_else(PoisonError::into_inner) {
                        if first.is_ok() {
                            first = core::mem::replace(&mut slot.result, Ok(()));
                        }
                    }
                }
                *subs = recycle(cells);
                first
            });
        *pieces = recycle(list);
        result
    }

    /// Writes `movement`'s items of each of `pieces` on `exec`: the items of
    /// the piece's atom, cropped to its bands, admitted through `gate`, or
    /// through a gate of this sub-slot's own without one.
    fn write_pieces<T>(
        &self,
        component: usize,
        movement: Movement<'_, D>,
        pieces: &mut [Piece<'_, T>],
        index: &mut [usize],
        mut exec: Exec<'_>,
        gate: Option<&LeafGate>,
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
    {
        let own;
        let gate = match gate {
            Some(gate) => gate,
            None => {
                own = LeafGate::new(self.parts.layouts.max_planned_elements());
                &own
            }
        };
        let schedule = self.schedule;
        // Why a fresh adapter and stride vector per stage, not lane-retained
        // ones: every write here passes its compiled role, the compiled zero
        // source strides and this sub-slot's traversal index, so neither is
        // ever grown, and both are built without allocating.
        let mut kernels = self.writer.kernels();
        let mut zero_strides = Vec::new();
        for piece in pieces {
            match movement {
                Movement::Direct { src } => {
                    let direct = schedule.component_direct(component);
                    let first = direct.partition_point(|item| item.slot() < piece.slot);
                    for &item in direct[first..]
                        .iter()
                        .take_while(|item| item.slot() == piece.slot)
                    {
                        self.writer.direct(
                            &mut kernels,
                            &mut zero_strides,
                            item,
                            src,
                            piece.data,
                            piece.start,
                            piece.band,
                            exec.reborrow(),
                            gate,
                            index,
                        )?;
                    }
                }
                Movement::Scatter { group, packed } => {
                    let job = &self.parts.plan.jobs()[group];
                    let columns =
                        &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()];
                    let first = columns.partition_point(|column| column.slot < piece.slot);
                    for column in columns[first..]
                        .iter()
                        .take_while(|column| column.slot == piece.slot)
                    {
                        self.writer.scatter(
                            &mut kernels,
                            &mut zero_strides,
                            column.dst_layout,
                            packed,
                            column.packed_offset - job.dst_offset,
                            piece.data,
                            piece.start,
                            piece.band,
                            exec.reborrow(),
                            gate,
                            index,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Splits component `component`'s atom slices into `pieces` for `bands`
    /// sub-slots, reborrowing them for one stage.
    ///
    /// At one sub-slot every atom is one whole piece. Otherwise the
    /// component's band units (each band of a bandable atom, each
    /// indivisible atom) are numbered in address order; unit `u`, with `P`
    /// destination elements in the units before it out of `total`, goes to
    /// sub-slot `min(bands - 1, floor(P bands / total))`: contiguous ranges
    /// balanced by written elements, the same map for every stage of the
    /// component at that sub-slot count. A run of one atom's units on one
    /// sub-slot is one piece. O(atoms + bands).
    fn band<'x, T>(
        &self,
        component: usize,
        bands: usize,
        atoms: &'x mut [&mut [T]],
        pieces: &mut Vec<Piece<'x, T>>,
    ) -> Result<(), OperationError> {
        let schedule = self.schedule;
        let range = schedule.component_atoms(component);
        pieces.reserve(range.len() + bands);
        if bands == 1 {
            for (slot, data) in range.zip(atoms.iter_mut()) {
                pieces.push(Piece {
                    slot: slot as u32,
                    sub: 0,
                    band: None,
                    start: schedule.atoms[slot].lo,
                    data,
                });
            }
            return Ok(());
        }
        let total = schedule.atoms[range.clone()]
            .iter()
            .fold(0u128, |sum, atom| sum + atom.weight as u128)
            .max(1);
        let width = bands as u128;
        let sub_of = |before: u128| ((before * width / total) as usize).min(bands - 1);
        let mut before = 0u128;
        for (slot, data) in range.zip(atoms.iter_mut()) {
            let atom = schedule.atoms[slot];
            let band = schedule.bands[slot];
            let data: &'x mut [T] = data;
            if band.extent < 2 {
                pieces.push(Piece {
                    slot: slot as u32,
                    sub: sub_of(before) as u32,
                    band: None,
                    start: atom.lo,
                    data,
                });
                before += atom.weight as u128;
                continue;
            }
            let (extent, stride) = (band.extent, band.stride);
            let unit = (atom.weight / extent) as u128;
            let mut rest = data;
            let mut k0 = 0usize;
            while k0 < extent {
                let sub = sub_of(before + k0 as u128 * unit);
                // The first band of a later sub-slot: the smallest `k` with
                // `(before + k unit) bands >= (sub + 1) total`.
                let k1 = if sub + 1 >= bands || unit == 0 {
                    extent
                } else {
                    let need = (sub as u128 + 1) * total;
                    let have = before * width;
                    let step = unit * width;
                    (need.saturating_sub(have).div_ceil(step) as usize).clamp(k0 + 1, extent)
                };
                let end = (k1 - k0)
                    .checked_mul(stride)
                    .map_or(rest.len(), |len| len.min(rest.len()));
                let (data, tail) = core::mem::take(&mut rest).split_at_mut(end);
                rest = tail;
                pieces.push(Piece {
                    slot: slot as u32,
                    sub: sub as u32,
                    band: Some((k0, k1)),
                    start: atom.lo + offset_to_isize(k0 * stride)?,
                    data,
                });
                k0 = k1;
            }
            before += atom.weight as u128;
        }
        Ok(())
    }

    /// Group `group`'s pack of `src` into `source` on `exec`: one flat fork over
    /// `min(width, columns)` contiguous column ranges. The pack ranges first
    /// cap the selected width; under `Auto` the actual-fork pays check
    /// ([`fork_parts`]) may then choose inline execution; a forced policy
    /// keeps its parts.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pack(
        &self,
        group: usize,
        src: &[D],
        source: &mut HostScratchBuffer<D>,
        packs: &mut Vec<PackCell<'static, u8>>,
        index: &mut [usize],
        estimate: lanes::Estimate,
        exec: lanes::Exec<'_>,
    ) -> Result<(), OperationError> {
        let job = &self.parts.plan.jobs()[group];
        let len = job.rows * job.contracted;
        let columns = &self.schedule.pack_columns[self.schedule.scatter_groups[group].pack.clone()];
        let policy = *exec.batch_policy();
        let token = exec.into_token();
        let parts = token.budget().get().min(columns.len());
        let parts = fork_parts(&token, &policy, estimate, parts);
        source.overwrite_with_columns(len, |mut writer| {
            if parts <= 1 {
                return self.pack_columns(&mut writer, columns, job.lhs_offset, src, index);
            }
            let mut cells: Vec<PackCell<'_, D>> = recycle(core::mem::take(packs));
            let mut indices = &mut index[..parts * self.rank];
            let mut push = |writer, range: core::ops::Range<usize>| {
                let columns = &columns[range];
                let (traversal, tail) = core::mem::take(&mut indices).split_at_mut(self.rank);
                indices = tail;
                cells.push(RwLock::new(Some(PackSlot {
                    writer,
                    columns,
                    start: columns[0].packed_offset,
                    index: traversal,
                    result: Ok(()),
                })));
            };
            let (mut rest, mut lo) = (writer, 0);
            for part in 1..parts {
                let hi = part * columns.len() / parts;
                let (own, tail) =
                    rest.split_at(columns[hi].packed_offset - columns[lo].packed_offset)?;
                push(own, lo..hi);
                (rest, lo) = (tail, hi);
            }
            push(rest, lo..columns.len());
            token.fork(parts, |part, _| {
                let mut cell = cells[part].write().unwrap_or_else(PoisonError::into_inner);
                if let Some(PackSlot {
                    writer,
                    columns,
                    start,
                    index,
                    result,
                }) = cell.as_mut()
                {
                    *result = self.pack_columns(writer, columns, *start, src, index);
                }
            });
            let mut first = Ok(());
            for cell in &mut cells {
                if let Some(slot) = cell.get_mut().unwrap_or_else(PoisonError::into_inner) {
                    if first.is_ok() {
                        first = core::mem::replace(&mut slot.result, Ok(()));
                    }
                }
            }
            *packs = recycle(cells);
            first
        })
    }

    /// Packs `columns` through `writer`, whose first slot is packed offset
    /// `start`.
    fn pack_columns(
        &self,
        writer: &mut PackColumnWriter<'_, D>,
        columns: &[TreeTransformPackReplay],
        start: usize,
        src: &[D],
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        with_conjugation!(D, self.parts.storage_conjugate, |map| {
            for column in columns {
                writer.write_column(
                    column.packed_offset - start,
                    layouts.role(column.src_layout)?,
                    src,
                    layouts.entry(column.src_layout).offset,
                    index,
                    map,
                )?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tenferro_cpu::threadteam::{self, DispatchCosts, WidthCosts};
    use tenferro_cpu::Pool;

    use super::{fork_parts, lanes};

    /// What: the check reads the cost row of the token that forks: a 200 ns
    /// section pays for two parts on a token held outside the pool (1 ns)
    /// and not on one held by a pool worker (1000 ns), where it runs inline.
    #[test]
    fn a_fork_is_checked_on_the_cost_row_of_the_token_that_forks() {
        let (cheap, dear) = (
            WidthCosts::flat(Duration::from_nanos(1)),
            WidthCosts::flat(Duration::from_nanos(1000)),
        );
        let pool = Pool::builder()
            .threads(2)
            .dispatch_costs(DispatchCosts::new(dear, dear, cheap))
            .build()
            .unwrap();
        let policy = lanes::BatchPolicy::default();
        let estimate = lanes::Estimate {
            serial: Duration::from_nanos(200),
            payback: None,
            sections: 1,
            widen: true,
        };
        let root = threadteam::Exec::new(&pool, 2).unwrap();
        assert_eq!(fork_parts(&root, &policy, estimate, 2), 2);
        let inner = std::sync::Mutex::new(Vec::new());
        // A fresh two-slot token on each worker: its row is the pool's own.
        root.fork(2, |_, _| {
            let token = threadteam::Exec::new(&pool, 2).unwrap();
            inner
                .lock()
                .unwrap()
                .push(fork_parts(&token, &policy, estimate, 2));
        });
        assert_eq!(inner.into_inner().unwrap(), [1, 1]);
    }
}
