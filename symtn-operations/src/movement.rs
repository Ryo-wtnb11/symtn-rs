//! Structural movement counts. Coefficients and payload values never enter them.
//! Estimates saturate so metadata-only structures keep replay's validation limits.

use crate::{OperationError, TreeTransformLayoutTable};

// Structural classes: real-map contiguous/strided, inactive contiguous/strided.
pub(crate) const CLASSES: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Point {
    pub runs: usize,
    pub elements: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    pub items: u64,
    pub runs: u64,
    pub elements: u64,
}

impl Counts {
    pub(crate) fn add(self, rhs: Self) -> Result<Self, OperationError> {
        Ok(Self {
            items: add(self.items, rhs.items)?,
            runs: add(self.runs, rhs.runs)?,
            elements: add(self.elements, rhs.elements)?,
        })
    }
}

fn add(a: u64, b: u64) -> Result<u64, OperationError> {
    Ok(a.saturating_add(b))
}

/// Cumulative structural components at an executed-job boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Job {
    pub counts: [Counts; CLASSES],
    pub planned: [Counts; CLASSES],
}

impl Job {
    pub(crate) fn add(self, rhs: Self) -> Result<Self, OperationError> {
        let mut result = self;
        for (left, right) in result.counts.iter_mut().zip(rhs.counts) {
            *left = left.add(right)?;
        }
        for (left, right) in result.planned.iter_mut().zip(rhs.planned) {
            *left = left.add(right)?;
        }
        Ok(result)
    }
}

/// Saturated metadata stays conservative when subtracting cumulative counts.
pub(crate) fn job_range(prefix: &[Job], range: core::ops::Range<usize>) -> Job {
    let end = prefix[range.end - 1];
    let start = range
        .start
        .checked_sub(1)
        .map_or(Job::default(), |i| prefix[i]);
    let difference = |end: [Counts; CLASSES], start: [Counts; CLASSES]| {
        core::array::from_fn(|c| {
            let subtract = |a: u64, b: u64| if a == u64::MAX { a } else { a - b };
            Counts {
                items: subtract(end[c].items, start[c].items),
                runs: subtract(end[c].runs, start[c].runs),
                elements: subtract(end[c].elements, start[c].elements),
            }
        })
    };
    Job {
        counts: difference(end.counts, start.counts),
        planned: difference(end.planned, start.planned),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Summary {
    pub counts: [Counts; CLASSES],
    pub planned: [Counts; CLASSES],
    /// Smallest retained planned role; zero when none is planned.
    pub planned_min: usize,
    /// Cumulative counts in executed-job order, only where a decision reads them.
    pub prefix: Vec<Job>,
    /// Payload elements touched, source and destination included.
    pub working_set: u64,
    pub traffic: u64,
    pub destination_reads: u64,
    // Per-buffer address envelopes: exact when coalesced, otherwise upper bounds.
    extents: [Option<(isize, isize)>; 4],
}

impl Summary {
    pub(crate) fn push(&mut self, class: usize, point: Point) -> Result<Counts, OperationError> {
        let count = Counts {
            items: 1,
            runs: u64::try_from(point.runs).unwrap_or(u64::MAX),
            elements: u64::try_from(point.elements).unwrap_or(u64::MAX),
        };
        self.counts[class] = self.counts[class].add(count)?;
        let touched = count.elements.saturating_mul([2, 2, 1, 1][class]);
        self.traffic = add(self.traffic, touched)?;
        Ok(count)
    }

    pub(crate) fn job(&self) -> Job {
        Job {
            counts: self.counts,
            planned: self.planned,
        }
    }

    pub(crate) fn record_job(&mut self, job: &[Counts; CLASSES]) -> Result<(), OperationError> {
        if job.iter().any(|count| count.items != 0) {
            self.prefix.push(self.job());
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), OperationError> {
        self.prefix.shrink_to_fit();
        self.working_set = self
            .extents
            .iter()
            .flatten()
            .try_fold(0, |sum, &(lo, hi)| {
                let span = u64::try_from(hi as i128 - lo as i128 + 1).unwrap_or(u64::MAX);
                add(sum, span)
            })?;
        Ok(())
    }

    /// Stack-only aggregation for a chunk; no hull rebuilding or copying.
    pub(crate) fn totals<'a>(
        summaries: impl IntoIterator<Item = &'a Self>,
    ) -> Result<Self, OperationError> {
        let mut result = Self::default();
        for summary in summaries {
            result.add_totals(summary)?;
        }
        result.finish()?;
        Ok(result)
    }

    pub(crate) fn add_totals(&mut self, rhs: &Self) -> Result<(), OperationError> {
        for (count, &other) in self.counts.iter_mut().zip(&rhs.counts) {
            *count = count.add(other)?;
        }
        for (count, &other) in self.planned.iter_mut().zip(&rhs.planned) {
            *count = count.add(other)?;
        }
        if rhs.planned_min != 0 {
            self.planned_min = if self.planned_min == 0 {
                rhs.planned_min
            } else {
                self.planned_min.min(rhs.planned_min)
            };
        }
        self.traffic = add(self.traffic, rhs.traffic)?;
        self.destination_reads = add(self.destination_reads, rhs.destination_reads)?;
        for (buffer, extent) in rhs.extents.iter().enumerate() {
            if let Some((lo, hi)) = *extent {
                self.touch((buffer as u8, lo, hi));
            }
        }
        Ok(())
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.prefix.capacity() * core::mem::size_of::<Job>()
    }

    pub(crate) fn logical_traffic(&self, read_destination: bool) -> u64 {
        self.traffic
            .saturating_add(u64::from(read_destination) * self.destination_reads)
    }

    pub(crate) fn add_role(
        &mut self,
        layouts: &TreeTransformLayoutTable,
        layout: usize,
        inactive: bool,
        dst: (u8, isize, isize),
        src: (u8, isize, isize),
    ) -> Result<(usize, Counts), OperationError> {
        let (class, point) = role(layouts, layout, inactive)?;
        if point.elements == 0 {
            return Ok((class, Counts::default()));
        }
        let count = self.push(class, point)?;
        if !inactive && layouts.role(layout)?.plan.is_some() {
            self.planned[class] = self.planned[class].add(count)?;
            self.planned_min = if self.planned_min == 0 {
                point.elements
            } else {
                self.planned_min.min(point.elements)
            };
        }
        if dst.0 == 1 {
            self.destination_reads = add(self.destination_reads, count.elements)?;
        }
        self.touch(dst);
        if !inactive {
            self.touch(src);
        }
        Ok((class, count))
    }

    fn touch(&mut self, (buffer, lo, hi): (u8, isize, isize)) {
        let extent = &mut self.extents[buffer as usize];
        *extent = Some(match *extent {
            Some((a, b)) => (a.min(lo), b.max(hi)),
            None => (lo, hi),
        });
    }
}

pub(crate) fn role(
    layouts: &TreeTransformLayoutTable,
    layout: usize,
    inactive: bool,
) -> Result<(usize, Point), OperationError> {
    let elements = layouts.entry(layout).element_count;
    let (inner, strided) = if inactive {
        let (dims, strides) = layouts.inactive_role(layout)?;
        (dims[0], strides[0] != 1)
    } else {
        let role = layouts.role(layout)?;
        (
            role.dims()[0],
            role.dst_strides()[0] != 1 || role.src_strides()[0] != 1,
        )
    };
    Ok((
        2 * usize::from(inactive) + usize::from(strided),
        Point {
            runs: elements.checked_div(inner).unwrap_or(0),
            elements,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Estimates saturate without rejecting metadata; small counts and extents
    /// are the control that still accumulates exactly.
    #[test]
    fn summary_arithmetic_saturates_without_rejecting_metadata() {
        for elements in [3, usize::MAX] {
            let mut phase = Summary::default();
            for _ in 0..2 {
                let count = phase
                    .push(
                        0,
                        Point {
                            runs: elements,
                            elements,
                        },
                    )
                    .unwrap();
                phase
                    .record_job(&[
                        count,
                        Counts::default(),
                        Counts::default(),
                        Counts::default(),
                    ])
                    .unwrap();
            }
            phase.destination_reads = elements as u64;
            phase.touch((1, 0, 2));
            if elements == usize::MAX {
                phase.touch((2, isize::MIN, isize::MAX));
            }
            phase.finish().unwrap();
            assert_eq!(
                phase.counts[0].elements,
                (elements as u64).saturating_mul(2)
            );
            assert_eq!(
                phase.logical_traffic(true),
                (elements as u64).saturating_mul(5)
            );
            assert_eq!(
                phase.working_set,
                if elements == usize::MAX {
                    (usize::MAX as u64).saturating_add(4)
                } else {
                    3
                }
            );
        }
    }

    fn paired_strided_singles(second_source: usize) -> Summary {
        use crate::{TreeTransformBlockSpec, TreeTransformStructure};
        use symtn_core::{BlockKey, BlockSpec, BlockStructure};
        let source = BlockStructure::from_blocks_with_rank(
            1,
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![2], 0).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![2], 4).unwrap(),
            ],
        )
        .unwrap();
        let destination = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        TreeTransformStructure::compile_structures(
            &destination,
            &source,
            &[
                TreeTransformBlockSpec::single(0, 0, 0.75_f64),
                TreeTransformBlockSpec::single(1, second_source, -0.25),
            ],
        )
        .unwrap()
        .parallel_schedule()
        .single_movement
        .clone()
    }

    /// Source envelopes collapse reuse and conservatively include stride gaps;
    /// a distinct source is the control with the larger working set.
    #[test]
    fn working_set_bounds_repeated_strided_sources() {
        for (second_source, working_set) in [(0, 7), (1, 11)] {
            let phase = paired_strided_singles(second_source);
            assert_eq!(phase.working_set, working_set);
            assert_eq!(phase.traffic, 8);
            assert_eq!(
                phase.counts,
                [
                    Counts::default(),
                    Counts {
                        items: 2,
                        runs: 2,
                        elements: 4
                    },
                    Counts::default(),
                    Counts::default()
                ]
            );
        }
    }

    /// Accumulation includes one destination read per output element;
    /// overwrite is the control requiring only source reads and writes.
    #[test]
    fn accumulation_traffic_includes_destination_reads() {
        let phase = paired_strided_singles(0);
        assert_eq!(phase.logical_traffic(true), 12);
        assert_eq!(phase.logical_traffic(false), 8);
    }

    /// Per-group envelopes combine without retaining a summary for every width;
    /// separate groups are the control touching their own packed destinations.
    #[test]
    fn chunk_summaries_use_whole_phase_source_unions() {
        use crate::{TreeTransformBlockSpec, TreeTransformStructure};
        use symtn_core::BlockStructure;
        let source = BlockStructure::packed_column_major(1, [vec![3], vec![3]]).unwrap();
        let destination = BlockStructure::packed_column_major(1, vec![vec![3]; 4]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &destination,
            &source,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        let phases = &transform.parallel_schedule().pack_movement;
        let combined = Summary::totals(phases).unwrap();
        assert_eq!(combined.working_set, 18);
        assert_eq!(combined.traffic, 24);
        for phase in phases {
            assert_eq!(phase.working_set, 12);
            assert_eq!(phase.traffic, 12);
        }
    }
}
