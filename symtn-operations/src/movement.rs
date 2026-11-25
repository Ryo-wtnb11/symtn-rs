//! Structural movement counts. Coefficients and payload values never enter them.

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
    pub items: u128,
    pub runs: u128,
    pub elements: u128,
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

fn add(a: u128, b: u128) -> Result<u128, OperationError> {
    a.checked_add(b).ok_or(OperationError::ElementCountOverflow)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Summary {
    pub counts: [Counts; CLASSES],
    pub hulls: [Vec<Point>; CLASSES],
    /// Payload elements touched, source and destination included.
    pub working_set: u128,
    pub traffic: u128,
    pub destination_reads: u128,
    /// Executed jobs retain their joint class counts, including every role setup.
    pub jobs: Vec<[Counts; CLASSES]>,
    // Per-buffer address envelopes: exact when coalesced, otherwise upper bounds.
    extents: [Option<(i128, i128)>; 4],
}

impl Summary {
    pub(crate) fn push(&mut self, class: usize, point: Point) -> Result<(), OperationError> {
        self.counts[class] = self.counts[class].add(Counts {
            items: 1,
            runs: point.runs as u128,
            elements: point.elements as u128,
        })?;
        self.hulls[class].push(point);
        let mut job = [Counts::default(); CLASSES];
        job[class] = Counts {
            items: 1,
            runs: point.runs as u128,
            elements: point.elements as u128,
        };
        self.jobs.push(job);
        let touched = (point.elements as u128)
            .checked_mul([2, 2, 1, 1][class])
            .ok_or(OperationError::ElementCountOverflow)?;
        self.working_set = add(self.working_set, touched)?;
        self.traffic = add(self.traffic, touched)?;
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), OperationError> {
        // Both whole-call destination modes must have representable traffic.
        add(self.traffic, self.destination_reads)?;
        for points in &mut self.hulls {
            *points = upper_hull(core::mem::take(points));
        }
        self.working_set = self
            .extents
            .iter()
            .flatten()
            .try_fold(0, |sum, &(lo, hi)| add(sum, (hi - lo + 1) as u128))?;
        Ok(())
    }

    pub(crate) fn merge(&self, rhs: &Self) -> Result<Self, OperationError> {
        Self::merge_all([self, rhs])
    }

    pub(crate) fn merge_all<'a>(
        summaries: impl IntoIterator<Item = &'a Self>,
    ) -> Result<Self, OperationError> {
        let mut result = Self::default();
        for summary in summaries {
            result.add_totals(summary)?;
            for c in 0..CLASSES {
                result.hulls[c].extend_from_slice(&summary.hulls[c]);
            }
            result.jobs.extend_from_slice(&summary.jobs);
        }
        result.finish()?;
        Ok(result)
    }

    /// Stack-only aggregation for a chunk; no hull rebuilding or job copying.
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

    fn add_totals(&mut self, rhs: &Self) -> Result<(), OperationError> {
        for c in 0..CLASSES {
            self.counts[c] = self.counts[c].add(rhs.counts[c])?;
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

    /// Each ownership run is indivisible; its roles all execute on one leaf.
    pub(crate) fn fuse_jobs(&mut self, starts: &[u32]) -> Result<(), OperationError> {
        let mut jobs = Vec::with_capacity(starts.len());
        for (i, &start) in starts.iter().enumerate() {
            let end = starts
                .get(i + 1)
                .map_or(self.jobs.len(), |&end| end as usize);
            let mut total = [Counts::default(); CLASSES];
            for job in &self.jobs[start as usize..end] {
                for c in 0..CLASSES {
                    total[c] = total[c].add(job[c])?;
                }
            }
            jobs.push(total);
        }
        self.jobs = jobs;
        Ok(())
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.hulls
            .iter()
            .map(|h| h.capacity() * core::mem::size_of::<Point>())
            .sum::<usize>()
            + self.jobs.capacity() * core::mem::size_of::<[Counts; CLASSES]>()
    }

    pub(crate) fn logical_traffic(&self, read_destination: bool) -> u128 {
        self.traffic + u128::from(read_destination) * self.destination_reads
    }

    pub(crate) fn add_role(
        &mut self,
        layouts: &TreeTransformLayoutTable,
        layout: usize,
        inactive: bool,
        dst: (u8, i128, i128),
        src: (u8, i128, i128),
    ) -> Result<(), OperationError> {
        let (class, point) = role(layouts, layout, inactive)?;
        if point.elements == 0 {
            return Ok(());
        }
        self.push(class, point)?;
        if dst.0 == 1 {
            self.destination_reads = add(self.destination_reads, point.elements as u128)?;
        }
        self.touch(dst);
        if !inactive {
            self.touch(src);
        }
        Ok(())
    }

    fn touch(&mut self, (buffer, lo, hi): (u8, i128, i128)) {
        let extent = &mut self.extents[buffer as usize];
        *extent = Some(match *extent {
            Some((a, b)) => (a.min(lo), b.max(hi)),
            None => (lo, hi),
        });
    }
}

/// Increasing runs, decreasing elements, with strictly decreasing edge slopes.
/// Points below an upper chord cannot maximize a nonnegative linear form.
fn upper_hull(mut points: Vec<Point>) -> Vec<Point> {
    // Keep the largest element value for each run count.
    points.sort_unstable_by_key(|p| (p.runs, core::cmp::Reverse(p.elements)));
    points.dedup_by_key(|p| p.runs);
    let mut frontier = Vec::new();
    let mut largest = None;
    for &point in points.iter().rev() {
        if largest.is_none_or(|elements| point.elements > elements) {
            frontier.push(point);
            largest = Some(point.elements);
        }
    }
    frontier.reverse();
    let mut hull: Vec<Point> = Vec::new();
    for point in frontier {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let left = (b.runs - a.runs) as u128 * (b.elements - point.elements) as u128;
            let right = (point.runs - b.runs) as u128 * (a.elements - b.elements) as u128;
            if left > right {
                break;
            }
            hull.pop();
        }
        hull.push(point);
    }
    hull.shrink_to_fit();
    hull
}

pub(crate) fn maximum(hull: &[Point], run: f64, element: f64) -> f64 {
    let cost = |p: Point| run * p.runs as f64 + element * p.elements as f64;
    if hull.is_empty() {
        return 0.0;
    }
    let (mut lo, mut hi) = (0, hull.len() - 1);
    while lo < hi {
        let middle = lo + (hi - lo) / 2;
        if cost(hull[middle]) < cost(hull[middle + 1]) {
            lo = middle + 1;
        } else {
            hi = middle;
        }
    }
    cost(hull[lo])
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

    /// The maximizing item changes with the coefficients; coordinate maxima
    /// from different items must never be combined into an imaginary item.
    #[test]
    fn hull_keeps_both_coefficient_dependent_maximizers() {
        let points = vec![
            Point {
                runs: 100,
                elements: 100,
            },
            Point {
                runs: 1,
                elements: 1000,
            },
        ];
        let hull = upper_hull(points);
        assert_eq!(maximum(&hull, 10.0, 0.1), 1010.0);
        assert_eq!(maximum(&hull, 0.1, 10.0), 10000.1);
    }

    /// Exhaustive coefficient pairs on deterministic small random sets agree
    /// with an independent linear scan, including dominated and duplicate points.
    #[test]
    fn hull_binary_search_matches_linear_scan() {
        let mut seed = 17u64;
        for count in 0..30 {
            let points = (0..count)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    Point {
                        runs: (seed >> 32) as usize % 30,
                        elements: seed as usize % 30,
                    }
                })
                .collect::<Vec<_>>();
            let hull = upper_hull(points.clone());
            for r in 0..10 {
                for e in 0..10 {
                    let expected = points
                        .iter()
                        .map(|p| (r * p.runs + e * p.elements) as f64)
                        .fold(0.0, f64::max);
                    assert_eq!(maximum(&hull, r as f64, e as f64), expected);
                }
            }
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
        for chunk in 0..2 {
            assert_eq!(phases[chunk].working_set, 12);
            assert_eq!(phases[chunk].traffic, 12);
        }
    }
}
