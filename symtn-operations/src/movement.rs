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
    pub items: usize,
    pub runs: usize,
    pub elements: usize,
}

impl Counts {
    fn add(self, rhs: Self) -> Result<Self, OperationError> {
        Ok(Self {
            items: add(self.items, rhs.items)?,
            runs: add(self.runs, rhs.runs)?,
            elements: add(self.elements, rhs.elements)?,
        })
    }
}

fn add(a: usize, b: usize) -> Result<usize, OperationError> {
    a.checked_add(b).ok_or(OperationError::ElementCountOverflow)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Summary {
    pub counts: [Counts; CLASSES],
    pub hulls: [Vec<Point>; CLASSES],
    /// Payload elements touched, source and destination included.
    pub working_set: usize,
    pub traffic: usize,
    pub destination_reads: usize,
    // Compile-only address union; discarded after every range has been baked.
    touches: Vec<(u8, isize, isize)>,
}

impl Summary {
    pub(crate) fn push(&mut self, class: usize, point: Point) -> Result<(), OperationError> {
        self.counts[class] = self.counts[class].add(Counts {
            items: 1,
            runs: point.runs,
            elements: point.elements,
        })?;
        self.hulls[class].push(point);
        let touched = point
            .elements
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
        self.touches.sort_unstable();
        let mut merged: Vec<(u8, isize, isize)> = Vec::new();
        for &(buffer, lo, hi) in &self.touches {
            match merged.last_mut() {
                Some(last) if last.0 == buffer && lo <= last.2.saturating_add(1) => {
                    last.2 = last.2.max(hi)
                }
                _ => merged.push((buffer, lo, hi)),
            }
        }
        if !merged.is_empty() {
            self.working_set = merged.iter().try_fold(0, |sum, &(_, lo, hi)| {
                add(sum, (hi as i128 - lo as i128 + 1) as usize)
            })?;
        }
        self.touches = merged;
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
            for c in 0..CLASSES {
                result.counts[c] = result.counts[c].add(summary.counts[c])?;
                result.hulls[c].extend_from_slice(&summary.hulls[c]);
            }
            result.working_set = add(result.working_set, summary.working_set)?;
            result.traffic = add(result.traffic, summary.traffic)?;
            result.destination_reads = add(result.destination_reads, summary.destination_reads)?;
            result.touches.extend_from_slice(&summary.touches);
        }
        result.finish()?;
        Ok(result)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.hulls
            .iter()
            .map(|h| h.capacity() * core::mem::size_of::<Point>())
            .sum::<usize>()
            + self.touches.capacity() * core::mem::size_of::<(u8, isize, isize)>()
    }

    pub(crate) fn freeze(&mut self) {
        self.touches = Vec::new();
    }

    pub(crate) fn logical_traffic(&self, read_destination: bool) -> usize {
        self.traffic + usize::from(read_destination) * self.destination_reads
    }

    pub(crate) fn add_role(
        &mut self,
        layouts: &TreeTransformLayoutTable,
        layout: usize,
        inactive: bool,
        dst: (u8, isize),
        src: (u8, isize),
    ) -> Result<(), OperationError> {
        let (class, point) = role(layouts, layout, inactive)?;
        if point.elements == 0 {
            return Ok(());
        }
        self.push(class, point)?;
        if dst.0 == 1 {
            self.destination_reads = add(self.destination_reads, point.elements)?;
        }
        if inactive {
            let (dims, strides) = layouts.inactive_role(layout)?;
            self.touch(dst, dims, strides)?;
        } else {
            let role = layouts.role(layout)?;
            self.touch(dst, role.dims(), role.dst_strides())?;
            self.touch(src, role.dims(), role.src_strides())?;
        }
        Ok(())
    }

    fn touch(
        &mut self,
        (buffer, offset): (u8, isize),
        dims: &[usize],
        strides: &[isize],
    ) -> Result<(), OperationError> {
        let inner = if strides[0].unsigned_abs() == 1 {
            dims[0]
        } else {
            1
        };
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or(OperationError::ElementCountOverflow)?;
        for start in (0..elements).step_by(inner) {
            let (mut linear, mut at) = (start, offset as i128);
            for (&dim, &stride) in dims.iter().zip(strides) {
                at += (linear % dim) as i128 * stride as i128;
                linear /= dim;
            }
            let end = at + (inner - 1) as i128 * strides[0] as i128;
            let lo =
                isize::try_from(at.min(end)).map_err(|_| OperationError::ElementCountOverflow)?;
            let hi =
                isize::try_from(at.max(end)).map_err(|_| OperationError::ElementCountOverflow)?;
            self.touches.push((buffer, lo, hi));
        }
        Ok(())
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

/// The exact phases executed by every possible plan-group chunk width.
/// A width above the group count executes the one whole chunk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ranges {
    widths: Vec<Vec<Summary>>,
}

impl Ranges {
    pub(crate) fn new(leaves: Vec<Summary>) -> Result<Self, OperationError> {
        let mut widths = Vec::with_capacity(leaves.len());
        for width in 1..=leaves.len() {
            let mut chunks = Vec::with_capacity(leaves.len().div_ceil(width));
            for chunk in leaves.chunks(width) {
                let mut sum = Summary::merge_all(chunk)?;
                sum.freeze();
                chunks.push(sum);
            }
            widths.push(chunks);
        }
        Ok(Self { widths })
    }

    pub(crate) fn get(&self, width: usize, chunk: usize) -> &Summary {
        &self.widths[width.min(self.widths.len()) - 1][chunk]
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.widths.capacity() * core::mem::size_of::<Vec<Summary>>()
            + self
                .widths
                .iter()
                .map(|chunks| {
                    chunks.capacity() * core::mem::size_of::<Summary>()
                        + chunks.iter().map(Summary::retained_bytes).sum::<usize>()
                })
                .sum::<usize>()
    }
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

    /// Reusing a strided source changes W but not repeated traffic X; a
    /// distinct source block is the control with the larger working set.
    #[test]
    fn working_set_unions_repeated_strided_sources() {
        for (second_source, working_set) in [(0, 6), (1, 8)] {
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

    /// Aligned chunk summaries union repeated pack sources across groups;
    /// separate chunks are the control touching the source once each.
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
        assert_eq!(phases.get(2, 0).working_set, 18);
        assert_eq!(phases.get(2, 0).traffic, 24);
        for chunk in 0..2 {
            assert_eq!(phases.get(1, chunk).working_set, 12);
            assert_eq!(phases.get(1, chunk).traffic, 12);
        }
    }
}
