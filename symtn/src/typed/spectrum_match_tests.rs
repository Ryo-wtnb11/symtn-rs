//! #2081: `add_spectrum_into` pairs blocks with spectrum entries in O(G).
//!
//! The operation counter is the number of blocks that missed the cursor and
//! scanned the spectrum; it is zero when both sides run in sector order, so
//! the walk costs one comparison per block regardless of G.

use std::sync::Arc;

use super::fusion_tree::add_spectrum_counting_misses;
use super::{GradedSpace, SectorSpectrum, TensorMap};
use crate::runtime::Runtime;
use symtn_core::{U1FusionRule, U1Irrep};

fn misses_and_data(groups: usize) -> (usize, Vec<f64>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let sectors: Vec<_> = (0..groups as i32)
        .map(|k| U1Irrep::new(31 * k - 15_000))
        .collect();
    let bond =
        GradedSpace::try_new(Arc::new(U1FusionRule), sectors.iter().map(|&s| (s, 1))).unwrap();
    let compact = TensorMap::<U1FusionRule, f64>::diagonal(
        engine,
        &runtime,
        &bond,
        sectors
            .iter()
            .enumerate()
            .map(|(i, &sector)| SectorSpectrum {
                sector,
                values: vec![1.0 + i as f64],
            }),
    )
    .unwrap();
    let mut data = vec![0.0; groups];
    let misses = add_spectrum_counting_misses(
        compact.logical_space().space(),
        &mut data,
        compact.spectrum().unwrap(),
        2.0,
    )
    .unwrap();
    (misses, data)
}

#[test]
fn spectrum_walk_scans_nothing_and_scatters_every_sector() {
    for groups in [1000, 2000] {
        let (misses, data) = misses_and_data(groups);
        assert_eq!(misses, 0, "G = {groups}");
        // Block offsets follow the structure's own layout, not sector order,
        // so compare as a multiset; which entry each block gets is pinned by
        // the cursor test below.
        let mut data = data;
        data.sort_by(f64::total_cmp);
        let expected: Vec<f64> = (0..groups).map(|i| 2.0 * (1.0 + i as f64)).collect();
        assert_eq!(data, expected, "G = {groups}");
    }
}

#[test]
fn cursor_pairs_by_sector_on_shuffled_gapped_spectrum() {
    use super::fusion_tree::SpectrumCursor;
    use symtn_matrixalgebra::SectorSpectrum as Entry;
    let id = |k: i32| crate::sector::SectorId::from(U1Irrep::new(k));
    // Entries for 3, 1, 5 (shuffled); sector 9 has no entry, 7 has no block.
    let spectrum: Vec<Entry<f64>> = [3, 1, 5, 7]
        .iter()
        .map(|&k| Entry {
            sector: id(k),
            values: vec![f64::from(k)],
        })
        .collect();
    let mut cursor = SpectrumCursor::default();
    let mut got = Vec::new();
    for k in [1, 3, 5, 9] {
        got.push(cursor.find(&spectrum, id(k)).map(|e| e.values[0]));
    }
    // 1: miss (next is 3) -> scan, cursor 2; 3: miss (next is 5) -> scan;
    // 5: cursor now 1 -> entry 1 != 5, miss; 9: miss, absent.
    assert_eq!(got, [Some(1.0), Some(3.0), Some(5.0), None]);
    assert_eq!(cursor.misses, 4);
    // In order, every lookup hits.
    let mut cursor = SpectrumCursor::default();
    let ordered: Vec<Entry<f64>> = [1, 3, 5]
        .iter()
        .map(|&k| Entry {
            sector: id(k),
            values: vec![f64::from(k)],
        })
        .collect();
    for k in [1, 3, 5] {
        assert_eq!(
            cursor.find(&ordered, id(k)).unwrap().values[0],
            f64::from(k)
        );
    }
    assert_eq!(cursor.misses, 0);
}
