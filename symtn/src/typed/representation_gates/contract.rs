use super::*;

#[test]
fn coupled_region_inner_rejects_malformed_scalar_range() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = su2_lazy_fixture(engine);
    let structure = owned(&tensor).space.space().structure();
    let data = tensor.dense_data().unwrap();
    let error = coupled_region_inner(
        structure,
        owned(&tensor).space.space().nout(),
        &data[..data.len() - 1],
        data,
        |_| Ok::<_, Error>(1.0),
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(message) if
        message.contains("internal coupled-layout invariant violated")));
}

#[test]
fn coupled_region_inner_keeps_empty_and_non_fusion_boundaries() {
    let empty = BlockStructure::empty(3);
    assert_eq!(
        coupled_region_inner::<f64, _, Error>(&empty, 1, &[], &[], |_| Ok(7.0)).unwrap(),
        Complex64::new(0.0, 0.0)
    );

    let trivial = BlockStructure::trivial(&[2, 2]).unwrap();
    let error = coupled_region_inner(&trivial, 1, &[1.0; 4], &[1.0; 4], |_| Ok::<_, Error>(1.0))
        .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(message) if
        message.contains("non-packed coupled-sector layout")));
}

#[test]
fn weighted_trace_keeps_misaligned_and_nonpacked_layouts_on_the_literal_walk() {
    let tree = |dual| {
        FusionTreeKey::try_from_sector_ids_for_rule(&Z2FusionRule, [0], 0, [dual], [], []).unwrap()
    };
    let (a, b) = (tree(false), tree(true));
    let block = |row: &FusionTreeKey, col: &FusionTreeKey, offset| {
        BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(row.clone(), col.clone())),
            vec![1, 1],
            vec![1, 2],
            offset,
        )
        .unwrap()
    };
    // The packed matrix's logical row order is [a, b] while its columns
    // are [b, a]; its two literal diagonal blocks occur b then a.
    let misaligned = BlockStructure::from_blocks(vec![
        block(&a, &b, 0),
        block(&b, &b, 1),
        block(&a, &a, 2),
        block(&b, &a, 3),
    ])
    .unwrap();
    assert!(!misaligned.coupled_sector_regions(1).unwrap().unwrap()[0].has_aligned_diagonal());
    let mut weights = Vec::new();
    let value = weighted_trace(&misaligned, 1, &[10.0, 20.0, 30.0, 40.0], |sector| {
        weights.push(sector);
        Ok::<_, Error>(2.0)
    })
    .unwrap();
    assert_eq!(value, Complex64::new(100.0, 0.0));
    assert_eq!(weights, vec![SectorId::new(0), SectorId::new(0)]);

    let nonpacked = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(a.clone(), a)),
        vec![2, 2],
        vec![2, 4],
        0,
    )
    .unwrap()])
    .unwrap();
    assert_eq!(nonpacked.coupled_sector_regions(1).unwrap(), None);
    let value = weighted_trace(&nonpacked, 1, &[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], |_| {
        Ok::<_, Error>(2.0)
    })
    .unwrap();
    assert_eq!(value, Complex64::new(16.0, 0.0));
}
