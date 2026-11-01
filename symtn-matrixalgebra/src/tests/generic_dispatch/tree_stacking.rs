use super::*;

/// Hermitian and general payloads of one coupled-sector tiling, written per
/// region so the Hermitian one is Hermitian in the tree basis.
fn region_payloads(
    regions: &[symtn_core::CoupledSectorRegion],
    len: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let entry = |salt: usize| {
        let value = |seed: usize| ((seed * 2_654_435_761) % 1009) as f64 / 1009.0 - 0.5;
        Complex64::new(value(salt), value(salt + 7919))
    };
    let mut general = vec![Complex64::zero(); len];
    let mut hermitian = general.clone();
    for region in regions {
        let (rows, start) = (region.rows(), region.range().start);
        for column in 0..region.cols() {
            for row in 0..rows {
                general[start + row + rows * column] = entry(start + row + rows * column);
            }
        }
        if rows == region.cols() {
            for column in 0..rows {
                for row in 0..rows {
                    hermitian[start + row + rows * column] = general[start + row + rows * column]
                        + general[start + column + rows * row].conj();
                }
            }
        }
    }
    (hermitian, general)
}

// The leg-degeneracy (facade) builder lists multi-leg Generic trees in
// its own order, which a `FusionTreeKey`-Ord admission would refuse, packing
// the whole payload before QR/SVD/LQ and the value-only spectra. A coupled-sector tiling is the same matricization the
// pack would build, so no pack may happen.
#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_facade_layouts_take_the_direct_region_path() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let x = SectorId::new(1);
    let v = SectorLeg::new([(SectorId::new(0), 1), (x, 2)], false);
    let w = SectorLeg::new([(x, 1)], false);
    let product =
        |legs: &[&SectorLeg]| FusionProductSpace::new(legs.iter().map(|&leg| leg.clone()));
    let cases = [
        (product(&[&v, &v]), product(&[&w])),
        (product(&[&w]), product(&[&v, &v])),
        (product(&[&v, &w]), product(&[&v, &w])),
        (product(&[&v, &v, &w]), product(&[&v])),
    ];
    let mut unordered_fixtures = 0;
    for (codomain, domain) in cases {
        let endomorphism = codomain == domain;
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: FactorGenericRule.rule_identity(),
        });
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            FusionTreeHomSpace::new(codomain, domain),
        )
        .unwrap();
        let regions = space
            .space()
            .structure()
            .coupled_sector_regions(space.space().nout())
            .unwrap()
            .expect("the facade builder lays out a coupled-sector tiling");
        let ord_ordered = regions.iter().all(|region| {
            [region.row_trees(), region.col_trees()]
                .iter()
                .all(|trees| trees.windows(2).all(|pair| pair[0].tree() < pair[1].tree()))
        });
        unordered_fixtures += usize::from(!ord_ordered);
        let (hermitian, general) = region_payloads(&regions, space.space().required_len().unwrap());
        let input = BoundDynamicTensorRef::try_new(&space, &general).unwrap();
        let mut dense = symtn_dense::DefaultDenseExecutor::new();

        let generic_pair_rec = crate::probe::record();
        let compact_qr_copy_rec = crate::probe::record();
        let qr = qr_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &qr.q, None, &qr.r);
        assert_eq!(compact_qr_copy_rec.compact_qr_copy().input_pack_bytes, 0);

        let compact_svd_copy_rec = crate::probe::record();
        let svd = svd_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &svd.u, Some(&svd.s), &svd.vh);
        assert_eq!(compact_svd_copy_rec.compact_svd_copy().input_pack_bytes, 0);

        let compact_lq_copy_rec = crate::probe::record();
        let lq = lq_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &lq.l, None, &lq.q);
        assert_eq!(compact_lq_copy_rec.compact_lq_copy().input_pack_bytes, 0);

        // The fresh bond-space builder enumerates the preserved side's trees
        // in the facade input's order, so the per-call route proof publishes
        // every factor in place: no pack in, no scatter out.
        let publication = generic_pair_rec.generic_pair_publication();
        assert_eq!(
            (
                publication.canonical_publications,
                publication.fallback_publications
            ),
            (3, 0),
            "{publication:?}"
        );
        let values_fallbacks_rec = crate::probe::record();
        svd_vals_dyn_checked_generic(&mut dense, &input).unwrap();
        if endomorphism {
            let hermitian_input = BoundDynamicTensorRef::try_new(&space, &hermitian).unwrap();
            eigh_vals_dyn_checked_generic(&mut dense, &hermitian_input).unwrap();
            eig_vals_dyn_checked_generic(&mut dense, &input).unwrap();
        }
        assert_eq!(
            values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
            0
        );
    }
    assert!(
        unordered_fixtures > 0,
        "the fixtures must include the facade order that Ord admission packed"
    );
}
