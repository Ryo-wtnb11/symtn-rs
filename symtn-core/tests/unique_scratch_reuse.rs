//! `UniqueTreePairScratch` reuse contract: one scratch serves every
//! tree pair, whatever rule, rank, dual pattern or operation came before.

use symtn_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    PreparedTreePairOperation, SectorId, SectorLeg, U1FusionRule, U1Irrep, UniqueTreePairScratch,
    Z2FusionRule,
};

/// Every case of one HomSpace with codomain duals `codomain` and domain duals
/// `domain`, through one permute, braid and transpose, against the key form.
fn compare<R>(
    rule: &R,
    sectors: &[SectorId],
    codomain: &[bool],
    domain: &[bool],
    scratch: &mut UniqueTreePairScratch,
) -> usize
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let leg = |is_dual: bool| SectorLeg::new(sectors.iter().map(|&s| (s, 1)), is_dual);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain.iter().map(|&d| leg(d)).collect::<Vec<_>>()),
        FusionProductSpace::new(domain.iter().map(|&d| leg(d)).collect::<Vec<_>>()),
    );
    let (nout, nin) = (codomain.len(), domain.len());
    let rank = nout + nin;
    // Reverse the legs, move one leg across, and rotate by one.
    let reversed: Vec<usize> = (0..rank).rev().collect();
    let (permute_codomain, permute_domain) = reversed.split_at(rank.div_ceil(2));
    let levels: Vec<usize> = (0..rank).collect();
    // A planar transpose rotates the cyclic order (codomain, then the domain
    // reversed) by one leg and keeps the codomain width.
    let mut cycle: Vec<usize> = (0..nout).chain((nout..rank).rev()).collect();
    cycle.rotate_left(1);
    let rotate_codomain = &cycle[..nout];
    let rotate_domain: Vec<usize> = cycle[nout..].iter().rev().copied().collect();
    let operations = [
        PreparedTreePairOperation::prepare_permute(
            rule,
            nout,
            nin,
            permute_codomain,
            permute_domain,
        )
        .unwrap(),
        PreparedTreePairOperation::prepare_braid(
            rule,
            nout,
            nin,
            permute_codomain,
            permute_domain,
            &levels[..nout],
            &levels[nout..],
        )
        .unwrap(),
        PreparedTreePairOperation::prepare_transpose(nout, nin, rotate_codomain, &rotate_domain)
            .unwrap(),
    ];
    let keys = homspace.fusion_tree_keys(rule);
    let mut compared = 0;
    for operation in &operations {
        for key in keys.iter().take(6) {
            let (fresh_key, fresh) = operation.execute_unique_rigid(rule, key).unwrap();
            let reused = operation
                .execute_unique_rigid_into(rule, key, scratch)
                .unwrap();
            let reused_key = scratch.pair_view().to_key();
            assert_eq!(reused_key, fresh_key);
            assert_eq!(format!("{reused_key:?}"), format!("{fresh_key:?}"));
            assert_eq!(reused.to_bits(), fresh.to_bits());
            compared += 1;
        }
    }
    compared
}

/// What: a scratch reused across rules, rising then falling ranks, dual
/// patterns and operations leaves no state behind: every result equals the
/// fresh key form, key, `Debug` and coefficient bits.
#[test]
fn reused_unique_scratch_matches_the_key_form() {
    let mut scratch = UniqueTreePairScratch::with_rank(2);
    let u1: Vec<SectorId> = [-1, 0, 1].map(|q| U1Irrep::new(q).sector_id()).to_vec();
    let z2 = [SectorId::new(0), SectorId::new(1)];
    let mut compared = 0;
    compared += compare(&U1FusionRule, &u1, &[false, true], &[true], &mut scratch);
    compared += compare(
        &FermionParityFusionRule,
        &z2,
        &[true, false, true],
        &[false, false, true],
        &mut scratch,
    );
    compared += compare(
        &Z2FusionRule,
        &z2,
        &[false, true, true, false],
        &[true, false, false, true],
        &mut scratch,
    );
    compared += compare(
        &U1FusionRule,
        &u1,
        &[true, false],
        &[false, true],
        &mut scratch,
    );
    compared += compare(
        &FermionParityFusionRule,
        &z2,
        &[false],
        &[true, true],
        &mut scratch,
    );
    assert!(compared >= 60, "{compared} cases");
}
