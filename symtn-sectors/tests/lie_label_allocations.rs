//! Warm label adaptation allocates only storage for returned channel collections.

use crate::counting_alloc;

use symtn_sectors::{CheckedFusionAlgebra, FusionRule, SU2FusionRule, SectorId};

#[test]
fn warm_su2_fusion_and_dual_allocate_only_the_returned_channels() {
    let _serial = crate::counting_alloc::serial();
    for (spin, calls, bytes) in [(7, 0, 0), (8, 1, 16 * size_of::<SectorId>() as u64)] {
        let sector = SectorId::new(spin);
        let _ = SU2FusionRule.fusion_channels(sector, sector);
        let (channels, allocs) =
            counting_alloc::measure(|| SU2FusionRule.fusion_channels(sector, sector));
        assert_eq!(channels.len(), spin + 1);
        // Eight channels fit inline; nine grow the returned SmallVec to 16.
        assert_eq!((allocs.calls, allocs.bytes), (calls, bytes), "{allocs:?}");
        let _ = SU2FusionRule.try_dual_sector(sector).unwrap();
        let (dual, allocs) =
            counting_alloc::measure(|| SU2FusionRule.try_dual_sector(sector).unwrap());
        assert_eq!(dual, sector);
        assert_eq!((allocs.calls, allocs.bytes), (0, 0), "{allocs:?}");
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_lie_group_channel_coordinate_encoding_does_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    use symtn_sectors::LieGroupFusionRule;

    use fusion_symbols::lie::RootSystem;
    let su2_su3_u1 = GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 1).unwrap();
    for (group, dynkin, charges) in [
        (GroupId::su(3).unwrap(), &[1, 1][..], &[][..]),
        (GroupId::su(4).unwrap(), &[1, 0, 1][..], &[][..]),
        (GroupId::u(2).unwrap(), &[1][..], &[-3][..]),
        (su2_su3_u1, &[1, 1, 1][..], &[2][..]),
    ] {
        let rule = LieGroupFusionRule::new(group.clone()).unwrap();
        let category = LieGroup::new(group).unwrap();
        let adjoint = category.irrep_with_charges(dynkin, charges).unwrap();
        let channel_labels: Vec<_> = category
            .fusion(&adjoint, &adjoint)
            .unwrap()
            .map(|channel| (channel.dynkin(), channel.charges().collect::<Vec<_>>()))
            .collect();
        let encode = || {
            channel_labels
                .iter()
                .fold(0usize, |sum, (dynkin, charges)| {
                    sum + rule
                        .encode_irrep_with_charges(dynkin, charges)
                        .unwrap()
                        .id()
                })
        };
        // The first pass interns each sector's irrep (fusion-symbols checks
        // representability once); later passes are arithmetic and lookups.
        let expected = encode();
        let (encoded, allocs) = counting_alloc::measure(encode);
        assert_eq!(encoded, expected);
        assert_eq!((allocs.calls, allocs.bytes), (0, 0), "{allocs:?}");
        // A warm decode allocates only the returned label vectors.
        let sector = rule.encode_irrep_with_charges(dynkin, charges).unwrap();
        let (decoded, allocs) = counting_alloc::measure(|| rule.decode_irrep(sector).unwrap());
        assert_eq!(decoded, (dynkin.to_vec(), charges.to_vec()));
        let returned = [dynkin.len(), charges.len()];
        assert_eq!(
            (allocs.calls, allocs.bytes),
            (
                returned.iter().filter(|&&len| len > 0).count() as u64,
                (returned.iter().sum::<usize>() * size_of::<i64>()) as u64
            ),
            "{allocs:?}"
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn lie_group_reverse_cache_miss_allocates_only_the_interned_entry() {
    let _serial = crate::counting_alloc::serial();
    use fusion_symbols::lie::{GroupId, LieGroup};
    use symtn_sectors::{CategoryProvider, LieGroupFusionRule};

    for (n, labels) in [(3, &[1, 1][..]), (4, &[1, 0, 1][..])] {
        let rule = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap();
        let category = LieGroup::new(GroupId::su(n).unwrap()).unwrap();
        let object = category.irrep(labels).unwrap();
        // A separate rule: encoding interns, and this rule's table must stay cold.
        let expected = LieGroupFusionRule::new(GroupId::su(n).unwrap())
            .unwrap()
            .encode_irrep(labels)
            .unwrap();
        let _ = CategoryProvider::category(&rule);
        let (sector, allocs) = counting_alloc::measure(|| rule.sector_id(&object).unwrap());
        assert_eq!(sector, expected);
        // One Arc entry and the initial backing allocation of each of the two maps.
        // Provider label clones share their weight buffer; dynkin_labels() adds none.
        assert_eq!(allocs.calls, 1 + 2, "SU({n}): {allocs:?}");
        let (sector, allocs) = counting_alloc::measure(|| rule.sector_id(&object).unwrap());
        assert_eq!(sector, expected);
        assert_eq!((allocs.calls, allocs.bytes), (0, 0), "{allocs:?}");
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_lie_group_order_key_does_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    use fusion_symbols::lie::{GroupId, RootSystem};
    use symtn_sectors::{CheckedGenericFusion, LieGroupFusionRule};

    // Up to four factor positions fit the key's inline storage.
    let su2_su3_u1 = GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 1).unwrap();
    for (group, dynkin, charges, len) in [
        (GroupId::su(3).unwrap(), &[1, 1][..], &[][..], 1),
        (GroupId::u(2).unwrap(), &[1][..], &[-3][..], 2),
        (GroupId::spin(4).unwrap(), &[2, 1][..], &[][..], 2),
        (su2_su3_u1, &[1, 1, 1][..], &[2][..], 3),
    ] {
        let rule = LieGroupFusionRule::new(group).unwrap();
        let sector = rule.encode_irrep_with_charges(dynkin, charges).unwrap();
        let (key, allocs) =
            counting_alloc::measure(|| CheckedGenericFusion::sector_order_key(&rule, sector));
        assert_eq!(key.positions().len(), len);
        assert_eq!((allocs.calls, allocs.bytes), (0, 0), "{allocs:?}");
    }
}
