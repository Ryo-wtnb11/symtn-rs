//! Warm label adaptation allocates only storage for returned channel collections.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use symtn_sectors::{CheckedFusionAlgebra, FusionRule, SU2FusionRule, SectorId};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn warm_su2_fusion_and_dual_allocate_only_the_returned_channels() {
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
fn warm_sun_channel_coordinate_encoding_does_not_allocate() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    use symtn_sectors::SUNFusionRule;

    for (n, labels) in [(3, &[1, 1][..]), (4, &[1, 0, 1][..])] {
        let rule = SUNFusionRule::new(n).unwrap();
        let category = LieGroup::new(GroupId::su(n).unwrap()).unwrap();
        let adjoint = category.irrep(labels).unwrap();
        let channel_labels: Vec<_> = category
            .fusion(&adjoint, &adjoint)
            .unwrap()
            .map(|channel| channel.dynkin())
            .collect();
        let encode = || {
            channel_labels.iter().fold(0usize, |sum, labels| {
                sum + rule.encode_dynkin(labels).unwrap().id()
            })
        };
        let expected = encode();
        let (encoded, allocs) = counting_alloc::measure(encode);
        assert_eq!(encoded, expected);
        // The shared slice/iterator codec only: provider fusion and interning are outside.
        assert_eq!((allocs.calls, allocs.bytes), (0, 0), "{allocs:?}");
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_reverse_cache_miss_allocates_only_the_interned_entry() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    use symtn_sectors::{CategoryProvider, SUNFusionRule};

    for (n, labels) in [(3, &[1, 1][..]), (4, &[1, 0, 1][..])] {
        let rule = SUNFusionRule::new(n).unwrap();
        let category = LieGroup::new(GroupId::su(n).unwrap()).unwrap();
        let object = category.irrep(labels).unwrap();
        let expected = rule.encode_dynkin(labels).unwrap();
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
