//! A rule identity that outlives its rule retains at most what it charges a
//! cache.

use crate::counting_alloc;

use fusion_symbols::table::TableCategory;
use symtn_sectors::{CategoryRule, EnumeratedCodec, FusionRule, MultiplicityFreeAdmissionMode};

#[test]
fn a_surviving_identity_retains_at_most_its_charge() {
    let _serial = counting_alloc::serial();
    let identity = || {
        let rule = CategoryRule::try_new(
            TableCategory::fibonacci(),
            EnumeratedCodec::<MultiplicityFreeAdmissionMode>::new(),
        )
        .unwrap();
        FusionRule::rule_identity(&rule)
    };
    // Warm fusion-symbols' own table state, which is not the key's.
    drop(identity());
    let (identity, allocs) = counting_alloc::measure(identity);
    assert!(allocs.live_bytes > 0);
    assert!(
        allocs.live_bytes as usize <= identity.charged_retained_bytes(),
        "{allocs:?} > {}",
        identity.charged_retained_bytes()
    );
}
