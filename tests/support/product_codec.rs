//! A bounded custom codec exercises public codec identity and rejection paths.

use symtn_sectors::{ProductSectorCodec, SectorId, TensorKitProductCodec};

pub struct ParityProductCodec;

impl ProductSectorCodec for ParityProductCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId> {
        if right.id() > 1 {
            return None;
        }
        TensorKitProductCodec::try_encode(right, left)
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)> {
        let (right, left) = TensorKitProductCodec::decode(sector)?;
        (right.id() <= 1).then_some((left, right))
    }
}
