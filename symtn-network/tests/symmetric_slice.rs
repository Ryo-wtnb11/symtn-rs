use symtn::expert::SectorLeg;
use symtn::sector::FusionRule;
use symtn::sector::{
    FermionParityFusionRule, FermionParityIrrep, SectorId, U1FusionRule, U1Irrep, ZNFusionRule,
};
use symtn_network::{
    DegeneracyRange, NetworkIR, SectorSlice, SliceError, SliceKind, SymmetricSlicePlan,
    SymmetricSliceSpec, TemporaryLabel, TensorAxis, TensorId,
};

fn label(name: &str) -> TemporaryLabel {
    TemporaryLabel::new(name)
}

fn range(start: usize, end: usize) -> DegeneracyRange {
    DegeneracyRange::new(start, end).unwrap()
}

fn piece(sector: usize, start: usize, end: usize) -> SectorSlice {
    SectorSlice::new(SectorId::new(sector), range(start, end))
}

fn axis(tensor: usize) -> TensorAxis {
    axis_at(tensor, 0)
}

fn axis_at(tensor: usize, axis: usize) -> TensorAxis {
    TensorAxis::new(TensorId::new(tensor), axis)
}

fn internal_ir() -> NetworkIR {
    NetworkIR::from_labels(vec![vec![label("x")], vec![label("x")]], vec![]).unwrap()
}

fn spec(
    label_name: &str,
    authority: TensorAxis,
    leg: SectorLeg,
    pieces: Vec<SectorSlice>,
) -> SymmetricSliceSpec {
    SymmetricSliceSpec::new(label(label_name), authority, leg, pieces)
}

fn checked(
    ir: &NetworkIR,
    specs: Vec<SymmetricSliceSpec>,
) -> Result<SymmetricSlicePlan, SliceError> {
    SymmetricSlicePlan::try_new(ir, U1FusionRule.rule_identity(), specs)
}

#[test]
fn canonicalizes_input_and_piece_order_without_merging_adjacent_ranges() {
    let ir = NetworkIR::from_labels(
        vec![vec![label("x")], vec![label("x")], vec![label("y")]],
        vec![label("y")],
    )
    .unwrap();
    let x_leg = SectorLeg::new([(SectorId::new(1), 2), (SectorId::new(3), 1)], false);
    let y_leg = SectorLeg::new([(SectorId::new(2), 2)], false);

    let x_forward = vec![piece(1, 0, 1), piece(1, 1, 2), piece(3, 0, 1)];
    let x_shuffled = vec![piece(3, 0, 1), piece(1, 1, 2), piece(1, 0, 1)];
    let first = checked(
        &ir,
        vec![
            spec("y", axis(2), y_leg.clone(), vec![piece(2, 0, 2)]),
            spec("x", axis(0), x_leg.clone(), x_shuffled),
        ],
    )
    .unwrap();
    let second = checked(
        &ir,
        vec![
            spec("x", axis(0), x_leg, x_forward.clone()),
            spec("y", axis(2), y_leg, vec![piece(2, 0, 2)]),
        ],
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.indices()[0].label(), &label("x"));
    assert_eq!(first.indices()[0].kind(), SliceKind::Internal);
    assert_eq!(first.indices()[1].kind(), SliceKind::Output);
    assert_eq!(first.indices()[0].output_position(), None);
    assert_eq!(first.indices()[1].output_position(), Some(0));
    assert_eq!(first.indices()[0].pieces(), x_forward);
    assert_eq!(first.indices()[0].pieces().len(), 3);

    let merged = checked(
        &ir,
        vec![
            spec(
                "x",
                axis(0),
                SectorLeg::new([(SectorId::new(1), 2), (SectorId::new(3), 1)], false),
                vec![piece(1, 0, 2), piece(3, 0, 1)],
            ),
            spec(
                "y",
                axis(2),
                SectorLeg::new([(SectorId::new(2), 2)], false),
                vec![piece(2, 0, 2)],
            ),
        ],
    )
    .unwrap();
    assert_ne!(first, merged, "adjacent partition boundaries are identity");
}

#[test]
fn counts_and_enumerates_multi_sector_multi_index_cartesian_product() {
    let ir = NetworkIR::from_labels(
        vec![
            vec![label("x")],
            vec![label("x")],
            vec![label("y")],
            vec![label("y")],
        ],
        vec![],
    )
    .unwrap();
    let plan = checked(
        &ir,
        vec![
            spec(
                "y",
                axis(2),
                SectorLeg::new([(SectorId::new(4), 2)], false),
                vec![piece(4, 0, 1), piece(4, 1, 2)],
            ),
            spec(
                "x",
                axis(0),
                SectorLeg::new([(SectorId::new(1), 1), (SectorId::new(3), 1)], false),
                vec![piece(3, 0, 1), piece(1, 0, 1)],
            ),
        ],
    )
    .unwrap();

    assert_eq!(plan.nslices(), 4);
    let combinations = plan
        .combinations()
        .map(|items| {
            items
                .into_iter()
                .map(|item| (item.sector().id(), item.range().start()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        combinations,
        vec![
            vec![(1, 0), (4, 0)],
            vec![(1, 0), (4, 1)],
            vec![(3, 0), (4, 0)],
            vec![(3, 0), (4, 1)],
        ]
    );
}

#[test]
fn preserves_output_axis_order_separately_from_canonical_label_order() {
    let ir = NetworkIR::from_labels(
        vec![vec![label("a")], vec![label("z")]],
        vec![label("z"), label("a")],
    )
    .unwrap();
    let plan = checked(
        &ir,
        vec![
            spec(
                "z",
                axis(1),
                SectorLeg::new([(SectorId::new(0), 1)], false),
                vec![piece(0, 0, 1)],
            ),
            spec(
                "a",
                axis(0),
                SectorLeg::new([(SectorId::new(0), 1)], false),
                vec![piece(0, 0, 1)],
            ),
        ],
    )
    .unwrap();

    assert_eq!(plan.indices()[0].label(), &label("a"));
    assert_eq!(plan.indices()[0].output_position(), Some(1));
    assert_eq!(plan.indices()[1].label(), &label("z"));
    assert_eq!(plan.indices()[1].output_position(), Some(0));
}

#[test]
fn rejects_every_range_partition_and_authority_error() {
    assert_eq!(
        DegeneracyRange::new(2, 2),
        Err(SliceError::EmptyRange { at: 2 })
    );
    assert_eq!(
        DegeneracyRange::new(3, 2),
        Err(SliceError::ReversedRange { start: 3, end: 2 })
    );

    let ir = internal_ir();
    let leg = SectorLeg::new([(SectorId::new(1), 3)], false);
    let make = |pieces| spec("x", axis(0), leg.clone(), pieces);
    assert!(matches!(
        checked(
            &ir,
            vec![spec("missing", axis(0), leg.clone(), vec![piece(1, 0, 3)])]
        ),
        Err(SliceError::UnknownLabel(_))
    ));
    assert!(matches!(
        checked(
            &ir,
            vec![make(vec![piece(1, 0, 3)]), make(vec![piece(1, 0, 3)])]
        ),
        Err(SliceError::DuplicateLabel(_))
    ));
    assert!(matches!(
        checked(
            &ir,
            vec![spec("x", axis(1), leg.clone(), vec![piece(1, 0, 3)])]
        ),
        Err(SliceError::InvalidAuthority { .. })
    ));
    assert!(matches!(
        checked(&ir, vec![make(vec![piece(9, 0, 1)])]),
        Err(SliceError::UnknownSector { .. })
    ));
    assert!(matches!(
        checked(&ir, vec![make(vec![piece(1, 0, 4)])]),
        Err(SliceError::RangeOutOfBounds { .. })
    ));
    assert!(matches!(
        checked(&ir, vec![make(vec![piece(1, 0, 2), piece(1, 1, 3)])]),
        Err(SliceError::OverlappingRanges { .. })
    ));
    assert!(matches!(
        checked(&ir, vec![make(vec![piece(1, 0, 1), piece(1, 2, 3)])]),
        Err(SliceError::IncompleteCoverage { .. })
    ));
}

#[test]
fn distinguishes_empty_leg_from_no_sliced_label() {
    let ir = internal_ir();
    let unsliced = checked(&ir, vec![]).unwrap();
    let empty_leg = checked(
        &ir,
        vec![spec(
            "x",
            axis(0),
            SectorLeg::new(Vec::<(SectorId, usize)>::new(), false),
            vec![],
        )],
    )
    .unwrap();

    assert!(unsliced.is_empty());
    assert_eq!(unsliced.nslices(), 1);
    assert_eq!(unsliced.combinations().count(), 1);
    assert!(!empty_leg.is_empty());
    assert_eq!(empty_leg.nslices(), 0);
    assert_eq!(empty_leg.combinations().count(), 0);
}

#[test]
fn canonical_effective_occurrence_owns_non_self_dual_sector_identity() {
    let written_labels = [label("p"), label("x")];
    let effective_labels = vec![written_labels[1].clone(), written_labels[0].clone()];
    assert_eq!(effective_labels, vec![label("x"), label("p")]);
    let ir =
        NetworkIR::from_labels(vec![effective_labels, vec![label("x")]], vec![label("p")]).unwrap();
    let authority_charge = U1Irrep::new(-1).unwrap();
    let authority_leg = SectorLeg::new([(authority_charge, 2)], true);
    let partner_leg = authority_leg.dual(&U1FusionRule);
    let plan = checked(
        &ir,
        vec![spec(
            "x",
            axis_at(0, 0),
            authority_leg.clone(),
            vec![SectorSlice::new(authority_charge.into(), range(0, 2))],
        )],
    )
    .unwrap();

    // This explicitly models the post-adjoint rotation [p,x] -> [x,p]: the
    // first effective occurrence is the dual q=-1 authority at tensor0 axis0.
    // Its partner is q=+1, but the schema stores only the authority raw id;
    // later binding interprets the partner through the validated dual space.
    assert_eq!(plan.indices()[0].authority(), axis_at(0, 0));
    assert_eq!(plan.indices()[0].authority_leg(), &authority_leg);
    assert_eq!(
        plan.indices()[0].pieces()[0].sector(),
        authority_charge.into()
    );
    assert_eq!(
        partner_leg.sectors(),
        &[SectorId::from(U1Irrep::new(1).unwrap())]
    );
    assert!(!partner_leg.is_dual());
    assert!(matches!(
        checked(
            &ir,
            vec![spec(
                "x",
                axis_at(0, 1),
                authority_leg,
                vec![SectorSlice::new(authority_charge.into(), range(0, 2))],
            )]
        ),
        Err(SliceError::InvalidAuthority {
            expected,
            actual,
            ..
        }) if expected == axis_at(0, 0) && actual == axis_at(0, 1)
    ));
}

#[test]
fn rule_identity_seals_numeric_sector_meaning_without_coefficients() {
    let odd: SectorId = FermionParityIrrep::ODD.into();
    let charge_minus_one: SectorId = U1Irrep::new(-1).unwrap().into();
    assert_eq!(
        odd, charge_minus_one,
        "fixture requires the raw-id collision"
    );
    let ir = internal_ir();
    let shared_spec = || {
        vec![spec(
            "x",
            axis(0),
            SectorLeg::new([(odd, 1)], false),
            vec![SectorSlice::new(odd, range(0, 1))],
        )]
    };
    let z2 = ZNFusionRule::new(2).unwrap();
    let bosonic = SymmetricSlicePlan::try_new(&ir, z2.rule_identity(), shared_spec()).unwrap();
    let fermionic = SymmetricSlicePlan::try_new(
        &ir,
        FermionParityFusionRule::new().rule_identity(),
        shared_spec(),
    )
    .unwrap();

    assert_ne!(bosonic, fermionic);
    assert_eq!(bosonic.rule_identity(), &z2.rule_identity());
    assert_eq!(
        fermionic.rule_identity(),
        &FermionParityFusionRule::new().rule_identity()
    );
    // Both plans enumerate the same coordinate piece. The identity seal is the
    // only categorical distinction: no sign or fusion-tree field is introduced.
    assert_eq!(bosonic.indices(), fermionic.indices());
    assert_eq!(bosonic.combinations().count(), 1);
    assert_eq!(fermionic.combinations().count(), 1);
}

#[test]
fn structural_zero_sector_combinations_remain_in_enumeration() {
    let q0: SectorId = U1Irrep::new(0).unwrap().into();
    let q1: SectorId = U1Irrep::new(1).unwrap().into();
    assert_eq!(U1FusionRule.nsymbol(q1, q1, q0), 0);

    // One effective rank-three tensor carries the fusion-tree boundary
    // (a,b;c), so the three sliced labels refer to the same N-symbol triple.
    let ir = NetworkIR::from_labels(
        vec![vec![label("a"), label("b"), label("c")]],
        vec![label("a"), label("b"), label("c")],
    )
    .unwrap();
    let two_sectors = || SectorLeg::new([(q0, 1), (q1, 1)], false);
    let two_pieces = || {
        vec![
            SectorSlice::new(q0, range(0, 1)),
            SectorSlice::new(q1, range(0, 1)),
        ]
    };
    let plan = checked(
        &ir,
        vec![
            SymmetricSliceSpec::new(label("a"), axis_at(0, 0), two_sectors(), two_pieces()),
            SymmetricSliceSpec::new(label("b"), axis_at(0, 1), two_sectors(), two_pieces()),
            SymmetricSliceSpec::new(
                label("c"),
                axis_at(0, 2),
                SectorLeg::new([(q0, 1)], false),
                vec![SectorSlice::new(q0, range(0, 1))],
            ),
        ],
    )
    .unwrap();

    assert_eq!(plan.nslices(), 4);
    assert!(plan.combinations().any(|combination| {
        combination
            .iter()
            .map(|piece| piece.sector())
            .eq([q1, q1, q0])
    }));
}
