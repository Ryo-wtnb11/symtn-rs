//! The infallible Generic testing tier weighs a sector by its exact quantum
//! dimension, as the multiplicity-free tier does (#2076).
//!
//! TensorKitSectors `dim(::Sector)` is an exact categorical dimension (`Int`
//! for SU(2)); TensorKit `findtruncated(::SectorVector, ...)` compares
//! `totaldim > howmany` against it without slack (`truncation.jl:198`). The
//! oracle is SU(2) through both tiers: `SU2FusionRule` (multiplicity-free) and
//! the same data lifted to `GenericRigidSymbols`, whose `dim_scalar` is the
//! hand-written `2j + 1`.

use super::*;
use crate::seam::decide_bond_truncation;
use crate::TruncationDecision;
use std::collections::BTreeMap;
use symtn_core::MultiplicityFreeAdmissionMode;

/// The checked-tier truncation decision, with the provider's own identity and
/// sector order.
fn checked_decision(
    provider: &InfallibleGeneric<'_, Su2Generic>,
    spectra: &[SectorSpectrum],
    truncation: &Truncation,
) -> TruncationDecision {
    decide_bond_truncation::<CheckedGenericAdmissionMode, _, _>(
        provider,
        spectra,
        truncation,
        &CheckedGenericFusion::rule_identity(provider),
        |sector| CheckedGenericFusion::sector_order_key(provider, sector),
    )
    .unwrap()
}

/// SU(2) as an infallible Generic provider: 1x1 symbol blocks lifted from
/// [`SU2FusionRule`], under its own identity and the Generic style.
struct Su2Generic;

impl symtn_core::testing::Provider for Su2Generic {}

impl FusionRule for Su2Generic {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    // Generic, as the checked tier's root constructor admits only Generic
    // providers; SU(2) has no multiplicity, so the geometry is unchanged.
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        SU2FusionRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        SU2FusionRule.nsymbol(left, right, coupled)
    }
    fn sector_order_key(&self, sector: SectorId) -> symtn_core::SectorOrderKey {
        SU2FusionRule.sector_order_key(sector)
    }
}

impl GenericFusionSymbols for Su2Generic {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<f64> {
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        let value = SU2FusionRule.f_symbol_scalar(a, b, c, d, e, f);
        GenericFArray::new(vec![value; len], shape)
    }

    fn r_symbol_generic(&self, a: SectorId, b: SectorId, c: SectorId) -> GenericRMatrix<f64> {
        let (rows, cols) = (self.nsymbol(a, b, c), self.nsymbol(b, a, c));
        let value = SU2FusionRule.r_symbol_scalar(a, b, c);
        GenericRMatrix::new(vec![value; rows * cols], rows, cols)
    }
}

impl GenericRigidSymbols for Su2Generic {
    fn dim_scalar(&self, sector: SectorId) -> f64 {
        (sector.id() + 1) as f64
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        self.dim_scalar(sector).sqrt()
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        self.sqrt_dim_scalar(sector).recip()
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> f64 {
        SU2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

/// `(2j, values)` per sector: dims 1, 2, 3, 5, 7, 8; `sqrt(dim)^2` is off
/// for all but 1.
fn spectra() -> Vec<SectorSpectrum> {
    [
        (0, vec![0.93, 0.41]),
        (1, vec![0.87, 0.52, 0.11]),
        (2, vec![0.74, 0.29]),
        (4, vec![0.66, 0.2]),
        (6, vec![0.58]),
        (7, vec![0.47]),
    ]
    .into_iter()
    .map(|(twice, values)| SectorSpectrum {
        sector: SectorId::new(twice),
        values,
    })
    .collect()
}

#[test]
fn infallible_generic_reports_the_exact_dimension() {
    // What: the checked view's dim is `2j + 1`, not `sqrt(2j + 1)^2`, so the
    // trace channel factor `dim(c) / dim(a)` is the exact quotient.
    let provider = InfallibleGeneric::new(&Su2Generic);
    let mut inexact_squares = 0;
    for twice in 0..8 {
        let sector = SectorId::new(twice);
        let dim = (twice + 1) as f64;
        assert_eq!(
            provider.try_dim_scalar(sector).unwrap(),
            dim,
            "2j = {twice}"
        );
        let sqrt_dim = provider.try_sqrt_dim_scalar(sector).unwrap();
        inexact_squares += usize::from(sqrt_dim * sqrt_dim != dim);
    }
    // The sqrt² route would differ, so the assertions above discriminate.
    assert!(inexact_squares >= 4);
    let quotient = |c: usize, a: usize| {
        provider.try_dim_scalar(SectorId::new(c)).unwrap()
            / provider.try_dim_scalar(SectorId::new(a)).unwrap()
    };
    assert_eq!(quotient(4, 2), 5.0 / 3.0);
    assert_eq!(quotient(6, 7), 7.0 / 8.0);
}

#[test]
fn infallible_generic_keeps_a_state_that_meets_the_rank_budget_exactly() {
    // What: one spin-2 value weighs 5; a budget of 5 keeps it and 4 does not.
    // `sqrt(5)^2 = 5.000000000000001` would overflow the budget of 5.
    let provider = InfallibleGeneric::new(&Su2Generic);
    let spectra = [SectorSpectrum {
        sector: SectorId::new(4),
        values: vec![1.0],
    }];
    for (budget, kept) in [(4, false), (5, true)] {
        let decision = checked_decision(&provider, &spectra, &Truncation::rank(budget));
        assert_eq!(decision.kept, vec![vec![kept]], "budget {budget}");
    }
}

#[test]
fn multiplicity_free_and_infallible_generic_take_one_rank_decision() {
    // What: SU(2) through either tier keeps the same states and reports the
    // same discarded weight under every rank budget.
    let spectra = spectra();
    let provider = InfallibleGeneric::new(&Su2Generic);
    let total: usize = spectra
        .iter()
        .map(|entry| (entry.sector.id() + 1) * entry.values.len())
        .sum();
    for budget in 0..=total + 1 {
        let truncation = Truncation::rank(budget);
        let mf = decide_bond_truncation::<MultiplicityFreeAdmissionMode, _, _>(
            &SU2FusionRule,
            &spectra,
            &truncation,
            &FusionRule::rule_identity(&SU2FusionRule),
            |sector| FusionRule::sector_order_key(&SU2FusionRule, sector),
        )
        .unwrap();
        let generic = checked_decision(&provider, &spectra, &truncation);
        assert_eq!(mf.kept, generic.kept, "budget {budget}");
        assert_eq!(
            mf.error.to_bits(),
            generic.error.to_bits(),
            "budget {budget}"
        );
    }
}

/// One block of a published bond factor: its coupled sector, shape and
/// column-major values, each as its round-trip `Debug` text (bit-exact,
/// sign of zero included).
type BlockBits = (SectorId, Vec<usize>, Vec<String>);

/// The rectangular `S` of a full SVD on one SU(2) bond `W_row <- W_col`,
/// published through mode `M`'s factor-space authority, in block order.
fn rectangular_bond_blocks<M, R, D>(space: &BoundDynamicFusionMapSpace<R>) -> Vec<BlockBits>
where
    M: crate::seam::FactorMode<R>,
    M::Error: fmt::Debug,
    D: crate::factorize::FactorScalar + fmt::Debug,
{
    let sector = SectorId::new;
    let rows = BTreeMap::from([(sector(0), 2), (sector(2), 3), (sector(4), 1)]);
    let cols = BTreeMap::from([(sector(0), 3), (sector(2), 2), (sector(6), 2)]);
    // Sector 0 carries one value fewer than its `min(rows, cols)` diagonal;
    // sectors 4 and 6 have no partner and so no block.
    let spectra = [
        SectorSpectrum {
            sector: sector(2),
            values: vec![0.7, 1.0 / 3.0],
        },
        SectorSpectrum {
            sector: sector(0),
            values: vec![2.5],
        },
    ];
    let s = crate::seam::rectangular_diagonal_bond_tensor::<_, R, D>(
        &M::authority(space),
        &spectra,
        &rows,
        &cols,
    )
    .unwrap();
    let structure = s.space().space().structure();
    (0..structure.block_count())
        .map(|index| {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(tree) = block.key() else {
                unreachable!("a bond factor has fusion-tree blocks");
            };
            let (shape, strides) = (block.shape(), block.strides());
            let values = (0..shape[1])
                .flat_map(|col| (0..shape[0]).map(move |row| (row, col)))
                .map(|(row, col)| {
                    format!(
                        "{:?}",
                        s.data()[block.offset() + row * strides[0] + col * strides[1]]
                    )
                })
                .collect();
            (tree.codomain_tree().coupled(), shape.to_vec(), values)
        })
        .collect()
}

#[test]
fn multiplicity_free_and_checked_publish_one_rectangular_bond() {
    // What: the full-SVD `S` on SU(2) through either tier has the same
    // blocks, shapes and values bit for bit, for a real and a complex
    // payload.
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(2), 1)], false)]),
    );
    let mf = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(SU2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(InfallibleGeneric::new(&Su2Generic)),
        homspace,
    )
    .unwrap();
    let real = rectangular_bond_blocks::<MultiplicityFreeAdmissionMode, _, f64>(&mf);
    assert_eq!(
        real.iter()
            .map(|(sector, shape, _)| (sector.id(), shape.clone()))
            .collect::<Vec<_>>(),
        [(0, vec![2, 3]), (2, vec![3, 2])],
    );
    assert_eq!(
        real,
        rectangular_bond_blocks::<CheckedGenericAdmissionMode, _, f64>(&checked)
    );
    assert_eq!(
        rectangular_bond_blocks::<MultiplicityFreeAdmissionMode, _, Complex64>(&mf),
        rectangular_bond_blocks::<CheckedGenericAdmissionMode, _, Complex64>(&checked),
    );
}
