use super::*;

// `A4FoldRule` (N(3,3,3) = 2) with symbols an inconsistent user provider of
// the open `GenericFusionSymbols` trait could return: R is always 1x1 and the
// bend F-block F(a, b, b̄, a, c, 1) is optionally truncated to one element.
#[derive(Clone, Copy, Debug)]
struct TruncatedSymbolRule {
    truncate_bend_f: bool,
}

const TRUNCATED_R: TruncatedSymbolRule = TruncatedSymbolRule {
    truncate_bend_f: false,
};

impl symtn_sectors::testing::Provider for TruncatedSymbolRule {}

impl FusionRule for TruncatedSymbolRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        A4FoldRule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        A4FoldRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        A4FoldRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        A4FoldRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        A4FoldRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        A4FoldRule.nsymbol(left, right, coupled)
    }
}

impl GenericFusionSymbols for TruncatedSymbolRule {
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
        if self.truncate_bend_f && f == self.vacuum() {
            GenericFArray::new(vec![1.0], (1, 1, 1, 1))
        } else {
            A4FoldRule.f_symbol_generic(a, b, c, d, e, f)
        }
    }
    fn r_symbol_generic(&self, _a: SectorId, _b: SectorId, _c: SectorId) -> GenericRMatrix<f64> {
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

impl GenericRigidSymbols for TruncatedSymbolRule {
    fn dim_scalar(&self, sector: SectorId) -> f64 {
        A4FoldRule.dim_scalar(sector)
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        A4FoldRule.sqrt_dim_scalar(sector)
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        A4FoldRule.inv_sqrt_dim_scalar(sector)
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> f64 {
        A4FoldRule.frobenius_schur_phase_scalar(sector)
    }
}

fn shape_mismatch() -> CoreError {
    CoreError::MalformedFusionTree {
        message: "Generic symbol shape mismatch",
    }
}

#[test]
fn generic_braid_tree_rejects_r_symbol_shape_mismatch() {
    // What: a 1x1 R for N(3,3,3) = 2 is a typed malformed-input error on the
    // infallible braid, as on the sibling F-move path, not a panic.
    let tree = a4_pair_rank2(1).codomain_tree().clone();
    let error = generic_braid_tree(&TRUNCATED_R, &tree, &[1, 0], &[0, 1]).unwrap_err();
    assert_eq!(error, shape_mismatch());
}

#[test]
fn generic_braid_tree_pair_rejects_r_symbol_shape_mismatch() {
    let error = generic_braid_tree_pair(
        &TRUNCATED_R,
        &a4_pair_rank2(1),
        &[1, 0],
        &[2],
        &[0, 1],
        &[2],
    )
    .unwrap_err();
    assert_eq!(error, shape_mismatch());
}

#[test]
fn generic_repartition_rejects_default_b_symbol_shape_mismatch() {
    // What: the native testing default reshapes this truncated F into a 1×1 B;
    // core rejects that B against N(3,3,3) = 2 before indexing.
    let rule = TruncatedSymbolRule {
        truncate_bend_f: true,
    };
    let error = generic_repartition_tree_pair(&rule, &a4_pair_rank2(2), 1).unwrap_err();
    assert_eq!(error, shape_mismatch());
}
