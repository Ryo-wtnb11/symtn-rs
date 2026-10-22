#[path = "support/report.rs"]
mod report;

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use symtn::typed::ContractSpec;
use symtn::typed::Direction;

use symtn::sector::SectorId;
use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorVec, TypedSectorAdmission,
};
use symtn::typed::{
    CheckedGenericPlanError, GenericTensorError, GradedSpace, TensorMap, TensorScalar,
};
use symtn::typed::{Complex64, Error, OperationError, Runtime};

use report::cause;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Label {
    Unit,
    X,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PivotalError {
    InvalidSector,
    FrobeniusSchur,
    Twist,
}

impl fmt::Display for PivotalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PivotalError {}

struct CheckedPivotalToy {
    identity_tag: u8,
    braiding: BraidingStyleKind,
    x_fs: f64,
    x_twist: f64,
    style_queries: AtomicUsize,
    vacuum_queries: AtomicUsize,
    fs_queries: AtomicUsize,
    twist_queries: AtomicUsize,
    other_queries: AtomicUsize,
    post_stage_queries: AtomicUsize,
    stage_after_twists: AtomicUsize,
    staged: AtomicBool,
    fail_twist_on: AtomicUsize,
    fail_fs_on: AtomicUsize,
    twist_sectors: Mutex<Vec<SectorId>>,
}

impl CheckedPivotalToy {
    fn new(identity_tag: u8, braiding: BraidingStyleKind, x_twist: f64) -> Self {
        Self::with_fs(identity_tag, braiding, 1.0, x_twist)
    }

    fn with_fs(identity_tag: u8, braiding: BraidingStyleKind, x_fs: f64, x_twist: f64) -> Self {
        Self {
            identity_tag,
            braiding,
            x_fs,
            x_twist,
            style_queries: AtomicUsize::new(0),
            vacuum_queries: AtomicUsize::new(0),
            fs_queries: AtomicUsize::new(0),
            twist_queries: AtomicUsize::new(0),
            other_queries: AtomicUsize::new(0),
            post_stage_queries: AtomicUsize::new(0),
            stage_after_twists: AtomicUsize::new(0),
            staged: AtomicBool::new(false),
            fail_twist_on: AtomicUsize::new(0),
            fail_fs_on: AtomicUsize::new(0),
            twist_sectors: Mutex::new(Vec::new()),
        }
    }

    fn unit() -> SectorId {
        SectorId::new(0)
    }

    fn x() -> SectorId {
        SectorId::new(1)
    }

    fn reset_ledger(&self, stage_after_twists: usize) {
        self.style_queries.store(0, Ordering::Relaxed);
        self.vacuum_queries.store(0, Ordering::Relaxed);
        self.fs_queries.store(0, Ordering::Relaxed);
        self.twist_queries.store(0, Ordering::Relaxed);
        self.other_queries.store(0, Ordering::Relaxed);
        self.post_stage_queries.store(0, Ordering::Relaxed);
        self.stage_after_twists
            .store(stage_after_twists, Ordering::Relaxed);
        self.staged.store(false, Ordering::Relaxed);
        self.fail_twist_on.store(0, Ordering::Relaxed);
        self.fail_fs_on.store(0, Ordering::Relaxed);
        self.twist_sectors.lock().unwrap().clear();
    }

    fn record_query(&self) {
        if self.staged.load(Ordering::Relaxed) {
            self.post_stage_queries.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_other_query(&self) {
        self.record_query();
        self.other_queries.fetch_add(1, Ordering::Relaxed);
    }

    fn finish_observation(&self) {
        self.staged.store(false, Ordering::Relaxed);
    }

    fn fusion_channels(left: SectorId, right: SectorId) -> SectorVec {
        match (left, right) {
            (left, right) if left == Self::unit() => [right].into_iter().collect(),
            (left, right) if right == Self::unit() => [left].into_iter().collect(),
            (left, right) if left == Self::x() && right == Self::x() => {
                [Self::x()].into_iter().collect()
            }
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if left == Self::x() && right == Self::x() && coupled == Self::x() {
            2
        } else {
            usize::from(Self::fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl symtn_core::testing::Provider for CheckedPivotalToy {}

impl CheckedGenericFusion for CheckedPivotalToy {
    type Error = PivotalError;

    fn rule_identity(&self) -> RuleIdentity {
        self.record_other_query();
        RuleIdentity::from_canonical_bytes::<Self>(
            0x951,
            Arc::<[u8]>::from(vec![
                self.identity_tag,
                self.braiding as u8,
                u8::from(self.x_fs.is_sign_negative()),
                u8::from(self.x_twist.is_sign_negative()),
            ]),
        )
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.record_other_query();
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.record_query();
        self.style_queries.fetch_add(1, Ordering::Relaxed);
        self.braiding
    }

    fn vacuum(&self) -> SectorId {
        self.record_query();
        self.vacuum_queries.fetch_add(1, Ordering::Relaxed);
        Self::unit()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.record_other_query();
        match sector {
            sector if sector == Self::unit() || sector == Self::x() => Ok(sector),
            _ => Err(PivotalError::InvalidSector),
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.record_other_query();
        Ok(Self::fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.record_other_query();
        Ok(Self::nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedPivotalToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.record_other_query();
        Ok(1.0)
    }

    fn try_inv_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.record_other_query();
        Ok(1.0)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        let query = self.fs_queries.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_fs_on.load(Ordering::Relaxed) == query {
            return Err(PivotalError::FrobeniusSchur);
        }
        if sector == Self::unit() {
            Ok(1.0)
        } else if sector == Self::x() {
            Ok(self.x_fs)
        } else {
            Err(PivotalError::InvalidSector)
        }
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.record_other_query();
        let shape = (
            Self::nsymbol(a, b, e),
            Self::nsymbol(e, c, d),
            Self::nsymbol(b, c, f),
            Self::nsymbol(a, f, d),
        );
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        Ok(GenericFArray::new(vec![0.0; len], shape))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.record_other_query();
        let rows = Self::nsymbol(a, b, c);
        let cols = Self::nsymbol(b, a, c);
        Ok(GenericRMatrix::new(vec![0.0; rows * cols], rows, cols))
    }
}

impl CheckedGenericPivotal for CheckedPivotalToy {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        let query = self.twist_queries.fetch_add(1, Ordering::Relaxed) + 1;
        self.twist_sectors.lock().unwrap().push(sector);
        if self.fail_twist_on.load(Ordering::Relaxed) == query {
            return Err(PivotalError::Twist);
        }
        let value = if sector == Self::unit() {
            1.0
        } else if sector == Self::x() {
            self.x_twist
        } else {
            return Err(PivotalError::InvalidSector);
        };
        if self.stage_after_twists.load(Ordering::Relaxed) == query {
            self.staged.store(true, Ordering::Relaxed);
        }
        Ok(value)
    }
}

impl TypedSectorAdmission for CheckedPivotalToy {
    type Sector = Label;
    type Error = PivotalError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Label) -> Result<SectorId, Self::Error> {
        self.record_other_query();
        Ok(match sector {
            Label::Unit => Self::unit(),
            Label::X => Self::x(),
        })
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Label, Self::Error> {
        self.record_other_query();
        if sector == Self::unit() {
            Ok(Label::Unit)
        } else if sector == Self::x() {
            Ok(Label::X)
        } else {
            Err(PivotalError::InvalidSector)
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
}

fn label_marker(label: Label) -> usize {
    match label {
        Label::Unit => 1,
        Label::X => 7,
    }
}

fn tree_marker(trees: &symtn::typed::BlockFusionTrees<Label>) -> usize {
    10_000 * label_marker(*trees.coupled())
        + trees
            .codomain_uncoupled()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 100 * label_marker(label))
            .sum::<usize>()
        + trees
            .codomain_innerlines()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 1_000 * label_marker(label))
            .sum::<usize>()
        + trees
            .codomain_vertices()
            .iter()
            .enumerate()
            .map(|(index, vertex)| (index + 1) * 10 * vertex.get())
            .sum::<usize>()
        + trees
            .domain_uncoupled()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 2_000 * label_marker(label))
            .sum::<usize>()
        + trees
            .domain_innerlines()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 3_000 * label_marker(label))
            .sum::<usize>()
        + trees
            .domain_vertices()
            .iter()
            .enumerate()
            .map(|(index, vertex)| (index + 1) * 20 * vertex.get())
            .sum::<usize>()
}

fn fixture<D>(
    runtime: &Runtime,
    provider: &Arc<CheckedPivotalToy>,
    value: impl Fn(usize) -> D,
) -> TensorMap<CheckedPivotalToy, D>
where
    D: TensorScalar,
{
    let mixed =
        GradedSpace::try_new(Arc::clone(provider), [(Label::Unit, 1), (Label::X, 2)]).unwrap();
    let x = GradedSpace::try_new(Arc::clone(provider), [(Label::X, 1)]).unwrap();
    TensorMap::from_subblock_fn(runtime, [&mixed, &x], [&x], |trees, indices| {
        value(
            tree_marker(trees)
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, index)| (axis + 1) * index)
                    .sum::<usize>(),
        )
    })
    .unwrap()
}

fn assert_values<D>(
    tensor: &TensorMap<CheckedPivotalToy, D>,
    value: impl Fn(usize) -> D,
    factor: impl Fn(&symtn::typed::BlockFusionTrees<Label>) -> f64,
) where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    for block_index in 0..tensor.subblock_count() {
        let block = tensor.subblock(block_index).unwrap();
        let trees = tensor.subblock_fusion_trees(block_index).unwrap();
        let elements = block.shape().iter().product::<usize>();
        for linear in 0..elements {
            let mut remainder = linear;
            let mut position = block.offset();
            let mut marker = tree_marker(&trees);
            for (axis, (&extent, &stride)) in block.shape().iter().zip(block.strides()).enumerate()
            {
                let index = remainder % extent;
                remainder /= extent;
                position += index * stride;
                marker += (axis + 1) * index;
            }
            assert_eq!(
                tensor.dense_data().unwrap()[position],
                value(marker) * D::from_real(factor(&trees))
            );
        }
    }
}

fn assert_nontrivial_case<D>(value: impl Fn(usize) -> D + Copy)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(1, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, value);
    let before = source.dense_data().unwrap().to_vec();
    let mut saw_outer_two = false;
    let mut saw_vertex_two = false;
    for index in 0..source.subblock_count() {
        let block = source.subblock(index).unwrap();
        let trees = source.subblock_fusion_trees(index).unwrap();
        saw_outer_two |= trees.codomain_uncoupled()[0] == Label::X && block.shape()[0] == 2;
        saw_vertex_two |= trees
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2);
    }
    assert!(saw_outer_two);
    assert!(saw_vertex_two);

    provider.reset_ledger(2);
    let codomain = source.twist(&[0], Direction::Forward).unwrap();
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.other_queries.load(Ordering::Relaxed), 0);
    assert_eq!(
        provider.twist_sectors.lock().unwrap().as_slice(),
        &[CheckedPivotalToy::unit(), CheckedPivotalToy::x()]
    );
    provider.finish_observation();
    assert!(std::ptr::eq(codomain.provider(), provider.as_ref()));
    assert_ne!(
        codomain.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
    assert_eq!(codomain.codomain(), source.codomain());
    assert_eq!(codomain.domain(), source.domain());
    assert_values(&codomain, value, |trees| {
        if trees.codomain_uncoupled()[0] == Label::X {
            -1.0
        } else {
            1.0
        }
    });

    provider.reset_ledger(1);
    let domain = source.twist(&[2], Direction::Inverse).unwrap();
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_values(&domain, value, |_| -1.0);

    provider.reset_ledger(2);
    let repeated = source.twist(&[0, 0], Direction::Forward).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_values(&repeated, value, |_| 1.0);

    let lazy = source.adjoint().unwrap();
    // What: logical adjoint axis 1 maps to parent codomain axis 0; logical
    // axis 0 would instead map to the uniformly-X parent domain axis 2.
    provider.reset_ledger(2);
    let lazy_twisted = lazy.twist(&[1], Direction::Forward).unwrap();
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    let direct = source
        .twist(&[0], Direction::Inverse)
        .unwrap()
        .adjoint()
        .unwrap();
    assert_eq!(
        lazy_twisted.materialize().unwrap().dense_data().unwrap(),
        direct.materialize().unwrap().dense_data().unwrap()
    );
    assert!(std::ptr::eq(lazy_twisted.provider(), provider.as_ref()));
    assert_eq!(source.dense_data().unwrap(), before);
}

#[test]
fn checked_generic_twist_scales_full_keys_for_real_and_complex_payloads() {
    // What: +1/-1 factors act on exact Generic tree keys, including μ=2
    // vertices and outer degeneracy two, for codomain/domain/repeated axes and
    // direct/lazy orientations without changing the input or provider Arc.
    assert_nontrivial_case(|marker| marker as f64);
    assert_nontrivial_case(|marker| Complex64::new(marker as f64, -(marker as f64) / 10.0));
}

#[test]
fn checked_generic_twist_precedence_and_late_failure_are_typed_and_nonpublishing() {
    // What: index and empty requests precede every provider query; NoBraiding
    // rejection is an early facade error, while a staged pivotal failure keeps
    // its provider type and leaves the source payload unchanged.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(2, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, |marker| marker as f64);
    let before = source.dense_data().unwrap().to_vec();

    provider.reset_ledger(0);
    assert!(matches!(
        source.twist(&[source.rank()], Direction::Forward),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.other_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    let empty = source.twist(&[], Direction::Forward).unwrap();
    assert_eq!(
        empty.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.other_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(2);
    provider.fail_twist_on.store(2, Ordering::Relaxed);
    assert!(matches!(
        source.twist(&[0], Direction::Forward),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::Twist
        )))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    assert_eq!(source.dense_data().unwrap(), before);
}

#[test]
fn checked_generic_twist_handles_nobraiding_bosonic_and_staged_identity_sharing() {
    // What: TensorKit's NoBraiding unit carve-out and bosonic short circuit
    // share storage without pivotal queries; an anyonic all-one provider stages
    // values once and then takes the same identity-sharing path.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let no_braiding = Arc::new(CheckedPivotalToy::new(
        3,
        BraidingStyleKind::NoBraiding,
        -1.0,
    ));
    let unit = GradedSpace::try_new(Arc::clone(&no_braiding), [(Label::Unit, 1)]).unwrap();
    let unit_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&unit], [&unit], |_, _| 3.0).unwrap();
    no_braiding.reset_ledger(0);
    let unit_twist = unit_tensor.twist(&[0, 1], Direction::Forward).unwrap();
    assert_eq!(
        unit_twist.dense_data().unwrap().as_ptr(),
        unit_tensor.dense_data().unwrap().as_ptr()
    );
    assert_eq!(no_braiding.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.vacuum_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.twist_queries.load(Ordering::Relaxed), 0);

    let nonunit = fixture(&runtime, &no_braiding, |marker| marker as f64);
    let before = nonunit.dense_data().unwrap().to_vec();
    no_braiding.reset_ledger(0);
    assert!(matches!(
        nonunit.twist(&[0], Direction::Forward),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(no_braiding.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.vacuum_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.twist_queries.load(Ordering::Relaxed), 0);
    assert_eq!(nonunit.dense_data().unwrap(), before);

    let bosonic = Arc::new(CheckedPivotalToy::new(4, BraidingStyleKind::Bosonic, -1.0));
    let bosonic_tensor = fixture(&runtime, &bosonic, |marker| marker as f64);
    bosonic.reset_ledger(0);
    let bosonic_twist = bosonic_tensor.twist(&[0], Direction::Forward).unwrap();
    assert_eq!(
        bosonic_twist.dense_data().unwrap().as_ptr(),
        bosonic_tensor.dense_data().unwrap().as_ptr()
    );
    assert_eq!(bosonic.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(bosonic.vacuum_queries.load(Ordering::Relaxed), 0);
    assert_eq!(bosonic.twist_queries.load(Ordering::Relaxed), 0);

    let identity = Arc::new(CheckedPivotalToy::new(5, BraidingStyleKind::Anyonic, 1.0));
    let identity_tensor = fixture(&runtime, &identity, |marker| marker as f64);
    identity.reset_ledger(2);
    let identity_twist = identity_tensor.twist(&[0], Direction::Forward).unwrap();
    assert_eq!(identity.post_stage_queries.load(Ordering::Relaxed), 0);
    assert_eq!(identity.twist_queries.load(Ordering::Relaxed), 2);
    identity.finish_observation();
    assert_eq!(
        identity_twist.dense_data().unwrap().as_ptr(),
        identity_tensor.dense_data().unwrap().as_ptr()
    );
    assert!(std::ptr::eq(identity_twist.provider(), identity.as_ref()));
}

fn assert_flip_nontrivial_case<D>(value: impl Fn(usize) -> D + Copy)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(6, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, value);
    let before = source.dense_data().unwrap().to_vec();
    let source_codomain = source.codomain();
    let source_domain = source.domain();

    provider.reset_ledger(2);
    let codomain = source.flip(&[0], Direction::Inverse).unwrap();
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert!(std::ptr::eq(codomain.provider(), provider.as_ref()));
    assert_eq!(
        codomain.codomain()[0].is_dual(),
        !source.codomain()[0].is_dual()
    );
    assert_values(&codomain, value, |trees| {
        if trees.codomain_uncoupled()[0] == Label::X {
            -1.0
        } else {
            1.0
        }
    });

    provider.reset_ledger(2);
    let domain = source.flip(&[2], Direction::Forward).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_eq!(domain.domain()[0].is_dual(), !source.domain()[0].is_dual());
    assert_values(&domain, value, |_| -1.0);

    provider.reset_ledger(0);
    match source.flip(&[0, 0], Direction::Forward) {
        Err(GenericTensorError::Facade(Error::InvalidArgument(message))) => {
            assert!(message.contains("listed more than once"), "{message}");
        }
        other => panic!("expected InvalidArgument, got {:?}", other.map(|_| ())),
    }
    for counter in [
        &provider.style_queries,
        &provider.vacuum_queries,
        &provider.other_queries,
        &provider.fs_queries,
        &provider.twist_queries,
        &provider.post_stage_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
    provider.finish_observation();

    let lazy = source.adjoint().unwrap();
    provider.reset_ledger(2);
    let lazy_flipped = lazy.flip(&[1], Direction::Forward).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_eq!(
        lazy_flipped.domain()[0].is_dual(),
        !lazy.domain()[0].is_dual()
    );
    provider.reset_ledger(2);
    let direct = source
        .flip(&[0], Direction::Inverse)
        .unwrap()
        .adjoint()
        .unwrap();
    provider.finish_observation();
    assert_eq!(
        lazy_flipped.materialize().unwrap().dense_data().unwrap(),
        direct.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(lazy_flipped.codomain(), direct.codomain());
    assert_eq!(lazy_flipped.domain(), direct.domain());
    assert!(std::ptr::eq(lazy_flipped.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::runtime_identity(lazy_flipped.runtime()).matches(direct.runtime())
    );
    assert_eq!(source.dense_data().unwrap(), before);
    assert_eq!(source.codomain(), source_codomain);
    assert_eq!(source.domain(), source_domain);
}

#[test]
fn checked_generic_flip_scales_full_keys_and_lazy_duality_for_real_and_complex_payloads() {
    // What: checked Generic replay uses staged χ/θ factors on complete μ=2
    // keys; repeated legs fail before provider queries, and lazy results own toggled metadata.
    assert_flip_nontrivial_case(|marker| marker as f64);
    assert_flip_nontrivial_case(|marker| Complex64::new(marker as f64, -(marker as f64) / 10.0));
}

#[test]
fn checked_generic_flip_rejects_nobraiding_before_pivotal_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(
        7,
        BraidingStyleKind::NoBraiding,
        -1.0,
    ));
    let source = fixture(&runtime, &provider, |marker| marker as f64);

    provider.reset_ledger(0);
    match source.flip(&[0, 0], Direction::Forward) {
        Err(GenericTensorError::Facade(Error::InvalidArgument(message))) => {
            assert!(message.contains("listed more than once"), "{message}");
        }
        other => panic!("expected InvalidArgument, got {:?}", other.map(|_| ())),
    }
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    assert!(matches!(
        source.flip(&[0], Direction::Forward),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.vacuum_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_flip_precedence_and_staged_failures_are_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(8, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, |marker| marker as f64);
    let before = source.dense_data().unwrap().to_vec();

    provider.reset_ledger(0);
    assert!(matches!(
        source.flip(&[source.rank()], Direction::Forward),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    let empty = source.flip(&[], Direction::Forward).unwrap();
    assert_eq!(
        empty.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    provider.fail_fs_on.store(2, Ordering::Relaxed);
    assert!(matches!(
        source.flip(&[0], Direction::Forward),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::FrobeniusSchur
        )))
    ));
    assert_eq!(source.dense_data().unwrap(), before);

    provider.reset_ledger(0);
    provider.fail_twist_on.store(2, Ordering::Relaxed);
    assert!(matches!(
        source.flip(&[0], Direction::Forward),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::Twist
        )))
    ));
    assert_eq!(source.dense_data().unwrap(), before);
}

#[test]
fn checked_generic_flip_uses_staged_nontrivial_fs_and_twist_factors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::with_fs(
        9,
        BraidingStyleKind::Anyonic,
        -1.0,
        1.0,
    ));
    let x_dual = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&x_dual], [&x_dual], |_, _| 3.0).unwrap();

    let codomain = source.flip(&[0], Direction::Forward).unwrap();
    let domain = source.flip(&[1], Direction::Forward).unwrap();
    assert_eq!(codomain.dense_data().unwrap(), &[-3.0]);
    assert_eq!(domain.dense_data().unwrap(), &[-3.0]);
    let roundtrip = codomain.flip(&[0], Direction::Inverse).unwrap();
    assert_eq!(
        roundtrip.dense_data().unwrap(),
        source.dense_data().unwrap()
    );
    assert_eq!(roundtrip.codomain(), source.codomain());
    assert_eq!(roundtrip.domain(), source.domain());
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_identity_case<D>(n: usize, label: Vec<i64>, value: impl Fn(usize) -> D)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let legs = [2, 1, 2].map(|degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), degeneracy)]).unwrap()
    });
    let source: TensorMap<_, D> = TensorMap::from_subblock_fn(
        &runtime,
        [&legs[0], &legs[1], &legs[2]],
        [],
        |trees, indices| {
            value(100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>())
        },
    )
    .unwrap();
    let twisted = source.twist(&[0, 1, 2], Direction::Forward).unwrap();
    assert_eq!(
        twisted.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
    assert_eq!(twisted.dense_data().unwrap(), source.dense_data().unwrap());
    assert!(std::ptr::eq(twisted.provider(), provider.as_ref()));
    assert_eq!(twisted.codomain(), source.codomain());
    assert_eq!(twisted.domain(), source.domain());
    assert_eq!(twisted.subblock_count(), source.subblock_count());
    for index in 0..source.subblock_count() {
        let before = source.subblock(index).unwrap();
        let after = twisted.subblock(index).unwrap();
        assert_eq!(after.key(), before.key());
        assert_eq!(after.shape(), before.shape());
        assert_eq!(after.strides(), before.strides());
        assert_eq!(after.offset(), before.offset());
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_twist_identity_shares_exact_provider_and_layout() {
    // What: the checked SU(3)/SU(4) pivotal implementation is θ=1, so all four
    // provider/dtype cases share payload and preserve the admitted layout.
    assert_sun_identity_case(3, vec![2, 2], |value| value as f64);
    assert_sun_identity_case(3, vec![2, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
    assert_sun_identity_case(4, vec![2, 0, 2], |value| value as f64);
    assert_sun_identity_case(4, vec![2, 0, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_flip_case<D>(n: usize, label: Vec<i64>, value: impl Fn(usize) -> D)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let legs = [2, 1, 2].map(|degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), degeneracy)]).unwrap()
    });
    let source: TensorMap<_, D> = TensorMap::from_subblock_fn(
        &runtime,
        [&legs[0], &legs[1], &legs[2]],
        [],
        |trees, indices| {
            value(100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>())
        },
    )
    .unwrap();
    let flipped = source.flip(&[0, 2], Direction::Forward).unwrap();
    assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
    assert!(flipped.codomain()[0].is_dual());
    assert!(flipped.codomain()[2].is_dual());
    assert_eq!(flipped.dense_data().unwrap(), source.dense_data().unwrap());
    let mut saw_vertex_two = false;
    for index in 0..source.subblock_count() {
        let before = source.subblock(index).unwrap();
        let after = flipped.subblock(index).unwrap();
        let before_trees = source.subblock_fusion_trees(index).unwrap();
        let after_trees = flipped.subblock_fusion_trees(index).unwrap();
        saw_vertex_two |= before_trees
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() > 1);
        assert_eq!(after_trees, before_trees);
        assert_eq!(after.shape(), before.shape());
        assert_eq!(after.strides(), before.strides());
        assert_eq!(after.offset(), before.offset());
    }
    assert!(saw_vertex_two);
    let roundtrip = flipped.flip(&[0, 2], Direction::Inverse).unwrap();
    assert_eq!(
        roundtrip.dense_data().unwrap(),
        source.dense_data().unwrap()
    );
    assert_eq!(roundtrip.codomain(), source.codomain());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_flip_preserves_full_key_layout_and_inverse_roundtrip() {
    assert_sun_flip_case(3, vec![2, 2], |value| value as f64);
    assert_sun_flip_case(3, vec![2, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
    assert_sun_flip_case(4, vec![2, 0, 2], |value| value as f64);
    assert_sun_flip_case(4, vec![2, 0, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
}

type TreeKey = (Label, Vec<Label>, Vec<Label>, Vec<usize>);

/// Every block as `(codomain tree, domain tree, shape, row-major values)`.
fn tree_blocks<D>(
    tensor: &TensorMap<CheckedPivotalToy, D>,
) -> Vec<(TreeKey, TreeKey, Vec<usize>, Vec<D>)>
where
    D: TensorScalar,
{
    (0..tensor.subblock_count())
        .map(|index| {
            let block = tensor.subblock(index).unwrap();
            let trees = tensor.subblock_fusion_trees(index).unwrap();
            let codomain = (
                *trees.coupled(),
                trees.codomain_uncoupled().to_vec(),
                trees.codomain_innerlines().to_vec(),
                trees.codomain_vertices().iter().map(|m| m.get()).collect(),
            );
            let domain = (
                *trees.coupled(),
                trees.domain_uncoupled().to_vec(),
                trees.domain_innerlines().to_vec(),
                trees.domain_vertices().iter().map(|m| m.get()).collect(),
            );
            let shape = block.shape().to_vec();
            let len = shape.iter().product::<usize>();
            let values = (0..len)
                .map(|linear| {
                    let (mut remainder, mut position) = (linear, block.offset());
                    for (&extent, &stride) in shape.iter().zip(block.strides()).rev() {
                        position += (remainder % extent) * stride;
                        remainder /= extent;
                    }
                    tensor.dense_data().unwrap()[position]
                })
                .collect();
            (codomain, domain, shape, values)
        })
        .collect()
}

/// Hand oracle for canonical composition in the fusion-tree basis:
/// `C[f1, f2][i, k] = sum_{t, j} A[f1, t][i, j] * B[t, f2][j, k]`, summed over
/// every middle tree `t` (inner lines and vertex labels included) and its
/// degeneracy multi-index `j`. No braid, twist, or sign enters.
fn assert_compose_oracle<D>(
    lhs: &TensorMap<CheckedPivotalToy, D>,
    rhs: &TensorMap<CheckedPivotalToy, D>,
    actual: &TensorMap<CheckedPivotalToy, D>,
) where
    D: TensorScalar + Into<Complex64>,
{
    let (lhs_nout, rhs_nout) = (lhs.codomain_rank(), rhs.codomain_rank());
    let (lhs_blocks, rhs_blocks) = (tree_blocks(lhs), tree_blocks(rhs));
    let actual_blocks = tree_blocks(actual);
    assert!(actual_blocks.len() >= 2);
    let mut nonzero = 0;
    for (f1, f2, shape, values) in &actual_blocks {
        let (rows, cols) = (
            shape[..lhs_nout].iter().product::<usize>(),
            shape[lhs_nout..].iter().product::<usize>(),
        );
        let mut expected = vec![Complex64::new(0.0, 0.0); rows * cols];
        for (a1, t, a_shape, a) in &lhs_blocks {
            if a1 != f1 {
                continue;
            }
            for (t2, b2, b_shape, b) in &rhs_blocks {
                if t2 != t || b2 != f2 {
                    continue;
                }
                let middle = b_shape[..rhs_nout].iter().product::<usize>();
                assert_eq!(middle, a_shape[lhs_nout..].iter().product::<usize>());
                for i in 0..rows {
                    for k in 0..cols {
                        for j in 0..middle {
                            expected[i * cols + k] +=
                                a[i * middle + j].into() * b[j * cols + k].into();
                        }
                    }
                }
            }
        }
        nonzero += usize::from(expected.iter().any(|value| value.norm() != 0.0));
        for (&got, want) in values.iter().zip(&expected) {
            let got: Complex64 = got.into();
            assert!(
                (got - want).norm() <= 1e-12 * (1.0 + want.norm()),
                "{got} != {want}"
            );
        }
    }
    // An empty middle reaches only the unit coupled sector; every other
    // destination block must stay zero, which the value check above proves.
    assert!(nonzero > 0);
}

fn assert_compose_any_braiding<D>(
    braiding: BraidingStyleKind,
    tag: u8,
    value: impl Fn(usize) -> D + Copy,
) where
    D: TensorScalar + symtn::typed::AdvancedLinalgScalar + Into<Complex64>,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(tag, braiding, -1.0));
    let v = GradedSpace::try_new(Arc::clone(&provider), [(Label::Unit, 1), (Label::X, 2)]).unwrap();
    let w = GradedSpace::try_new(Arc::clone(&provider), [(Label::Unit, 2), (Label::X, 3)]).unwrap();
    let filled = |offset: usize,
                  codomain: &[&GradedSpace<CheckedPivotalToy>],
                  domain: &[&GradedSpace<CheckedPivotalToy>]| {
        TensorMap::from_subblock_fn(
            &runtime,
            codomain.to_vec(),
            domain.to_vec(),
            |trees, indices| {
                value(
                    offset
                        + tree_marker(trees)
                        + indices
                            .iter()
                            .enumerate()
                            .map(|(axis, index)| (axis + 1) * index)
                            .sum::<usize>(),
                )
            },
        )
        .unwrap()
    };
    // A: V ⊗ W ← V and B: V ← V ⊗ W, so both A∘B (middle V) and B∘A
    // (middle V ⊗ W, with the μ = 1, 2 vertices of X ⊗ X → X) are compositions.
    let a = filled(0, &[&v, &w], &[&v]);
    let b = filled(3, &[&v], &[&v, &w]);
    // Dual legs in both middles: A': V ⊗ W* ← V* and B': V* ← V ⊗ W*.
    let (v_dual, w_dual) = (v.try_dual().unwrap(), w.try_dual().unwrap());
    let a_dual = filled(5, &[&v, &w_dual], &[&v_dual]);
    let b_dual = filled(7, &[&v_dual], &[&v, &w_dual]);
    // Empty middle (outer product through the unit): V ← () and () ← W.
    let column = filled(11, &[&v], &[]);
    let row = filled(13, &[], &[&w]);
    let composed = |lhs: &TensorMap<CheckedPivotalToy, D>,
                    rhs: &TensorMap<CheckedPivotalToy, D>| {
        let composed = lhs
            .compose(rhs)
            .unwrap_or_else(|error| panic!("{braiding:?}: {error:?}"));
        assert!(std::ptr::eq(composed.provider(), provider.as_ref()));
        assert_eq!(composed.codomain(), lhs.codomain());
        assert_eq!(composed.domain(), rhs.domain());
        assert_compose_oracle(lhs, rhs, &composed);
        composed
    };
    for (lhs, rhs) in [
        (&a, &b),
        (&b, &a),
        (&a_dual, &b_dual),
        (&b_dual, &a_dual),
        (&column, &row),
    ] {
        composed(lhs, rhs);
    }
    // One self-composition of the endomorphism A∘B.
    let endomorphism = composed(&a, &b);
    let squared = endomorphism.compose(&endomorphism).unwrap();
    assert_compose_oracle(&endomorphism, &endomorphism, &squared);
    // Composition is not a contraction over mismatched spaces; a lazy
    // adjoint operand composes as its materialization does (TensorKit
    // `mul!` over `block(a, c)'`).
    assert!(matches!(
        a.compose(&a),
        Err(GenericTensorError::Plan(
            CheckedGenericPlanError::Operation(OperationError::ContractAxisCountMismatch {
                lhs: 1,
                ..
            })
        ))
    ));
    let adjoint = a.adjoint().unwrap();
    let got = adjoint.compose(&a).unwrap();
    assert_compose_oracle(&adjoint.materialize().unwrap(), &a, &got);
}

#[test]
fn checked_generic_compose_admits_every_braiding_like_tensorkit_mul() {
    // What (#1378): canonical composition glues lhs.domain to rhs.codomain
    // without crossing legs, so it is blockwise matrix multiplication for
    // every braiding style (TensorKit `mul!`), with outer degeneracies,
    // several coupled sectors, multiplicity-two vertices, real and complex
    // payloads. Bosonic is the unchanged control.
    for (tag, braiding) in [
        (30, BraidingStyleKind::NoBraiding),
        (31, BraidingStyleKind::Anyonic),
        (32, BraidingStyleKind::Fermionic),
        (33, BraidingStyleKind::Bosonic),
    ] {
        assert_compose_any_braiding(braiding, tag, |marker| marker as f64 / 1000.0);
        assert_compose_any_braiding(braiding, tag, |marker| {
            Complex64::new(marker as f64 / 1000.0, -(marker as f64) / 7000.0)
        });
    }
}

#[test]
fn checked_generic_contract_keeps_its_braiding_boundaries() {
    // What (#1372, #1378): general `contract` shares the ordinary contraction's
    // symmetric-braiding boundary, canonical axes included, and the checked
    // Generic engine keeps Fermionic general contraction explicitly
    // unsupported (it needs the supertrace twist). `compose` on the same
    // operands succeeds for every style; bosonic contracts to the hand oracle.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (tag, braiding) in [
        (20, BraidingStyleKind::NoBraiding),
        (21, BraidingStyleKind::Anyonic),
        (22, BraidingStyleKind::Bosonic),
        (23, BraidingStyleKind::Fermionic),
    ] {
        let provider = Arc::new(CheckedPivotalToy::new(tag, braiding, 1.0));
        let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Unit, 2)]).unwrap();
        let matrix = |offset: f64| -> TensorMap<_, f64> {
            TensorMap::from_subblock_fn(&runtime, [&unit], [&unit], |_, index| {
                offset + (2 * index[0] + index[1]) as f64
            })
            .unwrap()
        };
        let (lhs, rhs) = (matrix(1.0), matrix(5.0));
        // Hand oracle: the one block is [[1,2],[3,4]]·[[5,6],[7,8]].
        let expected = TensorMap::from_subblock_fn(&runtime, [&unit], [&unit], |_, index| {
            [[19.0, 22.0], [43.0, 50.0]][index[0]][index[1]]
        })
        .unwrap();
        assert_eq!(
            lhs.compose(&rhs).unwrap().dense_data().unwrap(),
            expected.dense_data().unwrap()
        );
        let contract = lhs.contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        );
        let expected_message = match braiding {
            BraidingStyleKind::Bosonic => {
                assert_eq!(
                    contract.unwrap().dense_data().unwrap(),
                    expected.dense_data().unwrap()
                );
                continue;
            }
            BraidingStyleKind::Fermionic => "checked Generic contraction requires Bosonic braiding",
            _ => symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED,
        };
        let operation = match &contract {
            Err(GenericTensorError::Facade(Error::Operation(operation))) => &**operation,
            Err(GenericTensorError::Plan(CheckedGenericPlanError::Operation(operation))) => {
                operation
            }
            other => panic!("{braiding:?}: {:?}", other.as_ref().err()),
        };
        assert!(
            matches!(
                operation,
                symtn::typed::OperationError::UnsupportedTensorContractScope { message }
                    if *message == expected_message
            ),
            "{braiding:?}: {operation:?}"
        );
    }
}

#[test]
fn checked_generic_planar_contract_is_bosonic_only() {
    // What (#2284): a Bosonic checked provider's planar contraction is its
    // ordinary contraction; every other style is the typed
    // `UnsupportedTensorContractScope` (the checked engine has no planar
    // route), eager and `_into`, with the destination untouched.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    for (tag, braiding) in [
        (40, BraidingStyleKind::NoBraiding),
        (41, BraidingStyleKind::Anyonic),
        (42, BraidingStyleKind::Fermionic),
        (43, BraidingStyleKind::Bosonic),
    ] {
        let provider = Arc::new(CheckedPivotalToy::new(tag, braiding, 1.0));
        let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Unit, 2)]).unwrap();
        let matrix = |offset: f64| -> TensorMap<_, f64> {
            TensorMap::from_subblock_fn(&runtime, [&unit], [&unit], |_, index| {
                offset + (2 * index[0] + index[1]) as f64
            })
            .unwrap()
        };
        let (lhs, rhs) = (matrix(1.0), matrix(5.0));
        let mut destination = matrix(-3.0);
        let before = destination.dense_data().unwrap().to_vec();
        let eager = lhs.planar_contract(&rhs, &spec);
        let into = lhs.planar_contract_into(&rhs, &spec, &mut destination, 1.0, 1.0);
        if braiding == BraidingStyleKind::Bosonic {
            assert_eq!(
                eager.unwrap().dense_data().unwrap(),
                lhs.contract(&rhs, &spec).unwrap().dense_data().unwrap()
            );
            into.unwrap();
            continue;
        }
        for result in [eager.map(drop), into] {
            let operation = match &result {
                Err(GenericTensorError::Facade(Error::Operation(operation))) => &**operation,
                Err(GenericTensorError::Plan(CheckedGenericPlanError::Operation(operation))) => {
                    operation
                }
                other => panic!("{braiding:?}: {:?}", other.as_ref().err()),
            };
            assert!(
                matches!(
                    operation,
                    symtn::typed::OperationError::UnsupportedTensorContractScope { .. }
                ),
                "{braiding:?}: {operation:?}"
            );
        }
        let after = destination.dense_data().unwrap();
        assert!(
            before
                .iter()
                .zip(after.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{braiding:?}: destination written"
        );
    }
}

#[test]
fn factorization_leg_roles_keep_the_permute_braiding_boundary() {
    // What (#1553, #1620): factorization and solve leg roles are a `permute`, so a split
    // other than the current one needs a symmetric braiding. The anyonic
    // provider refuses it with the permute's typed error before any
    // factorization work, while its current split, which needs no braid,
    // factorizes.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(40, BraidingStyleKind::Anyonic, 1.0));
    let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Unit, 2)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::Unit, 3)]).unwrap();
    let t: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&unit], [&wide], |_, index| {
            1.0 + (3 * index[0] + index[1]) as f64
        })
        .unwrap();
    assert!(t.svd_compact(&[0], &[1]).is_ok());
    let anyonic_permute = |error: &GenericTensorError<_>| {
        matches!(
            cause::<OperationError>(error),
            Some(OperationError::UnsupportedBraidingStyle {
                style: BraidingStyleKind::Anyonic,
                ..
            })
        )
    };
    for error in [
        t.svd_compact(&[1], &[0]).err().unwrap(),
        t.qr_compact(&[1], &[0]).err().unwrap(),
        t.left_null(&[1], &[0]).err().unwrap(),
    ] {
        assert!(anyonic_permute(&error), "{error:?}");
    }

    let square: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&unit], [&unit]).unwrap();
    assert!(square.solve(&[0], &[1], &square, &[0], &[1]).is_ok());
    for error in [
        square.solve(&[1], &[0], &square, &[0], &[1]).err().unwrap(),
        square.solve(&[0], &[1], &square, &[1], &[0]).err().unwrap(),
    ] {
        assert!(anyonic_permute(&error), "{error:?}");
    }
}

#[test]
fn checked_generic_trace_gates_non_symmetric_braiding_before_the_pair_list() {
    // What: as TensorKit `trace_permute!`, the braiding gate comes before any
    // index handling, so a non-symmetric provider's trace fails even with an
    // empty or malformed pair list (#1872).
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for braiding in [
        BraidingStyleKind::NoBraiding,
        BraidingStyleKind::Anyonic,
        BraidingStyleKind::Bosonic,
        BraidingStyleKind::Fermionic,
    ] {
        let provider = Arc::new(CheckedPivotalToy::new(9, braiding, -1.0));
        let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
        let square = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&x], [&x], |_, indices| {
            indices.iter().sum::<usize>() as f64
        })
        .unwrap();
        for pairs in [vec![], vec![(0, 1)], vec![(0, 0)], vec![(0, 7)]] {
            let result = square.trace_pairs(&pairs).map(|_| ());
            if braiding.is_symmetric() {
                let malformed = pairs.iter().any(|&(lhs, rhs)| lhs == rhs || rhs > 1);
                assert_eq!(
                    result.is_ok(),
                    !malformed,
                    "{braiding:?} {pairs:?}: {result:?}"
                );
            } else {
                let error = result.expect_err("a non-symmetric trace is rejected");
                assert!(
                    matches!(
                        cause::<OperationError>(&error),
                        Some(OperationError::UnsupportedTensorContractScope {
                            message: "fusion tensortrace requires symmetric braiding"
                        })
                    ),
                    "{braiding:?} {pairs:?}: {error:?}"
                );
            }
        }
    }
}

#[test]
fn checked_trace_pivotal_failure_does_not_publish_destination() {
    const ISOLATED: &str = "SYMTN_CHECKED_TRACE_PUBLICATION_ISOLATED";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "checked_trace_pivotal_failure_does_not_publish_destination",
                "--nocapture",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated trace publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(51, BraidingStyleKind::Bosonic, 1.0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source = TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&x], [&x], |_, indices| {
        Complex64::new((indices[0] + 2 * indices[1]) as f64, 0.5)
    })
    .unwrap();
    let payload = source.dense_data().unwrap().as_ptr();
    symtn_core::clear_structure_caches();
    let owner = || {
        let info =
            symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure);
        (info.admissions(), info.entries(), info.charged_bytes())
    };
    let before = owner();
    assert_eq!(before, (0, 0, 0));
    provider.reset_ledger(0);
    provider.fail_twist_on.store(1, Ordering::Relaxed);

    let result = source.trace_pairs(&[(0, 1)]);
    assert!(matches!(
        result,
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::Twist
        )))
    ));
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 1);
    assert_eq!(source.dense_data().unwrap().as_ptr(), payload);
    // The complete-layout owner admits nothing for the rejected destination.
    assert_eq!(owner(), before);
}
