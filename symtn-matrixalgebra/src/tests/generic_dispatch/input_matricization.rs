use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValuesOperation {
    Svd,
    Eigh,
    Eig,
}

impl ValuesOperation {
    fn op(self) -> Op {
        match self {
            ValuesOperation::Svd => Op::SvdVals,
            ValuesOperation::Eigh => Op::EighVals,
            ValuesOperation::Eig => Op::EigVals,
        }
    }
}

const VALUES_OPS: [Op; 3] = [Op::SvdVals, Op::EighVals, Op::EigVals];
const NOT_VALUES: &str = "test only exercises values-only operations";

/// Values-only entries only.
fn script_values_only(script: &mut Script) {
    script
        .set_all(
            &[Op::Svd, Op::Qr, Op::Eigh, Op::DotGeneral],
            Action::Panic(NOT_VALUES),
        )
        .set_all(&VALUES_OPS, Action::Forward);
}

/// `operation`'s values-only entry only; its second call fails.
fn fail_second_values(operation: ValuesOperation) -> ScriptedExecutor {
    let mut dense = ScriptedExecutor::new(());
    script_values_only(&mut dense.script);
    for op in VALUES_OPS {
        if op != operation.op() {
            dense
                .script
                .set(op, Action::Panic("unexpected values-only operation"));
        }
    }
    let op: &'static [Op] = match operation {
        ValuesOperation::Svd => &[Op::SvdVals],
        ValuesOperation::Eigh => &[Op::EighVals],
        ValuesOperation::Eig => &[Op::EigVals],
    };
    dense
        .script
        .fail(op, Some(2), "values", "injected second-sector failure");
    dense
}

fn observed_input(input: &DenseRead<'_>, what: &str) -> (usize, Vec<usize>, usize, Vec<Complex64>) {
    match input {
        DenseRead::F64(view) => (
            view.data().as_ptr() as usize,
            view.strides().to_vec(),
            view.offset(),
            view.data()
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
        ),
        DenseRead::C64(view) => (
            view.data().as_ptr() as usize,
            view.strides().to_vec(),
            view.offset(),
            view.data().to_vec(),
        ),
        _ => panic!("checked Generic {what} fixture must be f64 or c64"),
    }
}

#[derive(Debug)]
struct ValuesInputObservation {
    operation: ValuesOperation,
    pointer: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    values: Vec<Complex64>,
}

#[derive(Default)]
struct ValuesInputSpy {
    observations: Vec<ValuesInputObservation>,
}

impl Observer for ValuesInputSpy {
    fn script(script: &mut Script) {
        script_values_only(script);
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        let Call::Read(op, input) = call else {
            return None;
        };
        let operation = match op {
            Op::SvdVals => ValuesOperation::Svd,
            Op::EighVals => ValuesOperation::Eigh,
            Op::EigVals => ValuesOperation::Eig,
            _ => return None,
        };
        let (pointer, strides, offset, values) = observed_input(input, "values");
        self.observations.push(ValuesInputObservation {
            operation,
            pointer,
            shape: input.shape().to_vec(),
            strides,
            offset,
            values,
        });
        None
    }
}

#[derive(Debug)]
struct CompactInputObservation {
    pointer: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    values: Vec<Complex64>,
}

struct CompactInputSpy {
    operation: crate::probe::CheckedCompactOperation,
    observations: Vec<CompactInputObservation>,
}

const NOT_COMPACT: &str = "test only exercises compact QR/SVD/LQ";

impl Observer for CompactInputSpy {
    fn script(script: &mut Script) {
        script
            .set_all(&[Op::Svd, Op::Qr], Action::Forward)
            .set(
                Op::SvdInto,
                Action::Panic("checked compact SVD must use the owned API"),
            )
            .set(
                Op::QrInto,
                Action::Panic("compact QR/LQ must not use qr_into"),
            )
            .set_all(&[Op::Eigh, Op::DotGeneral], Action::Panic(NOT_COMPACT));
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        use crate::probe::CheckedCompactOperation;
        let Call::Read(op @ (Op::Svd | Op::Qr), input) = call else {
            return None;
        };
        if op == Op::Svd {
            assert_eq!(self.operation, CheckedCompactOperation::Svd);
        } else {
            assert!(matches!(
                self.operation,
                CheckedCompactOperation::Qr | CheckedCompactOperation::Lq
            ));
        }
        let (pointer, strides, offset, values) = observed_input(input, "compact");
        self.observations.push(CompactInputObservation {
            pointer,
            shape: input.shape().to_vec(),
            strides,
            offset,
            values,
        });
        None
    }
}

impl CompactInputSpy {
    fn new(operation: crate::probe::CheckedCompactOperation) -> ScriptedExecutor<Self> {
        ScriptedExecutor::new(Self {
            operation,
            observations: Vec::new(),
        })
    }
}

fn assert_borrowed_values_inputs<D: FactorScalar>(
    observations: &[ValuesInputObservation],
    operation: ValuesOperation,
    data: &[D],
    regions: &[symtn_core::CoupledSectorRegion],
) {
    assert_eq!(observations.len(), regions.len());
    for (observation, region) in observations.iter().zip(regions) {
        assert_eq!(observation.operation, operation);
        assert_eq!(observation.shape, [region.rows(), region.cols()]);
        assert_eq!(observation.strides, [1, region.rows()]);
        assert_eq!(observation.offset, 0);
        assert_eq!(
            observation.pointer,
            data.as_ptr() as usize + region.range().start * std::mem::size_of::<D>()
        );
        let expected = data[region.range()]
            .iter()
            .map(|&value| value.widen_complex())
            .collect::<Vec<_>>();
        assert_eq!(observation.values, expected);
    }
}

fn assert_compact_input_observations<D: FactorScalar>(
    recording: &crate::probe::Recording,
    operation: crate::probe::CheckedCompactOperation,
    dense: &[CompactInputObservation],
    data: &[D],
    regions: &[symtn_core::CoupledSectorRegion],
) {
    let lowering = recording.checked_compact_inputs();
    assert_eq!(lowering.len(), regions.len());
    assert_eq!(dense.len(), regions.len());
    for ((lowering, dense), region) in lowering.iter().zip(dense).zip(regions) {
        let source_pointer =
            data.as_ptr() as usize + region.range().start * std::mem::size_of::<D>();
        assert_eq!(lowering.operation, operation);
        assert_eq!(lowering.input_pointer, data.as_ptr() as usize);
        assert_eq!(lowering.matrix_pointer, source_pointer);
        assert_eq!(lowering.elements, region.range().len());
        assert_eq!(dense.offset, 0);
        match operation {
            crate::probe::CheckedCompactOperation::Qr
            | crate::probe::CheckedCompactOperation::Svd => {
                assert_eq!(lowering.adjoint_pointer, None);
                assert_eq!(dense.pointer, source_pointer);
                assert_eq!(dense.shape, [region.rows(), region.cols()]);
                assert_eq!(dense.strides, [1, region.rows()]);
                assert_eq!(
                    dense.values,
                    data[region.range()]
                        .iter()
                        .map(|&value| value.widen_complex())
                        .collect::<Vec<_>>()
                );
            }
            crate::probe::CheckedCompactOperation::Lq => {
                assert_eq!(lowering.adjoint_pointer, Some(dense.pointer));
                assert_ne!(dense.pointer, source_pointer);
                assert_eq!(dense.shape, [region.cols(), region.rows()]);
                assert_eq!(dense.strides, [1, region.cols()]);
                let source = &data[region.range()];
                let expected = (0..region.rows())
                    .flat_map(|column| {
                        (0..region.cols()).map(move |row| {
                            source[column + region.rows() * row].widen_complex().conj()
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(dense.values, expected);
            }
        }
    }
}

fn assert_checked_compact_input_borrowing<D: FactorScalar>(
    space: BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: Vec<D>,
) {
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = data.clone();

    let compact_qr_copy_rec = crate::probe::record();
    let checked_inputs_rec = crate::probe::record();
    let mut qr = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Qr);
    let factors = qr_compact_dyn_checked_generic(&mut qr, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.q, None, &factors.r);
    assert_compact_input_observations(
        &checked_inputs_rec,
        crate::probe::CheckedCompactOperation::Qr,
        &qr.observations,
        &data,
        &regions,
    );
    assert_eq!(compact_qr_copy_rec.compact_qr_copy().input_pack_calls, 0);

    let compact_svd_copy_rec = crate::probe::record();
    let checked_inputs_rec = crate::probe::record();
    let mut svd = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Svd);
    let factors = svd_compact_dyn_checked_generic(&mut svd, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.u, Some(&factors.s), &factors.vh);
    assert_compact_input_observations(
        &checked_inputs_rec,
        crate::probe::CheckedCompactOperation::Svd,
        &svd.observations,
        &data,
        &regions,
    );
    assert_eq!(compact_svd_copy_rec.compact_svd_copy().input_pack_calls, 0);

    let compact_lq_copy_rec = crate::probe::record();
    let checked_inputs_rec = crate::probe::record();
    let mut lq = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Lq);
    let factors = lq_compact_dyn_checked_generic(&mut lq, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.l, None, &factors.q);
    assert_compact_input_observations(
        &checked_inputs_rec,
        crate::probe::CheckedCompactOperation::Lq,
        &lq.observations,
        &data,
        &regions,
    );
    let probe = compact_lq_copy_rec.compact_lq_copy();
    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.adjoint_scratch_fill_calls, regions.len());
    assert_eq!(
        probe.adjoint_scratch_fill_bytes,
        data.len() * std::mem::size_of::<D>()
    );
    assert!(data == before);
}

#[test]
fn checked_generic_compact_factors_borrow_real_and_complex_canonical_inputs() {
    let (_, real_space, real_data) = checked_svd_truncation_input::<f64>(false);
    assert_checked_compact_input_borrowing(real_space, real_data);
    let (_, complex_space, complex_data) = checked_svd_truncation_input::<Complex64>(true);
    assert_checked_compact_input_borrowing(complex_space, complex_data);
    let (_, wide_space, wide_data) = checked_svd_wide_input::<Complex64>();
    assert_checked_compact_input_borrowing(wide_space, wide_data);
}

#[test]
fn checked_generic_compact_geometry_preserves_complete_multi_tree_identity() {
    let (source, _, data) = generic_values_endomorphism_input();
    let (_, space) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    let multiplicity_region = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!(multiplicity_region.row_trees().len(), 2);
    let first = multiplicity_region.row_trees()[0].tree();
    let second = multiplicity_region.row_trees()[1].tree();
    assert_eq!(first.uncoupled(), second.uncoupled());
    assert_eq!(first.is_dual(), second.is_dual());
    assert_eq!(first.innerlines(), second.innerlines());
    assert_ne!(first.vertices(), second.vertices());

    let checked_inputs_rec = crate::probe::record();
    let mut qr = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Qr);
    let qr_factors = qr_compact_dyn_checked_generic(&mut qr, &input).unwrap();
    assert_compact_input_observations(
        &checked_inputs_rec,
        crate::probe::CheckedCompactOperation::Qr,
        &qr.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(&input, &qr_factors.q, None, &qr_factors.r);

    let checked_inputs_rec = crate::probe::record();
    let mut svd = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Svd);
    let svd_factors = svd_compact_dyn_checked_generic(&mut svd, &input).unwrap();
    assert_compact_input_observations(
        &checked_inputs_rec,
        crate::probe::CheckedCompactOperation::Svd,
        &svd.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(
        &input,
        &svd_factors.u,
        Some(&svd_factors.s),
        &svd_factors.vh,
    );

    let checked_inputs_rec = crate::probe::record();
    let mut lq = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Lq);
    let lq_factors = lq_compact_dyn_checked_generic(&mut lq, &input).unwrap();
    assert_compact_input_observations(
        &checked_inputs_rec,
        crate::probe::CheckedCompactOperation::Lq,
        &lq.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(&input, &lq_factors.l, None, &lq_factors.q);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_factors_keep_padded_reordered_input_pack() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (expert_space, expert_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let expert_space = BoundDynamicFusionMapSpace::bind_generic(
        expert_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let expert = BoundDynamicTensorRef::try_new(&expert_space, &expert_data).unwrap();
    let canonical_before = canonical_data.clone();
    let expert_before = expert_data.clone();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_qr_copy_rec = crate::probe::record();
    let actual_qr = qr_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(&expert, &actual_qr.q, None, &actual_qr.r);
    assert!(compact_qr_copy_rec.compact_qr_copy().input_pack_calls > 0);

    let compact_svd_copy_rec = crate::probe::record();
    let actual_svd = svd_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(
        &expert,
        &actual_svd.u,
        Some(&actual_svd.s),
        &actual_svd.vh,
    );
    assert!(compact_svd_copy_rec.compact_svd_copy().input_pack_calls > 0);

    let compact_lq_copy_rec = crate::probe::record();
    let checked_inputs_rec = crate::probe::record();
    let actual_lq = lq_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(&expert, &actual_lq.l, None, &actual_lq.q);
    let lq_probe = compact_lq_copy_rec.compact_lq_copy();
    assert!(lq_probe.input_pack_calls > 0);
    assert!(lq_probe.adjoint_scratch_fill_calls > 0);

    let start = expert_data.as_ptr() as usize;
    let end = start + expert_data.len() * std::mem::size_of::<f64>();
    for observation in checked_inputs_rec.checked_compact_inputs() {
        assert_eq!(observation.input_pointer, start);
        assert!(observation.matrix_pointer < start || observation.matrix_pointer >= end);
    }
    assert!(canonical_data == canonical_before);
    assert!(expert_data == expert_before);
    assert!(Arc::ptr_eq(actual_qr.q.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(actual_svd.s.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(actual_lq.q.space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_interleaved_fallback_keeps_literal_matrix_order() {
    let (canonical, _, canonical_data) = generic_values_endomorphism_input();
    let (expert_space, expert_data) =
        interleaved_generic_endomorphism_input(&canonical, &canonical_data);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let expert_space = BoundDynamicFusionMapSpace::bind_generic(
        expert_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let expert = BoundDynamicTensorRef::try_new(&expert_space, &expert_data).unwrap();
    let before = expert_data.clone();
    let direct = [
        vec![
            Complex64::new(3.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0),
        ],
        vec![Complex64::new(2.0, -1.0)],
    ];
    let adjoint = [
        vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ],
        vec![Complex64::new(2.0, 1.0)],
    ];

    let compact_qr_copy_rec = crate::probe::record();
    let mut qr = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Qr);
    let factors = qr_compact_dyn_checked_generic(&mut qr, &expert).unwrap();
    assert_eq!(
        qr.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        direct
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.q, None, &factors.r);
    assert!(compact_qr_copy_rec.compact_qr_copy().input_pack_calls > 0);

    let compact_svd_copy_rec = crate::probe::record();
    let mut svd = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Svd);
    let factors = svd_compact_dyn_checked_generic(&mut svd, &expert).unwrap();
    assert_eq!(
        svd.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        direct
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.u, Some(&factors.s), &factors.vh);
    assert!(compact_svd_copy_rec.compact_svd_copy().input_pack_calls > 0);

    let compact_lq_copy_rec = crate::probe::record();
    let mut lq = CompactInputSpy::new(crate::probe::CheckedCompactOperation::Lq);
    let factors = lq_compact_dyn_checked_generic(&mut lq, &expert).unwrap();
    assert_eq!(
        lq.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        adjoint
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.l, None, &factors.q);
    let probe = compact_lq_copy_rec.compact_lq_copy();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.adjoint_scratch_fill_calls > 0);
    assert_eq!(expert_data, before);
}

fn assert_real_spectra_by_sector_close(actual: &[SectorSpectrum], expected: &[SectorSpectrum]) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.sector == expected.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}

fn assert_complex_spectra_by_sector_close(
    actual: &[SectorSpectrum<Complex64>],
    expected: &[SectorSpectrum<Complex64>],
) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.sector == expected.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).norm() < 1.0e-10);
        }
    }
}

#[test]
fn checked_only_generic_values_borrow_canonical_input_regions() {
    let (_, rectangular_space, rectangular_data) = checked_svd_truncation_input::<f64>(false);
    let (rectangular_provider, rectangular_space) = bind_checked_only(&rectangular_space);
    let rectangular_calls = rectangular_provider.calls.get();
    let rectangular_before = rectangular_data.clone();
    let rectangular_regions = rectangular_space
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let mut spy = ScriptedExecutor::<ValuesInputSpy>::default();
    let values_fallbacks_rec = crate::probe::record();
    let spectra = svd_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&rectangular_space, &rectangular_data).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(
        &spy.observations,
        ValuesOperation::Svd,
        &rectangular_data,
        &rectangular_regions,
    );
    assert_real_spectra_by_sector_close(
        &spectra,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![4.0, 1.0],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![3.0, 2.0],
            },
        ],
    );
    assert_eq!(rectangular_provider.calls.get(), rectangular_calls);
    assert_eq!(rectangular_data, rectangular_before);

    let (endomorphism_space, hermitian, general) = generic_values_endomorphism_input();
    let (endomorphism_provider, endomorphism_space) = bind_checked_only(&endomorphism_space);
    let provider_calls = endomorphism_provider.calls.get();
    let regions = endomorphism_space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let multiplicity_region = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!(multiplicity_region.row_trees().len(), 2);
    assert_eq!(multiplicity_region.col_trees().len(), 2);
    let hermitian_before = hermitian.clone();
    let general_before = general.clone();
    spy.observations.clear();
    let eigh = eigh_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&endomorphism_space, &hermitian).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(
        &spy.observations,
        ValuesOperation::Eigh,
        &hermitian,
        &regions,
    );
    assert_real_spectra_by_sector_close(
        &eigh,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![-4.0],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![1.0, 3.0],
            },
        ],
    );
    spy.observations.clear();
    let eig = eig_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&endomorphism_space, &general).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(&spy.observations, ValuesOperation::Eig, &general, &regions);
    assert_complex_spectra_by_sector_close(
        &eig,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![Complex64::new(2.0, -1.0)],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![Complex64::new(1.0, 1.0), Complex64::new(3.0, -1.0)],
            },
        ],
    );
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        0
    );
    assert_eq!(endomorphism_provider.calls.get(), provider_calls);
    assert_eq!(hermitian, hermitian_before);
    assert_eq!(general, general_before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_values_keep_padded_reordered_fallback() {
    let (canonical_space, hermitian, general) = generic_values_endomorphism_input();
    let (padded_space, padded_hermitian) =
        padded_reordered_generic_endomorphism_input(&canonical_space, &hermitian);
    let (_, padded_general) =
        padded_reordered_generic_endomorphism_input(&canonical_space, &general);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let canonical_space = BoundDynamicFusionMapSpace::bind_generic(
        canonical_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let padded_space = BoundDynamicFusionMapSpace::bind_generic(
        padded_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    assert!(padded_space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let provider_calls = provider.calls.get();
    let canonical_general = BoundDynamicTensorRef::try_new(&canonical_space, &general).unwrap();
    let canonical_hermitian = BoundDynamicTensorRef::try_new(&canonical_space, &hermitian).unwrap();
    let padded_general_input =
        BoundDynamicTensorRef::try_new(&padded_space, &padded_general).unwrap();
    let padded_hermitian_input =
        BoundDynamicTensorRef::try_new(&padded_space, &padded_hermitian).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let expected_svd = svd_vals_dyn_checked_generic(&mut dense, &canonical_general).unwrap();
    let expected_eigh = eigh_vals_dyn_checked_generic(&mut dense, &canonical_hermitian).unwrap();
    let expected_eig = eig_vals_dyn_checked_generic(&mut dense, &canonical_general).unwrap();

    let general_before = padded_general.clone();
    let hermitian_before = padded_hermitian.clone();
    let mut spy = ScriptedExecutor::<ValuesInputSpy>::default();
    let values_fallbacks_rec = crate::probe::record();
    let actual_svd = svd_vals_dyn_checked_generic(&mut spy, &padded_general_input).unwrap();
    let actual_eigh = eigh_vals_dyn_checked_generic(&mut spy, &padded_hermitian_input).unwrap();
    let actual_eig = eig_vals_dyn_checked_generic(&mut spy, &padded_general_input).unwrap();
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        3
    );
    assert_real_spectra_by_sector_close(&actual_svd, &expected_svd);
    assert_real_spectra_by_sector_close(&actual_eigh, &expected_eigh);
    assert_complex_spectra_by_sector_close(&actual_eig, &expected_eig);
    assert_eq!(provider.calls.get(), provider_calls);
    assert_eq!(padded_general, general_before);
    assert_eq!(padded_hermitian, hermitian_before);

    let general_start = padded_general.as_ptr() as usize;
    let general_end = general_start + std::mem::size_of_val(padded_general.as_slice());
    let hermitian_start = padded_hermitian.as_ptr() as usize;
    let hermitian_end = hermitian_start + std::mem::size_of_val(padded_hermitian.as_slice());
    for observation in &spy.observations {
        let (start, end) = if observation.operation == ValuesOperation::Eigh {
            (hermitian_start, hermitian_end)
        } else {
            (general_start, general_end)
        };
        assert!(observation.pointer < start || observation.pointer >= end);
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_only_generic_values_preserve_empty_scalar_and_shape_boundaries() {
    let x = SectorId::new(1);
    let empty_leg = SectorLeg::new([(x, 0)], false);
    let empty_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([empty_leg.clone()]),
        FusionProductSpace::new([empty_leg]),
    );
    let empty_provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let empty_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&empty_provider),
        empty_homspace,
    )
    .unwrap();
    let empty_calls = empty_provider.calls.get();
    let empty_data: [f64; 0] = [];
    let empty = BoundDynamicTensorRef::try_new(&empty_space, &empty_data).unwrap();
    let mut reject = ScriptedExecutor::new(RejectExecutorCalls);
    assert!(svd_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert!(eigh_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    let empty_eigh =
        eigh_full_dyn_checked_generic(&mut reject, &empty, HermitianTol::DEFAULT).unwrap();
    assert!(empty_eigh.v().data().is_empty());
    assert!(empty_eigh.eigenvalues().is_empty());
    assert!(eig_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert_eq!(empty_provider.calls.get(), empty_calls);

    let scalar_provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let scalar_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&scalar_provider),
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
    )
    .unwrap();
    let scalar_calls = scalar_provider.calls.get();
    let scalar_data = [-3.0];
    let scalar = BoundDynamicTensorRef::try_new(&scalar_space, &scalar_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    assert_eq!(
        svd_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [3.0]
    );
    assert_eq!(
        eigh_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [-3.0]
    );
    assert_eq!(
        eig_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [Complex64::new(-3.0, 0.0)]
    );
    assert_eq!(scalar_provider.calls.get(), scalar_calls);

    let (_, rectangular, data) = checked_svd_truncation_input::<f64>(false);
    let (rectangular_provider, rectangular) = bind_checked_only(&rectangular);
    let rectangular_calls = rectangular_provider.calls.get();
    let rectangular = BoundDynamicTensorRef::try_new(&rectangular, &data).unwrap();
    assert!(matches!(
        eigh_vals_dyn_checked_generic(&mut reject, &rectangular),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::SpaceMismatch { .. }
        ))
    ));
    assert!(matches!(
        eig_vals_dyn_checked_generic(&mut reject, &rectangular),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::SpaceMismatch { .. }
        ))
    ));
    assert_eq!(rectangular_provider.calls.get(), rectangular_calls);
}

#[test]
fn spectrum_only_entry_points_return_the_published_orders() {
    // What: svd_vals descending, eigh_vals ascending, eig_vals ascending
    // lexicographic (re, im) within every sector (#1985).
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();

    let svd = svd_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    assert!(!svd.is_empty());
    for entry in &svd {
        for pair in entry.values.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12);
        }
    }
    let eigh = eigh_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
    assert!(!eigh.is_empty());
    for entry in &eigh {
        for pair in entry.values.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }
    let eig = eig_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    assert!(!eig.is_empty());
    for entry in &eig {
        for pair in entry.values.windows(2) {
            assert!((pair[0].re, pair[0].im) <= (pair[1].re, pair[1].im));
        }
    }
}

#[test]
fn values_only_public_boundaries_distinguish_empty_sectors_from_a_scalar() {
    // What: zero degeneracies remove the sector entirely, while a rank-zero
    // scalar remains one vacuum-sector 1x1 matrix for every values operation.
    let empty = rectangular_svd_tensor(0, 0);
    assert_eq!(empty.structure().block_count(), 0);
    assert!(empty.data().is_empty());
    let mut reject = ScriptedExecutor::new(RejectExecutorCalls);
    let values_fallbacks_rec = crate::probe::record();
    assert!(svd_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert!(eigh_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert!(eig_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        0
    );

    let rule = Z2FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let scalar = TestTensor::<f64>::from_vec_with_fusion_space(vec![-3.0], space).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    let eigh = eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    let eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    assert_eq!(svd[0].sector, rule.vacuum());
    assert_eq!(svd[0].values, vec![3.0]);
    assert_eq!(eigh[0].sector, rule.vacuum());
    assert_eq!(eigh[0].values, vec![-3.0]);
    assert_eq!(eig[0].sector, rule.vacuum());
    assert_eq!(eig[0].values, vec![Complex64::new(-3.0, 0.0)]);
}

#[test]
fn values_only_second_sector_failures_publish_no_partial_spectrum() {
    // What: after one successful sector, each dense values failure returns Err
    // without exposing the accumulated prefix or mutating borrowed input.
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let general = tsvd_test_tensor(&Z2FusionRule, &sectors);
    let hermitian = hermitian_test_tensor(&Z2FusionRule, &sectors);
    for operation in [
        ValuesOperation::Svd,
        ValuesOperation::Eigh,
        ValuesOperation::Eig,
    ] {
        let tensor = if operation == ValuesOperation::Eigh {
            &hermitian
        } else {
            &general
        };
        let input = bound_tensor(Arc::new(Z2FusionRule), tensor);
        let before = input.data().to_vec();
        let mut dense = fail_second_values(operation);
        let result = match operation {
            ValuesOperation::Svd => svd_vals(&mut dense, &input.as_ref()).map(|_| ()),
            ValuesOperation::Eigh => eigh_vals(&mut dense, &input.as_ref()).map(|_| ()),
            ValuesOperation::Eig => eig_vals(&mut dense, &input.as_ref()).map(|_| ()),
        };

        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(dense.counts().of(&VALUES_OPS), 2);
        assert_eq!(input.data(), before);
    }
}

#[test]
fn values_only_ties_follow_the_published_order_on_direct_and_padded_layouts() {
    // What: equal singular values keep the provider's order, the provider's
    // ascending EIGH spectrum (with a ±2 pair) is published unchanged, and a
    // conjugate EIG pair is published in (re, im) order, on both the borrowed
    // and packed sector paths.
    let rule = Z2FusionRule;
    let svd_input =
        one_sector_rectangular_matrix(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let eigh_input =
        one_sector_rectangular_matrix(vec![1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0], 3, 3);
    let eig_input =
        one_sector_rectangular_matrix(vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let svd_padded = padded_copy(&rule, &svd_input);
    let eigh_padded = padded_copy(&rule, &eigh_input);
    let eig_padded = padded_copy(&rule, &eig_input);
    let shape = [3, 3];
    let strides = [1, 3];
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let raw_svd = dense
        .svd_vals(DenseRead::F64(
            symtn_dense::DenseView::new(svd_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_f64_slice()
        .unwrap()
        .to_vec();
    let raw_eigh = dense
        .eigh_vals(DenseRead::F64(
            symtn_dense::DenseView::new(eigh_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_f64_slice()
        .unwrap()
        .to_vec();
    // The provider's spectrum is ascending; the values are a diagonal's, up
    // to the driver's rounding.
    for (got, want) in raw_eigh.iter().zip([-2.0, 1.0, 2.0]) {
        assert!(
            (got - want).abs() <= 64.0 * f64::EPSILON * 2.0,
            "{raw_eigh:?}"
        );
    }
    assert_eq!(raw_eigh.len(), 3);
    let mut raw_eig = dense
        .eig_vals(DenseRead::F64(
            symtn_dense::DenseView::new(eig_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_c64_slice()
        .unwrap()
        .to_vec();
    raw_eig.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());

    let direct_svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &svd_input)).unwrap();
    let padded_svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &svd_padded)).unwrap();
    let direct_eigh =
        eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eigh_input)).unwrap();
    let padded_eigh =
        eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eigh_padded)).unwrap();
    let direct_eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eig_input)).unwrap();
    let padded_eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eig_padded)).unwrap();

    assert_eq!(direct_svd[0].values, raw_svd);
    assert_eq!(direct_eigh[0].values, raw_eigh);
    assert_eq!(direct_eig[0].values, raw_eig);
    assert_real_spectra_close(&direct_svd, &padded_svd);
    assert_real_spectra_close(&direct_eigh, &padded_eigh);
    assert_complex_spectra_close(&direct_eig, &padded_eig);
}

#[test]
fn values_only_entry_points_match_untruncated_decomposition_spectra() {
    // The `_vals` paths call LAPACK `job='N'` (no vectors) and must reproduce
    // the untruncated decomposition's spectrum. This is a numerical-agreement check,
    // not bit-for-bit: LAPACK backends may route the vectors-vs-no-vectors
    // cases through different routines (e.g. `gesdd` divide-and-conquer for the
    // full SVD vs `gesvd` QR for values-only), which differ in the last ULPs.
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();

    let tol = 1e-10;
    let assert_real_close = |vals: &[SectorSpectrum], full: &[SectorSpectrum]| {
        assert_eq!(vals.len(), full.len());
        for (a, b) in vals.iter().zip(full) {
            assert_eq!(a.sector, b.sector);
            assert_eq!(a.values.len(), b.values.len());
            for (x, y) in a.values.iter().zip(&b.values) {
                assert!((x - y).abs() <= tol, "{x} vs {y}");
            }
        }
    };

    let svd_vals_spectra = svd_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    let svd_compact_spectra = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap()
    .singular_values;
    assert_real_close(&svd_vals_spectra, &svd_compact_spectra);

    let eigh_vals_spectra = eigh_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
    let eigh_full_spectra = eigh_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap()
    .eigenvalues;
    assert_real_close(&eigh_vals_spectra, &eigh_full_spectra);

    let eig_vals_spectra = eig_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    let eig_full_spectra = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap()
    .eigenvalues;
    assert_eq!(eig_vals_spectra.len(), eig_full_spectra.len());
    for (a, b) in eig_vals_spectra.iter().zip(&eig_full_spectra) {
        assert_eq!(a.sector, b.sector);
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).norm() <= tol, "{x} vs {y}");
        }
    }
}
