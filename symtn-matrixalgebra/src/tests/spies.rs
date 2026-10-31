pub(super) use super::scripted_executor::{
    Action, Call, Observer, Op, Reply, Script, ScriptedExecutor,
};
use super::*;

// The doubles of this suite, as observers of the one `ScriptedExecutor`:
// `script` sets what each entry does, the hooks record what a test
// inspects. Entries left at `Action::Default` run the trait's default body,
// as the hand-written doubles did for every method they did not override.

const SVD_AND_INTO: &[Op] = &[Op::Svd, Op::SvdInto];
const REQUIRED_EXCEPT_EIGH: &[Op] = &[Op::Svd, Op::Qr, Op::DotGeneral];

fn injected(label: &'static str, message: &'static str) -> Reply {
    Err(DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op: label,
        message: message.to_string(),
    })
}

fn f64_input(input: &DenseRead<'_>) -> Vec<f64> {
    let DenseRead::F64(input) = input else {
        panic!("test input must be f64")
    };
    input.data().to_vec()
}

fn assert_zeroed_f64(write: &DenseWrite<'_>, what: &str) {
    let DenseWrite::F64(view) = write else {
        panic!("test {what} must be f64")
    };
    assert!(view.data().iter().all(|&value| value == 0.0));
}

/// Validation must reject the input before any dense work.
#[derive(Default)]
pub(super) struct RejectExecutorCalls;

impl Observer for RejectExecutorCalls {
    fn script(script: &mut Script) {
        script
            .set(
                Op::Svd,
                Action::Panic("validation must reject the input before SVD execution"),
            )
            .set(
                Op::Qr,
                Action::Panic("validation must reject the input before QR execution"),
            )
            .set(
                Op::Eigh,
                Action::Panic("validation must reject the input before EIGH execution"),
            )
            .set(
                Op::DotGeneral,
                Action::Panic("validation must reject the input before dense execution"),
            );
    }
}

/// Forwards; tests read `counts().svd`.
#[derive(Default)]
pub(super) struct SvdCallSpy;

impl Observer for SvdCallSpy {}

/// Owned compact SVD only: `svd_into` fails; records the owned `U`/`Vh`
/// buffers and every GEMM operand.
#[derive(Default)]
pub(super) struct RejectSvdInto {
    pub(super) output_ptrs: Vec<(usize, usize)>,
    pub(super) gemm_ptrs: Vec<(usize, usize)>,
    pub(super) gemm_views: Vec<(DenseReadView, DenseReadView, bool, bool)>,
}

impl Observer for RejectSvdInto {
    fn script(script: &mut Script) {
        script
            .set_all(&[Op::Svd, Op::Qr, Op::Eigh], Action::Forward)
            .fail(
                &[Op::SvdInto],
                None,
                "svd_into",
                "direct compact SVD must not use svd_into",
            );
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        if let Call::Gemm { lhs, rhs, config } = call {
            self.gemm_ptrs
                .push((dense_read_pointer(lhs), dense_read_pointer(rhs)));
            self.gemm_views.push((
                dense_read_view(lhs),
                dense_read_view(rhs),
                config.lhs_conj(),
                config.rhs_conj(),
            ));
        }
        None
    }

    fn outputs(&mut self, op: Op, outputs: &[DenseTensor]) {
        if op == Op::Svd {
            self.output_ptrs.push((
                dense_tensor_pointer(&outputs[0]),
                dense_tensor_pointer(&outputs[2]),
            ));
        }
    }
}

#[derive(Debug)]
pub(super) struct DenseReadView {
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) offset: usize,
}

/// Owned compact EIGH only: `eigh_into` fails; records the owned vectors.
#[derive(Default)]
pub(super) struct RejectEighInto {
    pub(super) vector_ptrs: Vec<usize>,
}

impl Observer for RejectEighInto {
    fn script(script: &mut Script) {
        script
            .set_all(&[Op::Svd, Op::Qr, Op::Eigh], Action::Forward)
            .fail(
                &[Op::EighInto],
                None,
                "eigh_into",
                "direct compact EIGH must not use eigh_into",
            );
    }

    fn outputs(&mut self, op: Op, outputs: &[DenseTensor]) {
        if op == Op::Eigh {
            self.vector_ptrs.push(dense_tensor_pointer(&outputs[1]));
        }
    }
}

fn dense_tensor_pointer(tensor: &DenseTensor) -> usize {
    if let Ok(data) = tensor.as_f32_slice() {
        return data.as_ptr() as usize;
    }
    if let Ok(data) = tensor.as_f64_slice() {
        return data.as_ptr() as usize;
    }
    if let Ok(data) = tensor.as_c32_slice() {
        return data.as_ptr() as usize;
    }
    if let Ok(data) = tensor.as_c64_slice() {
        return data.as_ptr() as usize;
    }
    panic!("compact SVD fixture must return a supported host dtype")
}

fn dense_read_pointer(read: &DenseRead<'_>) -> usize {
    match read {
        DenseRead::F32(view) => view.data().as_ptr() as usize,
        DenseRead::F64(view) => view.data().as_ptr() as usize,
        DenseRead::I32(view) => view.data().as_ptr() as usize,
        DenseRead::I64(view) => view.data().as_ptr() as usize,
        DenseRead::Bool(view) => view.data().as_ptr() as usize,
        DenseRead::C32(view) => view.data().as_ptr() as usize,
        DenseRead::C64(view) => view.data().as_ptr() as usize,
    }
}

fn dense_write_pointer(write: &DenseWrite<'_>) -> usize {
    match write {
        DenseWrite::F32(view) => view.data().as_ptr() as usize,
        DenseWrite::F64(view) => view.data().as_ptr() as usize,
        DenseWrite::I32(view) => view.data().as_ptr() as usize,
        DenseWrite::I64(view) => view.data().as_ptr() as usize,
        DenseWrite::Bool(view) => view.data().as_ptr() as usize,
        DenseWrite::C32(view) => view.data().as_ptr() as usize,
        DenseWrite::C64(view) => view.data().as_ptr() as usize,
    }
}

fn dense_read_view(read: &DenseRead<'_>) -> DenseReadView {
    let (shape, strides, offset) = match read {
        DenseRead::F32(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::F64(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::I32(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::I64(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::Bool(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::C32(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::C64(view) => (view.shape(), view.strides(), view.offset()),
    };
    DenseReadView {
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        offset,
    }
}

/// An inverse: no SVD, no recomposition GEMM. Records solve destinations.
#[derive(Default)]
pub(super) struct SolveCallSpy {
    pub(super) destination_ptrs: Vec<usize>,
}

fn script_inverse(script: &mut Script) {
    script
        .set(Op::Svd, Action::Panic("inverse must not execute an SVD"))
        .set(
            Op::DotGeneral,
            Action::Panic("inverse must not recompose factors"),
        )
        .set_all(&[Op::Qr, Op::Eigh, Op::Solve], Action::Forward);
}

impl Observer for SolveCallSpy {
    fn script(script: &mut Script) {
        script_inverse(script);
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        if let Call::Solve { x, .. } = call {
            self.destination_ptrs.push(dense_write_pointer(x));
        }
        None
    }
}

/// [`SolveCallSpy`]'s inverse whose second solve fails.
#[derive(Default)]
pub(super) struct FailSecondSolve;

impl Observer for FailSecondSolve {
    fn script(script: &mut Script) {
        script_inverse(script);
        script.fail(
            &[Op::Solve],
            Some(2),
            "solve_into",
            "injected second-sector failure",
        );
    }
}

/// SVD whose second call (owned or destination) fails; no EIGH or GEMM.
#[derive(Default)]
pub(super) struct FailSecondSvd;

impl Observer for FailSecondSvd {
    fn script(script: &mut Script) {
        script
            .set_all(&[Op::Svd, Op::SvdInto, Op::Qr], Action::Forward)
            .set(Op::Eigh, Action::Panic("test only exercises SVD"))
            .set(Op::DotGeneral, Action::Panic("test only exercises SVD"))
            .fail(
                SVD_AND_INTO,
                Some(2),
                "svd_into",
                "injected second-sector failure",
            );
    }
}

/// Records each SVD input and fails it, unless `outputs` holds a reply.
#[derive(Default)]
pub(super) struct FailAfterObservingSvdInput {
    pub(super) observed: Vec<Vec<f64>>,
    pub(super) outputs: Option<Vec<DenseTensor>>,
}

impl Observer for FailAfterObservingSvdInput {
    fn script(script: &mut Script) {
        script
            .set(Op::Qr, Action::Panic("test only exercises SVD"))
            .set(Op::Eigh, Action::Panic("test only exercises SVD"))
            .set(Op::DotGeneral, Action::Panic("test only exercises SVD"));
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        match call {
            Call::Read(Op::Svd, input) => {
                if let Some(outputs) = self.outputs.take() {
                    return Some(Ok(outputs));
                }
                self.observed.push(f64_input(input));
                Some(injected("svd_into", "injected failure"))
            }
            Call::Into(Op::SvdInto, input, writes) => {
                self.observed.push(f64_input(input));
                for (write, what) in writes.iter().zip(["U", "singular values", "Vh"]) {
                    assert_zeroed_f64(write, what);
                }
                Some(injected("svd_into", "injected failure"))
            }
            _ => None,
        }
    }
}

/// Records each owned QR input; it succeeds only with `qr_succeeds`, unless
/// `outputs` holds a reply. `qr_into` always fails.
#[derive(Default)]
pub(super) struct FailAfterObservingQrInput {
    pub(super) observed: Vec<Vec<f64>>,
    pub(super) qr_succeeds: bool,
    pub(super) outputs: Option<Vec<DenseTensor>>,
}

impl Observer for FailAfterObservingQrInput {
    fn script(script: &mut Script) {
        script
            .set(Op::Svd, Action::Panic("test only exercises QR"))
            .set(Op::Eigh, Action::Panic("test only exercises QR"))
            .set(Op::DotGeneral, Action::Panic("test only exercises QR"))
            .set(Op::Qr, Action::Forward)
            .fail(&[Op::QrInto], None, "qr_into", "injected failure");
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        let Call::Read(Op::Qr, input) = call else {
            return None;
        };
        if let Some(outputs) = self.outputs.take() {
            return Some(Ok(outputs));
        }
        self.observed.push(f64_input(input));
        (!self.qr_succeeds).then(|| injected("qr", "injected failure"))
    }
}

/// Shape-based null space: no SVD, EIGH or GEMM; every QR fails.
#[derive(Default)]
pub(super) struct FailNullQr;

impl Observer for FailNullQr {
    fn script(script: &mut Script) {
        script
            .set_all(
                &[Op::Svd, Op::SvdInto, Op::Eigh, Op::DotGeneral],
                Action::Panic("the shape-based null space runs only QR"),
            )
            .fail(&[Op::QrInto], None, "qr_into", "injected null QR failure");
    }
}

/// Shape-based null space: no SVD, EIGH or GEMM; the second QR fails.
#[derive(Default)]
pub(super) struct FailSecondNullQr;

impl Observer for FailSecondNullQr {
    fn script(script: &mut Script) {
        script
            .set_all(
                &[Op::Svd, Op::SvdInto, Op::Eigh, Op::DotGeneral],
                Action::Panic("the shape-based null space runs only QR"),
            )
            .set(Op::Qr, Action::Forward)
            .fail(
                &[Op::QrInto],
                Some(2),
                "qr_into",
                "injected second-sector failure",
            );
    }
}

/// Records each EIGH input and fails it, unless `outputs` holds a reply.
#[derive(Default)]
pub(super) struct FailAfterObservingEighInput {
    pub(super) observed: Vec<Vec<f64>>,
    pub(super) outputs: Option<Vec<DenseTensor>>,
}

impl Observer for FailAfterObservingEighInput {
    fn script(script: &mut Script) {
        script.set_all(
            REQUIRED_EXCEPT_EIGH,
            Action::Panic("test only exercises EIGH"),
        );
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        match call {
            Call::Read(Op::Eigh, input) => {
                self.observed.push(f64_input(input));
                if let Some(outputs) = self.outputs.take() {
                    return Some(Ok(outputs));
                }
                Some(injected("eigh_into", "injected failure"))
            }
            Call::Into(Op::EighInto, input, writes) => {
                self.observed.push(f64_input(input));
                for (write, what) in writes.iter().zip(["eigenvalues", "eigenvectors"]) {
                    assert_zeroed_f64(write, what);
                }
                Some(injected("eigh_into", "injected failure"))
            }
            _ => None,
        }
    }
}

/// Every EIGH entry fails; tests read `counts().of(EIGH_ENTRIES)`.
#[derive(Default)]
pub(super) struct EighCallSpy;

pub(super) const EIGH_ENTRIES: &[Op] = &[Op::Eigh, Op::EighInto, Op::EighVals];

impl Observer for EighCallSpy {
    fn script(script: &mut Script) {
        script
            .set_all(
                REQUIRED_EXCEPT_EIGH,
                Action::Panic("test only exercises EIGH"),
            )
            .fail(
                &[Op::Eigh, Op::EighInto],
                None,
                "eigh_into",
                "injected failure",
            )
            .fail(&[Op::EighVals], None, "eigh_vals", "injected failure");
    }
}

/// The compiled default provider, so every feature set exercises the direct
/// full-SVD path it actually ships; `None` only where no provider has one.
fn native_full_svd_executor() -> Option<symtn_dense::DefaultDenseExecutor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut executor = symtn_dense::DefaultDenseExecutor::new();
    if executor.supports_svd_full() {
        return Some(executor);
    }
    assert!(matches!(
        executor.svd_full_owned(engine, DenseOwned::F64(vec![1.0]), 1, 1),
        Err(DenseError::Unsupported {
            op: "svd_full_owned",
            ..
        })
    ));
    None
}

/// Native full SVD only: no compact SVD, completion QR, EIGH or GEMM.
pub(super) struct NativeFullSvdSpy;

fn script_native_full_svd(script: &mut Script, svd: &'static str, qr: &'static str) {
    script.supports_svd_full = true;
    script
        .set(Op::SvdFullOwned, Action::Forward)
        .set(Op::Svd, Action::Panic(svd))
        .set(Op::Qr, Action::Panic(qr))
        .set(Op::Eigh, Action::Panic("test only exercises full SVD"))
        .set(
            Op::DotGeneral,
            Action::Panic("test only exercises full SVD"),
        );
}

impl Observer for NativeFullSvdSpy {
    fn script(script: &mut Script) {
        script_native_full_svd(
            script,
            "native full SVD must not use the compact SVD route",
            "native full SVD must not use orthonormal completion",
        );
    }
}

impl NativeFullSvdSpy {
    pub(super) fn new() -> Option<ScriptedExecutor<Self>> {
        native_full_svd_executor().map(|inner| ScriptedExecutor::with_inner(inner, Self))
    }
}

/// [`NativeFullSvdSpy`] whose second full SVD fails.
pub(super) struct FailSecondOwnedFullSvd;

impl Observer for FailSecondOwnedFullSvd {
    fn script(script: &mut Script) {
        script_native_full_svd(
            script,
            "claimed native full SVD must not retry the compact SVD route",
            "claimed native full SVD must not use completion",
        );
        script.fail(
            &[Op::SvdFullOwned],
            Some(2),
            "svd_full_owned",
            "injected second-sector failure",
        );
    }
}

impl FailSecondOwnedFullSvd {
    pub(super) fn new() -> Option<ScriptedExecutor<Self>> {
        native_full_svd_executor().map(|inner| ScriptedExecutor::with_inner(inner, Self))
    }
}

/// EIGH only; records the raw eigenvalues it returns.
#[derive(Default)]
pub(super) struct RecordingEigh {
    pub(super) raw_values: Vec<Vec<f64>>,
}

impl Observer for RecordingEigh {
    fn script(script: &mut Script) {
        script
            .set_all(
                REQUIRED_EXCEPT_EIGH,
                Action::Panic("test only exercises EIGH"),
            )
            .set(Op::Eigh, Action::Forward);
    }

    fn outputs(&mut self, op: Op, outputs: &[DenseTensor]) {
        if op == Op::Eigh {
            self.raw_values
                .push(outputs[0].as_f64_slice().unwrap().to_vec());
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LateGenericError(pub(super) usize);

impl fmt::Display for LateGenericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "late Generic provider failure at call {}",
            self.0
        )
    }
}

impl std::error::Error for LateGenericError {}

pub(super) struct LateGenericSpy {
    pub(super) rule: FactorGenericRule,
    pub(super) fail_at: usize,
    pub(super) calls: Cell<usize>,
    /// Its own identity: a spy that counts and fails its calls does not
    /// answer as `rule` does, so structure caches must not share entries
    /// between spies (or with `rule`).
    pub(super) identity: RuleIdentity,
}

/// Forwards the factorization entries and counts them; see [`CountingDense`]
/// readers for which counts each old field summed.
#[derive(Default)]
pub(super) struct CountingDense;

impl Observer for CountingDense {
    fn script(script: &mut Script) {
        script.set_all(
            &[
                Op::Svd,
                Op::SvdVals,
                Op::SvdInto,
                Op::Qr,
                Op::Eigh,
                Op::Eig,
                Op::EighInto,
            ],
            Action::Forward,
        );
    }
}

#[derive(Debug)]
pub(super) struct FullQrObservation {
    pub(super) input_shape: Vec<usize>,
    pub(super) q_shape: Vec<usize>,
    pub(super) r_shape: Vec<usize>,
    pub(super) values: Vec<Complex64>,
}

/// Full QR/LQ through the destination API only; checks and records each
/// input's layout and the destination shapes.
#[derive(Default)]
pub(super) struct FullQrInputSpy {
    pub(super) observations: Vec<FullQrObservation>,
}

impl Observer for FullQrInputSpy {
    fn script(script: &mut Script) {
        script
            .set(Op::Svd, Action::Panic("test only exercises full QR/LQ"))
            .set(
                Op::Qr,
                Action::Panic("full QR/LQ must use the destination API"),
            )
            .set(Op::Eigh, Action::Panic("test only exercises full QR/LQ"))
            .set(
                Op::DotGeneral,
                Action::Panic("test only exercises full QR/LQ"),
            )
            .set(Op::QrInto, Action::Forward);
    }

    fn call(&mut self, call: Call<'_, '_>) -> Option<Reply> {
        let Call::Into(Op::QrInto, input, writes) = call else {
            return None;
        };
        let input_shape = input.shape().to_vec();
        let (strides, offset, data_len, values) = match input {
            DenseRead::F64(view) => (
                view.strides().to_vec(),
                view.offset(),
                view.data().len(),
                view.data()
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            DenseRead::C64(view) => (
                view.strides().to_vec(),
                view.offset(),
                view.data().len(),
                view.data().to_vec(),
            ),
            _ => panic!("full QR/LQ fixture must be f64 or c64"),
        };
        assert_eq!(offset, 0);
        assert_eq!(strides, [1, input_shape[0]]);
        assert_eq!(data_len, input_shape.iter().product::<usize>());
        self.observations.push(FullQrObservation {
            input_shape,
            q_shape: writes[0].shape().to_vec(),
            r_shape: writes[1].shape().to_vec(),
            values,
        });
        None
    }
}

impl symtn_core::testing::Provider for LateGenericSpy {}

impl FusionRule for LateGenericSpy {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.rule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.rule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.rule.nsymbol(left, right, coupled)
    }
}

impl LateGenericSpy {
    pub(super) fn call<T>(&self, value: impl FnOnce() -> T) -> Result<T, LateGenericError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail_at {
            Err(LateGenericError(call))
        } else {
            Ok(value())
        }
    }
}

impl CheckedGenericFusion for LateGenericSpy {
    type Error = LateGenericError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.call(|| self.rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.call(|| self.rule.fusion_channels(left, right))
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
        self.call(|| self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for LateGenericSpy {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.dim_scalar(sector))
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.sqrt_dim_scalar(sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.inv_sqrt_dim_scalar(sector))
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.frobenius_schur_phase_scalar(sector))
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        self.call(|| self.rule.f_symbol_generic(a, b, c, d, e, f))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        coupled: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        self.call(|| self.rule.r_symbol_generic(a, b, coupled))
    }
}

pub(super) fn late_spy_calls(run: &dyn Fn(&LateGenericSpy)) -> usize {
    let probe = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    };
    run(&probe);
    probe.calls.get()
}

/// Matrix functions: forwards and counts EIGH, solves and GEMMs; the
/// `fail_solve_number`-th solve fails when set (see [`matrix_function_spy`]).
#[derive(Default)]
pub(super) struct MatrixFunctionCallSpy;

pub(super) const MATRIX_FUNCTION_EIGH: &[Op] = &[Op::Eigh, Op::EighInto];

impl Observer for MatrixFunctionCallSpy {
    fn script(script: &mut Script) {
        script.set_all(
            &[Op::Svd, Op::Qr, Op::Eigh, Op::EighInto, Op::Solve],
            Action::Forward,
        );
    }
}

/// A [`MatrixFunctionCallSpy`] whose `fail_solve_number`-th solve fails.
pub(super) fn matrix_function_spy(
    fail_solve_number: Option<usize>,
) -> ScriptedExecutor<MatrixFunctionCallSpy> {
    let mut spy = ScriptedExecutor::new(MatrixFunctionCallSpy);
    if let Some(nth) = fail_solve_number {
        spy.script.fail(
            &[Op::Solve],
            Some(nth),
            "solve_into",
            "injected sector failure",
        );
    }
    spy
}

/// A spy whose never-failing twin shares its identity.
pub(super) trait SpyTwin: CheckedGenericFusion + FusionRule + Sized {
    fn twin(&self) -> Self;
}

impl SpyTwin for LateGenericSpy {
    fn twin(&self) -> Self {
        Self {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: self.identity.clone(),
        }
    }
}

/// `source`'s layout bound to `spy`. Why not `bind_generic(source, spy)`:
/// a spy that fails, or whose call counts a test compares against a cold
/// walk, has its own identity (to a structure cache it is not
/// `FactorGenericRule`), so the space is admitted for the spy by a
/// twin that neither fails nor counts. That admission walks the source layout
/// once, outside the spy's own counts.
pub(super) fn bind_to_spy<P>(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    spy: &Arc<P>,
) -> BoundDynamicFusionMapSpace<P>
where
    P: SpyTwin + 'static,
    P::Error: std::fmt::Debug,
{
    let admitted = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(spy.twin()),
        source.space().homspace().clone(),
    )
    .unwrap();
    assert_eq!(
        admitted.space().structure().as_ref(),
        source.space().structure().as_ref(),
        "the spy admits the source layout"
    );
    BoundDynamicFusionMapSpace::bind_generic(admitted.space().clone(), Arc::clone(spy)).unwrap()
}

#[test]
fn scripted_default_eig_vals_reaches_the_forwarded_eig() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the trait's default `eig_vals` calls `eig`; with `EigVals` at
    // `Default` and `Eig` at `Forward`, the scripted executor routes it back
    // through its own `eig`, so the values come from the wrapped backend and
    // both entries are counted (as the hand-written doubles did).
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let data = [2.0_f64, 0.0, 0.0, -1.0];
    let input = DenseRead::F64(symtn_dense::DenseView::new(&data, &[2, 2], &[1, 2], 0).unwrap());
    let values = dense.eig_vals(engine, input).unwrap();
    assert_eq!(values.shape(), [2]);
    assert_eq!(dense.counts().eig_vals, 1);
    assert_eq!(dense.counts().eig, 1);
}
