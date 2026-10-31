//! The one `DenseExecutor` test double of symtn-matrixalgebra.
//!
//! Each entry point has a scripted [`Action`]: forward to the wrapped
//! `DefaultDenseExecutor`, keep the trait's default body (which funnels into
//! the required `svd`/`qr`/`eigh`/`dot_general_into` entries of this same
//! executor), or panic. Faults fail chosen calls with a backend error, and an
//! [`Observer`] can inspect arguments and outputs, or answer a call itself.
//! Every entry is counted where it enters, before its action runs.
//!
//! The crate's unit tests use it as `crate::tests::scripted_executor`;
//! integration tests include this file with `#[path]`.

#![allow(dead_code)]

use std::ops::{Deref, DerefMut};
use symtn_dense::Engine;

use symtn_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseFactorization, DenseOwned, DenseRead, DenseTensor, DenseWrite,
};

/// One scripted executor entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Svd,
    SvdInto,
    SvdVals,
    SvdFullOwned,
    Qr,
    QrInto,
    Eigh,
    EighInto,
    EighVals,
    Eig,
    EigVals,
    Solve,
    DotGeneral,
    FactorizeBatch,
}

const OPS: usize = 14;

/// What an entry point does once counted, faults and the observer allowing.
#[derive(Clone, Copy, Debug)]
pub enum Action {
    /// Call the wrapped executor's entry.
    Forward,
    /// Run the trait's default body on this executor. For the required
    /// entries (`svd`, `qr`, `eigh`, `dot_general_into`), which have none,
    /// this forwards.
    Default,
    /// Panic with the message: the test must not reach this entry.
    Panic(&'static str),
}

/// Calls per entry point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub svd: usize,
    pub svd_into: usize,
    pub svd_vals: usize,
    pub svd_full: usize,
    pub qr: usize,
    pub qr_into: usize,
    pub eigh: usize,
    pub eigh_into: usize,
    pub eigh_vals: usize,
    pub eig: usize,
    pub eig_vals: usize,
    pub solve: usize,
    pub dot_general: usize,
    pub factorize_batch: usize,
}

impl Counts {
    fn slot(&mut self, op: Op) -> &mut usize {
        match op {
            Op::Svd => &mut self.svd,
            Op::SvdInto => &mut self.svd_into,
            Op::SvdVals => &mut self.svd_vals,
            Op::SvdFullOwned => &mut self.svd_full,
            Op::Qr => &mut self.qr,
            Op::QrInto => &mut self.qr_into,
            Op::Eigh => &mut self.eigh,
            Op::EighInto => &mut self.eigh_into,
            Op::EighVals => &mut self.eigh_vals,
            Op::Eig => &mut self.eig,
            Op::EigVals => &mut self.eig_vals,
            Op::Solve => &mut self.solve,
            Op::DotGeneral => &mut self.dot_general,
            Op::FactorizeBatch => &mut self.factorize_batch,
        }
    }

    pub fn get(mut self, op: Op) -> usize {
        *self.slot(op)
    }

    /// Calls summed over `ops`.
    pub fn of(self, ops: &[Op]) -> usize {
        ops.iter().map(|&op| self.get(op)).sum()
    }
}

/// Fails the `nth` call (1-based, counted over `ops` together), or every call
/// when `nth` is `None`, with `DenseError::Backend { op: label, message }`.
#[derive(Clone, Copy, Debug)]
pub struct Fault {
    pub ops: &'static [Op],
    pub nth: Option<usize>,
    pub label: &'static str,
    pub message: &'static str,
}

/// The arguments of one call, lent to the [`Observer`] before it runs.
pub enum Call<'c, 'v> {
    /// `svd`, `qr`, `eigh`, `eig` and the values-only entries.
    Read(Op, &'c DenseRead<'v>),
    /// `svd_into` (`[u, s, vt]`), `qr_into` (`[q, r]`), `eigh_into`
    /// (`[values, vectors]`).
    Into(Op, &'c DenseRead<'v>, &'c [&'c DenseWrite<'v>]),
    Solve {
        a: &'c DenseRead<'v>,
        b: &'c DenseRead<'v>,
        x: &'c DenseWrite<'v>,
    },
    Gemm {
        lhs: &'c DenseRead<'v>,
        rhs: &'c DenseRead<'v>,
        config: &'c DenseDotConfig,
    },
}

/// A reply that replaces the scripted action: owned outputs (empty for a
/// destination entry) or an error. Values-only entries take none.
pub type Reply = Result<Vec<DenseTensor>, DenseError>;

/// Per-test observation and scripting. Every method defaults to doing nothing.
pub trait Observer {
    /// Sets the actions and faults this double starts with.
    fn script(_executor: &mut Script) {}

    /// Sees a call after counting, panics and faults, before its action;
    /// `Some` answers the call instead of the action.
    fn call(&mut self, _call: Call<'_, '_>) -> Option<Reply> {
        None
    }

    /// Sees the owned outputs of a forwarded or defaulted `svd`, `qr`,
    /// `eigh` or `eig`. Destination, values-only, full-SVD, solve and GEMM
    /// entries have no owned outputs to show; `factorize_batch` has
    /// [`Observer::batch_outputs`].
    fn outputs(&mut self, _op: Op, _outputs: &[DenseTensor]) {}

    /// Sees the outputs of a forwarded `factorize_batch`.
    fn batch_outputs(&mut self, _outputs: &mut Vec<Vec<DenseTensor>>) {}
}

impl Observer for () {}

/// The scripted part of a [`ScriptedExecutor`]: actions, faults, counts and
/// the full-SVD capability.
pub struct Script {
    pub inner: DefaultDenseExecutor,
    actions: [Action; OPS],
    pub faults: Vec<Fault>,
    pub counts: Counts,
    pub supports_svd_full: bool,
}

impl Script {
    pub fn set(&mut self, op: Op, action: Action) -> &mut Self {
        self.actions[op as usize] = action;
        self
    }

    /// Sets `action` on every op in `ops`.
    pub fn set_all(&mut self, ops: &[Op], action: Action) -> &mut Self {
        for &op in ops {
            self.set(op, action);
        }
        self
    }

    pub fn fail(
        &mut self,
        ops: &'static [Op],
        nth: Option<usize>,
        label: &'static str,
        message: &'static str,
    ) -> &mut Self {
        self.faults.push(Fault {
            ops,
            nth,
            label,
            message,
        });
        self
    }

    /// Counts the call, then applies a panic action or a matching fault.
    fn enter(&mut self, op: Op) -> Result<Action, DenseError> {
        *self.counts.slot(op) += 1;
        let action = self.actions[op as usize];
        if let Action::Panic(message) = action {
            panic!("{message}");
        }
        for fault in &self.faults {
            if fault.ops.contains(&op)
                && fault.nth.is_none_or(|nth| self.counts.of(fault.ops) == nth)
            {
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: fault.label,
                    message: fault.message.to_string(),
                });
            }
        }
        Ok(action)
    }
}

/// See the module documentation. `Deref`s to its observer, so a test reads
/// what the observer recorded as fields of the executor.
pub struct ScriptedExecutor<O: Observer = ()> {
    pub script: Script,
    pub observer: O,
}

impl<O: Observer> ScriptedExecutor<O> {
    pub fn new(observer: O) -> Self {
        Self::with_inner(DefaultDenseExecutor::new(), observer)
    }

    pub fn with_inner(inner: DefaultDenseExecutor, observer: O) -> Self {
        let mut script = Script {
            inner,
            actions: [Action::Default; OPS],
            faults: Vec::new(),
            counts: Counts::default(),
            supports_svd_full: false,
        };
        O::script(&mut script);
        Self { script, observer }
    }

    pub fn counts(&self) -> Counts {
        self.script.counts
    }
}

impl<O: Observer + Default> Default for ScriptedExecutor<O> {
    fn default() -> Self {
        Self::new(O::default())
    }
}

impl<O: Observer> Deref for ScriptedExecutor<O> {
    type Target = O;
    fn deref(&self) -> &O {
        &self.observer
    }
}

impl<O: Observer> DerefMut for ScriptedExecutor<O> {
    fn deref_mut(&mut self) -> &mut O {
        &mut self.observer
    }
}

/// Runs a trait default body with this executor: implements only the
/// required entries, by calling back into the scripted executor, so the
/// default's inner calls are scripted and counted like direct ones.
struct Defaulted<'a, O: Observer>(&'a mut ScriptedExecutor<O>);

impl<O: Observer> DenseExecutor for Defaulted<'_, O> {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.svd(engine, input)
    }
    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.qr(engine, input)
    }
    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eigh(engine, input)
    }
    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.0.dot_general_into(engine, output, lhs, rhs, config)
    }
    // The one provided entry another default body calls (`eig_vals`).
    fn eig(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eig(engine, input)
    }
}

/// Runs the trait's default `eig` (which calls nothing), for `eig` itself:
/// [`Defaulted`] routes `eig` back into the scripted executor.
struct DefaultEig;

impl DenseExecutor for DefaultEig {
    fn svd(
        &mut self,
        _engine: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the default eig calls no other entry")
    }
    fn qr(
        &mut self,
        _engine: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the default eig calls no other entry")
    }
    fn eigh(
        &mut self,
        _engine: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the default eig calls no other entry")
    }
    fn dot_general_into(
        &mut self,
        _engine: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        unreachable!("the default eig calls no other entry")
    }
}

impl<O: Observer> ScriptedExecutor<O> {
    /// The owned-output entries: `svd`, `qr`, `eigh`, `eig`.
    fn owned(
        &mut self,
        engine: &mut dyn Engine,
        op: Op,
        input: DenseRead<'_>,
        forward: ForwardCall<Vec<DenseTensor>>,
        default: DefaultCall<O, Vec<DenseTensor>>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let action = self.script.enter(op)?;
        if let Some(reply) = self.observer.call(Call::Read(op, &input)) {
            return reply;
        }
        let outputs = match action {
            Action::Forward => forward(&mut self.script.inner, engine, input),
            _ => default(&mut Defaulted(self), engine, input),
        }?;
        self.observer.outputs(op, &outputs);
        Ok(outputs)
    }

    /// The values-only entries.
    fn values(
        &mut self,
        engine: &mut dyn Engine,
        op: Op,
        input: DenseRead<'_>,
        forward: ForwardCall<DenseTensor>,
        default: DefaultCall<O, DenseTensor>,
    ) -> Result<DenseTensor, DenseError> {
        let action = self.script.enter(op)?;
        assert!(
            self.observer.call(Call::Read(op, &input)).is_none(),
            "values-only entries take no observer reply"
        );
        match action {
            Action::Forward => forward(&mut self.script.inner, engine, input),
            _ => default(&mut Defaulted(self), engine, input),
        }
    }
}

/// A forwarded call to the wrapped executor.
type ForwardCall<T> =
    fn(&mut DefaultDenseExecutor, &mut dyn Engine, DenseRead<'_>) -> Result<T, DenseError>;
/// A call answered by the trait default over the scripted executor.
type DefaultCall<O, T> =
    for<'a> fn(&mut Defaulted<'a, O>, &mut dyn Engine, DenseRead<'_>) -> Result<T, DenseError>;

impl<O: Observer> DenseExecutor for ScriptedExecutor<O> {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(
            engine,
            Op::Svd,
            input,
            |e, engine, i| e.svd(engine, i),
            |e, engine, i| e.0.script.inner.svd(engine, i),
        )
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(
            engine,
            Op::Qr,
            input,
            |e, engine, i| e.qr(engine, i),
            |e, engine, i| e.0.script.inner.qr(engine, i),
        )
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(
            engine,
            Op::Eigh,
            input,
            |e, engine, i| e.eigh(engine, i),
            |e, engine, i| e.0.script.inner.eigh(engine, i),
        )
    }

    fn eig(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(
            engine,
            Op::Eig,
            input,
            |e, engine, i| e.eig(engine, i),
            |_, engine, i| DefaultEig.eig(engine, i),
        )
    }

    fn svd_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        self.values(
            engine,
            Op::SvdVals,
            input,
            |e, engine, i| e.svd_vals(engine, i),
            |e, engine, i| e.svd_vals(engine, i),
        )
    }

    fn eigh_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        self.values(
            engine,
            Op::EighVals,
            input,
            |e, engine, i| e.eigh_vals(engine, i),
            |e, engine, i| e.eigh_vals(engine, i),
        )
    }

    fn eig_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        self.values(
            engine,
            Op::EigVals,
            input,
            |e, engine, i| e.eig_vals(engine, i),
            |e, engine, i| e.eig_vals(engine, i),
        )
    }

    fn svd_into(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::SvdInto)?;
        if let Some(reply) = self
            .observer
            .call(Call::Into(Op::SvdInto, &input, &[&u, &s, &vt]))
        {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.svd_into(engine, input, u, s, vt),
            _ => Defaulted(self).svd_into(engine, input, u, s, vt),
        }
    }

    fn qr_into(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::QrInto)?;
        if let Some(reply) = self
            .observer
            .call(Call::Into(Op::QrInto, &input, &[&q, &r]))
        {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.qr_into(engine, input, q, r),
            _ => Defaulted(self).qr_into(engine, input, q, r),
        }
    }

    fn eigh_into(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::EighInto)?;
        if let Some(reply) =
            self.observer
                .call(Call::Into(Op::EighInto, &input, &[&values, &vectors]))
        {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.eigh_into(engine, input, values, vectors),
            _ => Defaulted(self).eigh_into(engine, input, values, vectors),
        }
    }

    fn solve_into(
        &mut self,
        engine: &mut dyn Engine,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::Solve)?;
        if let Some(reply) = self.observer.call(Call::Solve {
            a: &a,
            b: &b,
            x: &x,
        }) {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.solve_into(engine, a, b, x),
            _ => Defaulted(self).solve_into(engine, a, b, x),
        }
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.script.enter(Op::DotGeneral)?;
        if let Some(reply) = self.observer.call(Call::Gemm {
            lhs: &lhs,
            rhs: &rhs,
            config,
        }) {
            return reply.map(drop);
        }
        self.script
            .inner
            .dot_general_into(engine, output, lhs, rhs, config)
    }

    fn supports_svd_full(&self) -> bool {
        self.script.supports_svd_full
    }

    fn svd_full_owned(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        match self.script.enter(Op::SvdFullOwned)? {
            Action::Forward => self.script.inner.svd_full_owned(engine, input, rows, cols),
            _ => Defaulted(self).svd_full_owned(engine, input, rows, cols),
        }
    }

    fn factorize_batch(
        &mut self,
        engine: &mut dyn Engine,
        op: DenseFactorization,
        inputs: &[DenseRead<'_>],
    ) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
        let mut outputs = match self.script.enter(Op::FactorizeBatch)? {
            Action::Forward => self.script.inner.factorize_batch(engine, op, inputs),
            _ => Defaulted(self).factorize_batch(engine, op, inputs),
        }?;
        self.observer.batch_outputs(&mut outputs);
        Ok(outputs)
    }
}
