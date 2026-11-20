//! Captures the grouped GEMM calls of a real calculation: iTEBD imaginary-time
//! ground-state search of the spin-1/2 antiferromagnetic Heisenberg chain with
//! U(1) and SU(2) symmetry, at three bond dimensions, with SVD truncation.
//!
//! For each model and bond dimension the driver runs the `WARMUP` schedule
//! untraced, then records `TRACED` steady-state update pairs stage by stage
//! (`itebd.theta`, `itebd.svd` with truncation, `itebd.pinv`,
//! `itebd.gamma_left`, `itebd.gamma_right`). It then applies the two-site
//! effective Hamiltonian `H_eff * theta` (the DMRG Lanczos matvec; steps
//! `heff.*`, f64 and c64) on the bond space iTEBD converged to, with the
//! Heisenberg MPO's virtual space. The `H_eff` tensors are random: only block
//! structure decides the job lists.
//!
//! Output: JSON lines, format `symtn-gemm-trace/1`. The first line is a
//! provenance record (`symtn_commit`, `symtn_base`, `provider`,
//! `engine_threads`, `workload`, `job_fields`, `layout`). Every other line is
//! one distinct call of a (model, chi, scalar, step) with `multiplicity`
//! (how many identical calls it stands for), `dtype`, `ops` (N/T/C per
//! operand), `alpha`, `beta` (`null`: fresh destination), `threads`, `runs`
//! (the caller's strided-run partition), and `jobs`, each
//! `[a_offset, b_offset, c_offset, m, n, k, lda, ldb, ldc]` column-major.
//!
//! ```text
//! cargo run --release -p symtn-network --features gemm-trace \
//!     --example gemm_trace_capture -- OUT.jsonl BRANCH_SHA BASE_SHA THREADS
//! gzip -9 OUT.jsonl
//! ```

#[path = "../../tests/support/network.rs"]
mod network_support;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use network_support::{conj, net, op};
use symtn::expert::gemm_trace::{self, Call};
use symtn::expert::{DenseScalar, MatrixOp};
use symtn::sector::{GroupId, LieGroupFusionRule, MultiplicityFreeAdmissionMode};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    Complex64, ContractSpec, CpuEngine, Engine, GradedSpace, Pool, Runtime, Svd, TensorMap,
    Truncation,
};

const MAX_JOBS: usize = 1 << 24;
const CHIS: [usize; 3] = [64, 256, 1024];
/// Untraced warm-up: (imaginary time step, AB+BA update pairs).
const WARMUP: [(f64, usize); 2] = [(0.1, 200), (0.01, 100)];
const TRACED: usize = 2;
/// The traced steps use the last warm-up step size.
const DT: f64 = WARMUP[1].0;
const E_EXACT: f64 = 0.25 - std::f64::consts::LN_2;

fn op_name(op: MatrixOp) -> &'static str {
    match op {
        MatrixOp::Identity => "N",
        MatrixOp::Transpose => "T",
        MatrixOp::Adjoint => "C",
    }
}

fn scalar(value: &DenseScalar) -> String {
    match *value {
        DenseScalar::F32(x) => format!("[{x:?},0.0]"),
        DenseScalar::F64(x) => format!("[{x:?},0.0]"),
        DenseScalar::C32(z) => format!("[{:?},{:?}]", z.re, z.im),
        DenseScalar::C64(z) => format!("[{:?},{:?}]", z.re, z.im),
    }
}

/// The call fields of one record, without provenance.
fn call_json(call: &Call) -> String {
    let mut s = format!(
        "\"dtype\":\"{:?}\",\"ops\":[\"{}\",\"{}\"],\"alpha\":{},\"beta\":{},\"threads\":{},\"runs\":{:?},\"jobs\":[",
        call.dtype,
        op_name(call.ops[0]),
        op_name(call.ops[1]),
        scalar(&call.alpha),
        call.beta.as_ref().map_or("null".into(), scalar),
        call.threads,
        call.runs,
    );
    for (i, j) in call.jobs.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        write!(
            s,
            "{sep}[{},{},{},{},{},{},{},{},{}]",
            j.a_offset, j.b_offset, j.c_offset, j.m, j.n, j.k, j.lda, j.ldb, j.ldc
        )
        .unwrap();
    }
    s.push(']');
    s
}

/// The (model, bond) a step belongs to.
#[derive(Clone)]
struct Tag {
    model: &'static str,
    states: usize,
    multiplets: usize,
}

/// Distinct records in first-seen order, with their multiplicity.
#[derive(Default)]
struct Sink {
    order: Vec<String>,
    count: HashMap<String, usize>,
    calls: usize,
    jobs: usize,
}

impl Sink {
    /// Runs `f` traced and files its calls under `step`.
    fn step<T>(&mut self, tag: &Tag, scalar: &str, step: &str, f: impl FnOnce() -> T) -> T {
        gemm_trace::start(MAX_JOBS);
        let x = f();
        let trace = gemm_trace::take();
        assert_eq!(trace.dropped_calls, 0, "raise MAX_JOBS");
        for call in &trace.calls {
            self.calls += 1;
            self.jobs += call.jobs.len();
            let key = format!(
                "\"model\":\"{}\",\"chi_states\":{},\"chi_multiplets\":{},\"scalar\":\"{scalar}\",\"step\":\"{step}\",{}",
                tag.model,
                tag.states,
                tag.multiplets,
                call_json(call)
            );
            *self.count.entry(key.clone()).or_insert_with(|| {
                self.order.push(key);
                0
            }) += 1;
        }
        x
    }

    fn render(&self, provenance: &str) -> String {
        let mut out = String::from(provenance);
        for key in &self.order {
            let (head, body) = key.split_at(key.find("\"dtype\"").unwrap());
            writeln!(
                out,
                "{{\"kind\":\"call\",{head}\"multiplicity\":{},{body}}}",
                self.count[key]
            )
            .unwrap();
        }
        out
    }
}

/// `H_eff * theta` for theta: `v (x) p (x) p <- v`, environments
/// L: `v (x) w <- v`, R: `v <- w (x) v`, MPO W: `p (x) w <- w (x) p`.
macro_rules! heff {
    ($sink:expr, $engine:expr, $rt:expr, $tag:expr, $d:ty, $v:expr, $p:expr, $w:expr) => {{
        let (engine, rt, v, p, w) = (&mut *$engine, $rt, $v, $p, $w);
        let l = TensorMap::<_, $d>::rand_with_seed(engine, rt, [v, w], [v], 1).unwrap();
        let theta = TensorMap::<_, $d>::rand_with_seed(engine, rt, [v, p, p], [v], 2).unwrap();
        let mpo = TensorMap::<_, $d>::rand_with_seed(engine, rt, [p, w], [w, p], 3).unwrap();
        let r = TensorMap::<_, $d>::rand_with_seed(engine, rt, [v], [w, v], 4).unwrap();
        let s1 = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1, 2, 3],
            domain: &[4],
        };
        let s2 = ContractSpec {
            lhs: &[1, 2],
            rhs: &[2, 3],
            codomain: &[0, 3, 4, 1],
            domain: &[2],
        };
        let s3 = ContractSpec {
            lhs: &[2, 3],
            rhs: &[2, 3],
            codomain: &[0, 1, 3, 4],
            domain: &[2],
        };
        let s4 = ContractSpec {
            lhs: &[3, 4],
            rhs: &[1, 0],
            codomain: &[0, 1, 2],
            domain: &[3],
        };
        let scalar = stringify!($d);
        // Warm structure caches first; job lists do not depend on them.
        let x1 = l.contract(engine, &theta, &s1).unwrap();
        let x2 = x1.contract(engine, &mpo, &s2).unwrap();
        let x3 = x2.contract(engine, &mpo, &s3).unwrap();
        x3.contract(engine, &r, &s4).unwrap();
        let sink = &mut *$sink;
        let x1 = sink.step($tag, scalar, "heff.L_theta", || {
            l.contract(engine, &theta, &s1).unwrap()
        });
        let x2 = sink.step($tag, scalar, "heff.W1", || {
            x1.contract(engine, &mpo, &s2).unwrap()
        });
        let x3 = sink.step($tag, scalar, "heff.W2", || {
            x2.contract(engine, &mpo, &s3).unwrap()
        });
        sink.step($tag, scalar, "heff.R", || {
            x3.contract(engine, &r, &s4).unwrap()
        });
    }};
}

/// iTEBD in Vidal form for one fusion rule, as in
/// `symtn-network/tests/itebd_smoke.rs`, with each bond-update stage traced.
macro_rules! itebd_for {
    ($module:ident, $rule:ty) => {
        mod $module {
            use super::*;

            pub type Map = TensorMap<$rule, f64>;

            pub struct State {
                pub ga: Map,
                pub la: Map,
                pub gb: Map,
                pub lb: Map,
            }

            /// One Vidal bond update; returns `(g1', l_mid', g2')`.
            #[allow(clippy::too_many_arguments)]
            pub fn bond_update(
                sink: &mut Option<(&mut Sink, &Tag)>,
                engine: &mut dyn Engine,
                gate: &Map,
                l_out: &Map,
                g1: &Map,
                l_mid: &Map,
                g2: &Map,
                trunc: &Truncation,
            ) -> (Map, Map, Map) {
                let mut step = |name: &str, f: &mut dyn FnMut(&mut dyn Engine) -> Map| match sink {
                    Some((sink, tag)) => sink.step(tag, "f64", name, || f(&mut *engine)),
                    None => f(&mut *engine),
                };
                let theta = step("itebd.theta", &mut |engine| {
                    net(
                        &[
                            op(&["l"], &["x"]),
                            op(&["x", "qa"], &["y"]),
                            op(&["y"], &["z"]),
                            op(&["z", "qb"], &["w"]),
                            op(&["w"], &["r"]),
                            op(&["pa", "pb"], &["qa", "qb"]),
                        ],
                        &["l", "pa"],
                        &["pb", "r"],
                    )
                    .contract(engine, &[l_out, g1, l_mid, g2, l_out, gate])
                    .unwrap()
                });
                let mut svd = None;
                let l_new = step("itebd.svd", &mut |engine| {
                    let Svd { u, s, vh } = theta.svd_compact(engine, &[0, 1], &[2, 3]).unwrap();
                    let found = s.domain()[0]
                        .find_truncated(&s.diagview(engine).unwrap(), trunc)
                        .unwrap();
                    let u = u
                        .restrict_leg(engine, &[(u.codomain_rank(), &found.selection)])
                        .unwrap();
                    let s = s
                        .restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
                        .unwrap();
                    let vh = vh.restrict_leg(engine, &[(0, &found.selection)]).unwrap();
                    let inverse_norm = 1.0 / s.norm(engine, 2.0).unwrap();
                    let l = s.scale(engine, inverse_norm).unwrap();
                    svd = Some((u, vh));
                    l
                });
                let (u, vh) = svd.take().unwrap();
                let l_out_inv = step("itebd.pinv", &mut |engine| {
                    l_out.pinv(engine, &[0], &[1], 1e-12).unwrap()
                });
                let g1_new = step("itebd.gamma_left", &mut |engine| {
                    net(
                        &[op(&["l"], &["x"]), op(&["x", "pa"], &["m"])],
                        &["l", "pa"],
                        &["m"],
                    )
                    .contract(engine, &[&l_out_inv, &u])
                    .unwrap()
                });
                let g2_new = step("itebd.gamma_right", &mut |engine| {
                    net(
                        &[op(&["m"], &["pb", "x"]), op(&["x"], &["r"])],
                        &["m", "pb"],
                        &["r"],
                    )
                    .contract(engine, &[&vh, &l_out_inv])
                    .unwrap()
                });
                (g1_new, l_new, g2_new)
            }

            /// The energy per bond, averaged over the two bonds.
            pub fn energy(engine: &mut dyn Engine, h: &Map, st: &State) -> f64 {
                let mut bond = |l_out: &Map, g1: &Map, l_mid: &Map, g2: &Map| {
                    let theta = net(
                        &[
                            op(&["l"], &["x"]),
                            op(&["x", "pa"], &["y"]),
                            op(&["y"], &["z"]),
                            op(&["z", "pb"], &["w"]),
                            op(&["w"], &["r"]),
                        ],
                        &["l", "pa", "pb"],
                        &["r"],
                    )
                    .contract(engine, &[l_out, g1, l_mid, g2, l_out])
                    .unwrap();
                    let num = net(
                        &[
                            conj(op(&["l", "pa", "pb"], &["r"])),
                            op(&["pa", "pb"], &["qa", "qb"]),
                            op(&["l", "qa", "qb"], &["r"]),
                        ],
                        &[],
                        &[],
                    )
                    .contract(engine, &[&theta, h, &theta])
                    .unwrap()
                    .scalar(engine)
                    .unwrap();
                    num / theta.inner(engine, &theta).unwrap()
                };
                0.5 * (bond(&st.lb, &st.ga, &st.la, &st.gb) + bond(&st.la, &st.gb, &st.lb, &st.ga))
            }

            /// Runs `updates` AB+BA update pairs from `state`.
            pub fn evolve(
                sink: &mut Option<(&mut Sink, &Tag)>,
                engine: &mut dyn Engine,
                gate: &Map,
                state: State,
                updates: usize,
                trunc: &Truncation,
            ) -> State {
                let State {
                    mut ga,
                    mut la,
                    mut gb,
                    mut lb,
                } = state;
                for _ in 0..updates {
                    (ga, la, gb) = bond_update(sink, engine, gate, &lb, &ga, &la, &gb, trunc);
                    (gb, lb, ga) = bond_update(sink, engine, gate, &la, &gb, &lb, &ga, trunc);
                }
                State { ga, la, gb, lb }
            }
        }
    };
}

itebd_for!(u1, U1FusionRule);
itebd_for!(su2, LieGroupFusionRule<MultiplicityFreeAdmissionMode>);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, out_path, branch, base, threads] = &args[..] else {
        panic!("usage: gemm_trace_capture OUT.jsonl BRANCH_SHA BASE_SHA THREADS");
    };
    let threads: usize = threads.parse().unwrap();
    let engine = &mut if threads == 1 {
        CpuEngine::serial()
    } else {
        CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
    };
    let rt = &Runtime::builder().build().unwrap();
    let provider = if cfg!(feature = "cpu-blas") {
        "cpu-blas"
    } else {
        "cpu-faer"
    };
    let mut sink = Sink::default();
    let provenance = format!(
        "{{\"kind\":\"provenance\",\"format\":\"symtn-gemm-trace/1\",\"symtn_commit\":\"{branch}\",\"symtn_base\":\"{base}\",\"provider\":\"{provider}\",\"engine_threads\":{threads},\"workload\":\"iTEBD imaginary time, warm-up (dt, pairs) {WARMUP:?}, then {TRACED} traced AB+BA update pairs at dt={DT} per chi; H_eff on the converged bond\",\"job_fields\":[\"a_offset\",\"b_offset\",\"c_offset\",\"m\",\"n\",\"k\",\"lda\",\"ldb\",\"ldc\"],\"layout\":\"column-major\"}}\n"
    );

    // U(1): twice Sz. MPO virtual space: identity ends + Sz (charge 0, x3),
    // S+ and S- (charge +-2).
    {
        let rule = Arc::new(U1FusionRule);
        let space = |pairs: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::clone(&rule),
                pairs.iter().map(|&(q, d)| (U1Irrep::new(q).unwrap(), d)),
            )
            .unwrap()
        };
        let p = space(&[(1, 1), (-1, 1)]);
        let w = space(&[(-2, 1), (0, 3), (2, 1)]);
        let h = TensorMap::from_subblock_fn(engine, rt, [&p, &p], [&p, &p], |trees, _| {
            let (cod, dom) = (trees.codomain_uncoupled(), trees.domain_uncoupled());
            if cod == dom {
                if cod[0] == cod[1] {
                    0.25
                } else {
                    -0.25
                }
            } else {
                0.5
            }
        })
        .unwrap();
        let gates: Vec<_> = WARMUP
            .iter()
            .map(|&(dt, _)| {
                h.scale(engine, -dt)
                    .unwrap()
                    .exp(engine, &[0, 1], &[2, 3])
                    .unwrap()
            })
            .collect();
        let gate = &gates[1];
        let (vb, va) = (space(&[(0, 1)]), space(&[(1, 1), (-1, 1)]));
        for chi in CHIS {
            let one = |c: &[&GradedSpace<U1FusionRule>],
                       d: &[&GradedSpace<U1FusionRule>],
                       e: &mut CpuEngine| {
                TensorMap::from_subblock_fn(e, rt, c.iter().copied(), d.iter().copied(), |_, _| 1.0)
                    .unwrap()
            };
            let state = u1::State {
                ga: one(&[&vb, &p], &[&va], engine),
                la: one(&[&va], &[&va], engine),
                gb: one(&[&va, &p], &[&vb], engine),
                lb: one(&[&vb], &[&vb], engine),
            };
            let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(1e-12).unwrap());
            let mut state = state;
            for (gate, &(_, pairs)) in gates.iter().zip(&WARMUP) {
                state = u1::evolve(&mut None, engine, gate, state, pairs, &trunc);
            }
            let v = state.lb.domain()[0].clone();
            let states: usize = v.degeneracies().iter().sum();
            let tag = Tag {
                model: "u1_heisenberg",
                states,
                multiplets: states,
            };
            let state = u1::evolve(
                &mut Some((&mut sink, &tag)),
                engine,
                gate,
                state,
                TRACED,
                &trunc,
            );
            let v = state.lb.domain()[0].clone();
            heff!(&mut sink, engine, rt, &tag, f64, &v, &p, &w);
            heff!(&mut sink, engine, rt, &tag, Complex64, &v, &p, &w);
            eprintln!(
                "u1 chi={chi}: bond {states} states, {} sectors, E/bond {:.6} (exact {E_EXACT:.6})",
                v.degeneracies().len(),
                u1::energy(engine, &h, &state)
            );
        }
    }

    // SU(2): twice spin. MPO virtual space: identity ends (spin 0, x2) and
    // the spin vector (spin 1).
    {
        let rule = Arc::new(
            LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(GroupId::su(2).unwrap())
                .unwrap(),
        );
        let irrep = |j: i64| rule.category().irrep(&[j]).unwrap();
        let space = |pairs: &[(i64, usize)]| {
            GradedSpace::try_new(Arc::clone(&rule), pairs.iter().map(|&(j, d)| (irrep(j), d)))
                .unwrap()
        };
        let p = space(&[(1, 1)]);
        let w = space(&[(0, 2), (2, 1)]);
        let singlet = irrep(0);
        // S_1 . S_2 is -3/4 on the singlet and 1/4 on the triplet.
        let h = TensorMap::from_subblock_fn(engine, rt, [&p, &p], [&p, &p], |trees, _| {
            if *trees.coupled() == singlet {
                -0.75
            } else {
                0.25
            }
        })
        .unwrap();
        let gates: Vec<_> = WARMUP
            .iter()
            .map(|&(dt, _)| {
                h.scale(engine, -dt)
                    .unwrap()
                    .exp(engine, &[0, 1], &[2, 3])
                    .unwrap()
            })
            .collect();
        let gate = &gates[1];
        let (vb, va) = (space(&[(0, 1)]), space(&[(1, 1)]));
        for chi in CHIS {
            type Space = GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>;
            let one = |c: &[&Space], d: &[&Space], e: &mut CpuEngine| {
                TensorMap::from_subblock_fn(e, rt, c.iter().copied(), d.iter().copied(), |_, _| 1.0)
                    .unwrap()
            };
            let state = su2::State {
                ga: one(&[&vb, &p], &[&va], engine),
                la: one(&[&va], &[&va], engine),
                gb: one(&[&va, &p], &[&vb], engine),
                lb: one(&[&vb], &[&vb], engine),
            };
            let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(1e-12).unwrap());
            let mut state = state;
            for (gate, &(_, pairs)) in gates.iter().zip(&WARMUP) {
                state = su2::evolve(&mut None, engine, gate, state, pairs, &trunc);
            }
            let v = state.lb.domain()[0].clone();
            let tag = su2_tag(&v);
            let state = su2::evolve(
                &mut Some((&mut sink, &tag)),
                engine,
                gate,
                state,
                TRACED,
                &trunc,
            );
            let v = state.lb.domain()[0].clone();
            heff!(&mut sink, engine, rt, &tag, f64, &v, &p, &w);
            heff!(&mut sink, engine, rt, &tag, Complex64, &v, &p, &w);
            eprintln!(
                "su2 chi={chi}: bond {} states, {} multiplets, {} sectors, E/bond {:.6} (exact {E_EXACT:.6})",
                tag.states,
                tag.multiplets,
                v.degeneracies().len(),
                su2::energy(engine, &h, &state)
            );
        }
    }

    std::fs::write(out_path, sink.render(&provenance)).unwrap();
    eprintln!(
        "gemm_trace_capture: {} calls, {} jobs, {} distinct records -> {out_path}",
        sink.calls,
        sink.jobs,
        sink.order.len()
    );
}

fn su2_tag(v: &GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>) -> Tag {
    let multiplets = v.degeneracies().iter().sum();
    Tag {
        model: "su2_heisenberg",
        states: v.dim().unwrap().round() as usize,
        multiplets,
    }
}
