//! `DenseExecutor::solve_batch_into`: the default's per-matrix call order
//! and length refusal; on the native provider the lanes' descriptor
//! forwarding, copy and failure boundary, the ordered error reduction over
//! the original system indices (ordinary failures and engine refusals),
//! the per-matrix delegation of every system outside the lane set, and the
//! per-lane workspace ownership.

use super::*;

/// An executor whose `solve_into` records each call's divisor payload
/// address and fails at call `fail_at`.
struct Recording {
    calls: Vec<usize>,
    fail_at: Option<usize>,
}

impl DenseExecutor for Recording {
    fn svd(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("only solves are issued")
    }

    fn qr(&mut self, _: &mut dyn Engine, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("only solves are issued")
    }

    fn eigh(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("only solves are issued")
    }

    fn dot_general_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        unreachable!("only solves are issued")
    }

    fn solve_into(
        &mut self,
        _: &mut dyn Engine,
        a: DenseRead<'_>,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let DenseRead::F64(a) = a else {
            unreachable!("the fixture is f64")
        };
        self.calls.push(a.data().as_ptr() as usize);
        if self.fail_at == Some(self.calls.len() - 1) {
            return Err(DenseError::OutOfBounds);
        }
        Ok(())
    }
}

/// Purpose: the default reaches `solve_into` once per system in index
/// order and stops at the first error; an empty batch calls nothing; and a
/// length mismatch of `b`, then of `x`, is `ShapeMismatch` with op
/// `solve_batch_into` and the expected length `a.len()`, before any call.
#[test]
fn default_solves_in_index_order_and_stops_at_the_first_error() {
    let blocks = [[1.0_f64], [2.0], [3.0]];
    let mut outs = [[0.0_f64]; 3];
    let a: Vec<_> = blocks
        .iter()
        .map(|block| DenseRead::F64(DenseView::new(block, &[1, 1], &[1, 1], 0).unwrap()))
        .collect();
    let address = |index: usize| blocks[index].as_ptr() as usize;
    let run = |fail_at, b: &[DenseRead<'_>], outs: &mut [[f64; 1]]| {
        let mut x: Vec<_> = outs
            .iter_mut()
            .map(|out| DenseWrite::F64(DenseViewMut::new(out, &[1, 1], &[1, 1], 0).unwrap()))
            .collect();
        let mut executor = Recording {
            calls: Vec::new(),
            fail_at,
        };
        let result = executor.solve_batch_into(&mut CpuEngine::serial(), &a, b, &mut x);
        (executor.calls, result)
    };
    let b = a.clone();
    assert_eq!(
        run(None, &b, &mut outs),
        (vec![address(0), address(1), address(2)], Ok(()))
    );
    assert_eq!(
        run(Some(1), &b, &mut outs),
        (vec![address(0), address(1)], Err(DenseError::OutOfBounds))
    );

    let mut executor = Recording {
        calls: Vec::new(),
        fail_at: None,
    };
    assert_eq!(
        executor.solve_batch_into(&mut CpuEngine::serial(), &[], &[], &mut []),
        Ok(())
    );
    let mut x: Vec<_> = outs
        .iter_mut()
        .map(|out| DenseWrite::F64(DenseViewMut::new(out, &[1, 1], &[1, 1], 0).unwrap()))
        .collect();
    let short_b = executor.solve_batch_into(&mut CpuEngine::serial(), &a, &a[..2], &mut x);
    let short_x = executor.solve_batch_into(&mut CpuEngine::serial(), &a, &a, &mut x[..1]);
    assert!(executor.calls.is_empty());
    for (result, actual) in [(short_b, 2), (short_x, 1)] {
        assert_eq!(
            result,
            Err(DenseError::ShapeMismatch {
                op: "solve_batch_into",
                expected: vec![3],
                actual: vec![actual],
            })
        );
    }
}

#[cfg(all(
    feature = "cpu-faer",
    not(feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
mod native {
    use super::*;
    use crate::tests::block_svd::{engine, numerics, PanicsAt};
    use tcontract_exec::{BatchPolicy, BatchStrategy};

    /// A scalar of the native fixtures.
    trait S: Copy + std::fmt::Debug + PartialEq + 'static {
        fn from_c(value: Complex64) -> Self;
        fn to_c(self) -> Complex64;
        fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
        fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    }

    impl S for f64 {
        fn from_c(value: Complex64) -> Self {
            value.re
        }
        fn to_c(self) -> Complex64 {
            Complex64::new(self, 0.0)
        }
        fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
            DenseRead::F64(view)
        }
        fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
            DenseWrite::F64(view)
        }
    }

    impl S for Complex64 {
        fn from_c(value: Complex64) -> Self {
            value
        }
        fn to_c(self) -> Complex64 {
            self
        }
        fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
            DenseRead::C64(view)
        }
        fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
            DenseWrite::C64(view)
        }
    }

    const SENTINEL: f64 = 9.0;

    /// One system's storage: `a` (`n x n`) at column stride `n + 2`, row
    /// stride 2 and offset 1; `b` (`n x k`) row-major with row stride `k + 1`
    /// and offset 2; `x` compact `[1, n]` at offset 3 between sentinels.
    struct System<T> {
        n: usize,
        k: usize,
        a: Vec<T>,
        b: Vec<T>,
        x: Vec<T>,
        shapes: ([usize; 2], [usize; 2]),
        strides: ([usize; 2], [usize; 2], [usize; 2]),
    }

    impl<T: S> System<T> {
        /// A diagonally dominant non-Hermitian `a` (well conditioned), or
        /// with `singular` the `[[1, 2], [2, 4]]` pattern in its leading
        /// corner, which partial pivoting leaves with an exactly zero pivot.
        fn new(n: usize, k: usize, seed: usize, singular: bool) -> Self {
            let value = |i: usize, j: usize, salt: usize| {
                let t = (i * 7 + j * 3 + seed * 5 + salt) as f64;
                Complex64::new((t * 0.37).sin(), (t * 0.11).cos() * 0.5)
            };
            let a_strides = [2, 2 * n + 2];
            let mut a = vec![T::from_c(Complex64::new(SENTINEL, 0.0)); 1 + a_strides[1] * n];
            for j in 0..n {
                for i in 0..n {
                    let mut entry = value(i, j, 0);
                    if i == j {
                        entry += Complex64::new(n as f64 + 2.0, 0.0);
                    }
                    if singular {
                        entry = Complex64::new(
                            if i + j == 0 {
                                1.0
                            } else if i + j == 1 {
                                2.0
                            } else if i == 1 && j == 1 {
                                4.0
                            } else {
                                0.0
                            },
                            0.0,
                        );
                    }
                    a[1 + i * a_strides[0] + j * a_strides[1]] = T::from_c(entry);
                }
            }
            let b_strides = [k + 1, 1];
            let mut b = vec![T::from_c(Complex64::new(SENTINEL, 0.0)); 2 + b_strides[0] * n];
            for i in 0..n {
                for j in 0..k {
                    b[2 + i * b_strides[0] + j] = T::from_c(value(i, j, 11));
                }
            }
            Self {
                n,
                k,
                a,
                b,
                x: vec![T::from_c(Complex64::new(SENTINEL, 0.0)); 3 + n * k + 2],
                shapes: ([n, n], [n, k]),
                strides: (a_strides, b_strides, [1, n]),
            }
        }

        /// Asserts `A X = B` entrywise, `A X` recomputed from the original
        /// strided storage, under the workspace rule: `n` terms reach each
        /// entry, and partial-pivot LU's backward error
        /// (`|A X - B| <= gamma_n g |A| |X|`, growth `g <= 2` for these
        /// diagonally dominant `A`) scales the bound by `2 n max|A| max|X|`.
        /// A nonfinite entry of `X` fails it.
        fn assert_solves(&self, what: &str) {
            let a = |i: usize, j: usize| {
                self.a[1 + i * self.strides.0[0] + j * self.strides.0[1]].to_c()
            };
            let x = |i: usize, j: usize| self.x[3 + i + j * self.n].to_c();
            // The bound scales with `max|X|`, so a nonfinite entry is refused
            // before it can widen the tolerance.
            for e in 0..self.n * self.k {
                let value = x(e % self.n, e / self.n);
                assert!(value.is_finite(), "{what}: X entry {e} is {value}");
            }
            let (mut ax, mut want) = (Vec::new(), Vec::new());
            for j in 0..self.k {
                for i in 0..self.n {
                    ax.push((0..self.n).map(|p| a(i, p) * x(p, j)).sum::<Complex64>());
                    want.push(self.b[2 + i * self.strides.1[0] + j].to_c());
                }
            }
            let largest = |values: &mut dyn Iterator<Item = Complex64>| {
                values.map(|value| value.norm()).fold(1.0, f64::max)
            };
            let a_max = largest(&mut (0..self.n * self.n).map(|e| a(e % self.n, e / self.n)));
            let x_max = largest(&mut (0..self.n * self.k).map(|e| x(e % self.n, e / self.n)));
            let conditioning = 2.0 * self.n as f64 * a_max * x_max;
            numerics::assert_slices_close_scaled(what, &ax, &want, self.n, conditioning);
        }

        /// Whether every element outside `x`'s view, and with `whole` every
        /// element of it, still holds the sentinel.
        fn untouched(&self, whole: bool) -> bool {
            let len = self.n * self.k;
            self.x.iter().enumerate().all(|(index, value)| {
                let inside = (3..3 + len).contains(&index);
                (inside && !whole) || *value == T::from_c(Complex64::new(SENTINEL, 0.0))
            })
        }
    }

    fn lanes() -> CpuEngine {
        engine(4, BatchPolicy::new(BatchStrategy::Outer))
    }

    fn solve<T: S>(engine: &mut CpuEngine, systems: &mut [System<T>]) -> Result<(), DenseError> {
        let (mut a, mut b, mut x) = (Vec::new(), Vec::new(), Vec::new());
        for system in systems.iter_mut() {
            let (av, bv, xv) = split(system);
            a.push(av);
            b.push(bv);
            x.push(xv);
        }
        DefaultDenseExecutor::new().solve_batch_into(engine, &a, &b, &mut x)
    }

    /// One system's three views at once.
    fn split<T: S>(system: &mut System<T>) -> (DenseRead<'_>, DenseRead<'_>, DenseWrite<'_>) {
        let System {
            a,
            b,
            x,
            shapes,
            strides,
            ..
        } = system;
        (
            T::read(DenseView::new(a, &shapes.0, &strides.0, 1).unwrap()),
            T::read(DenseView::new(b, &shapes.1, &strides.1, 2).unwrap()),
            T::write(DenseViewMut::new(x, &shapes.1, &strides.2, 3).unwrap()),
        )
    }

    /// Purpose: the lanes read `A` and `B` through their own independent
    /// strides and offsets and write `X` at its own offset, so each system
    /// of unequal orders and right-hand-side widths satisfies `A X = B`
    /// (residual from the original storage) with the sentinels around `X`
    /// intact; and an exactly singular system fails with the provider's
    /// numerical failure, leaves its whole `X` untouched, and the systems
    /// around it still solve.
    fn lanes_forward_descriptors_and_leave_a_singular_system_untouched<T: S>() {
        let mut systems: Vec<System<T>> = [(4, 3), (1, 1), (3, 2), (2, 5)]
            .iter()
            .enumerate()
            .map(|(seed, &(n, k))| System::new(n, k, seed, false))
            .collect();
        assert_eq!(solve(&mut lanes(), &mut systems), Ok(()));
        for (index, system) in systems.iter().enumerate() {
            system.assert_solves(&format!("system {index}"));
            assert!(system.untouched(false), "system {index}: sentinels");
        }
        // The oracle itself rejects a solution poisoned with NaN, or holding
        // one infinite entry of either sign.
        let len = systems[0].n * systems[0].k;
        for (poison, all) in [
            (f64::NAN, true),
            (f64::INFINITY, false),
            (f64::NEG_INFINITY, false),
        ] {
            let mut poisoned = System::<T>::new(systems[0].n, systems[0].k, 0, false);
            poisoned.x.clone_from(&systems[0].x);
            let entries = if all { 3..3 + len } else { 3..4 };
            poisoned.x[entries].fill(T::from_c(Complex64::new(poison, 0.0)));
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                poisoned.assert_solves("poisoned")
            }));
            assert!(
                caught.is_err(),
                "a solution holding {poison} must fail the oracle"
            );
        }

        let mut systems = vec![
            System::<T>::new(3, 2, 0, false),
            System::new(2, 2, 1, true),
            System::new(2, 3, 2, false),
        ];
        let error = solve(&mut lanes(), &mut systems).unwrap_err();
        // The direct route names the singular pivot after `solve_read_into`.
        assert!(
            matches!(
                &error,
                DenseError::NumericalFailure { op: "solve_into", message, .. }
                    if message.contains("solve_read_into") && message.contains("singular")
            ),
            "{error:?}"
        );
        assert!(systems[1].untouched(true));
        systems[0].assert_solves("before the singular system");
        systems[2].assert_solves("after the singular system");
    }

    #[test]
    fn native_lanes_forward_descriptors_and_leave_a_singular_system_untouched_f64() {
        lanes_forward_descriptors_and_leave_a_singular_system_untouched::<f64>();
    }

    #[test]
    fn native_lanes_forward_descriptors_and_leave_a_singular_system_untouched_c64() {
        lanes_forward_descriptors_and_leave_a_singular_system_untouched::<Complex64>();
    }

    /// Purpose: the batch error is the lowest original index's whichever
    /// route its system takes: a lower lane singularity beats a higher
    /// per-matrix shape error and the reverse order returns the shape
    /// error; in a mixed batch a per-matrix system between the first lane
    /// system and the failing one is still solved, and a failing one there
    /// wins over the later lane failure.
    #[test]
    fn native_errors_reduce_over_original_indices() {
        let mut lane_ok = System::<f64>::new(2, 2, 0, false);
        let mut singular = System::<f64>::new(2, 2, 1, true);
        let mut delegated = System::<f64>::new(2, 2, 2, false);
        let mut wide = vec![0.0_f64; 6];
        let shape_error = |error: &DenseError| {
            matches!(
                error,
                DenseError::ShapeMismatch {
                    op: "solve_into",
                    ..
                }
            )
        };
        let singular_error = |error: &DenseError| {
            matches!(
                error,
                DenseError::NumericalFailure {
                    op: "solve_into",
                    ..
                }
            )
        };

        let (sa, sb, sx) = split(&mut singular);
        let (da, db, _) = split(&mut delegated);
        let mut x = vec![sx, wide_x(&mut wide)];
        let error = DefaultDenseExecutor::new()
            .solve_batch_into(&mut lanes(), &[sa, da], &[sb, db], &mut x)
            .unwrap_err();
        assert!(singular_error(&error), "{error:?}");
        drop(x);

        let (sa, sb, sx) = split(&mut singular);
        let mut x = vec![wide_x(&mut wide), sx];
        let error = DefaultDenseExecutor::new()
            .solve_batch_into(&mut lanes(), &[da, sa], &[db, sb], &mut x)
            .unwrap_err();
        assert!(shape_error(&error), "{error:?}");
        drop(x);

        // [lane ok, vector right-hand side (per-matrix), lane singular, bad x].
        let vector_b = [4.0_f64, 6.0];
        let mut vector_x = [0.0_f64; 2];
        let diagonal = [2.0_f64, 0.0, 0.0, 3.0];
        let (oa, ob, ox) = split(&mut lane_ok);
        let (sa, sb, sx) = split(&mut singular);
        let mut x = vec![
            ox,
            DenseWrite::F64(DenseViewMut::new(&mut vector_x, &[2], &[1], 0).unwrap()),
            sx,
            wide_x(&mut wide),
        ];
        let error = DefaultDenseExecutor::new()
            .solve_batch_into(
                &mut lanes(),
                &[
                    oa,
                    DenseRead::F64(DenseView::new(&diagonal, &[2, 2], &[1, 2], 0).unwrap()),
                    sa,
                    da,
                ],
                &[
                    ob,
                    DenseRead::F64(DenseView::new(&vector_b, &[2], &[1], 0).unwrap()),
                    sb,
                    db,
                ],
                &mut x,
            )
            .unwrap_err();
        drop(x);
        assert!(singular_error(&error), "{error:?}");
        assert_eq!(vector_x, [2.0, 2.0]);
        lane_ok.assert_solves("first lane system");

        // [lane ok, bad x (per-matrix), lane singular]: the per-matrix error
        // below the lane failure wins.
        let (oa, ob, ox) = split(&mut lane_ok);
        let (sa, sb, sx) = split(&mut singular);
        let mut x = vec![ox, wide_x(&mut wide), sx];
        let error = DefaultDenseExecutor::new()
            .solve_batch_into(&mut lanes(), &[oa, da, sa], &[ob, db, sb], &mut x)
            .unwrap_err();
        assert!(shape_error(&error), "{error:?}");
    }

    /// Purpose: a failure to enter the lanes is the outcome of the lowest
    /// lane system, so on a broken engine a lower per-matrix shape error
    /// is still returned, and with the systems swapped the engine refusal is
    /// — the per-matrix entry's own refusal on that engine.
    #[test]
    fn native_engine_refusal_takes_the_first_lane_index() {
        let mut broken = lanes();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::block_kernel::driver::drive(
                &mut broken,
                &PanicsAt(0),
                &mut [(); 2],
                &mut crate::lanes::LaneTables::default(),
            )
        }));
        assert!(unwound.is_err() && broken.is_broken());
        let mut system = System::<f64>::new(2, 2, 0, false);
        let mut wide = vec![0.0_f64; 6];
        let expected = {
            let (a, b, x) = split(&mut system);
            DefaultDenseExecutor::new()
                .solve_into(&mut broken, a, b, x)
                .unwrap_err()
        };
        assert!(
            matches!(
                expected,
                DenseError::Backend {
                    op: "solve_into",
                    ..
                }
            ),
            "{expected:?}"
        );
        let (a, b, x) = split(&mut system);
        let bad = DenseWrite::F64(DenseViewMut::new(&mut wide, &[2, 3], &[1, 2], 0).unwrap());
        let mut xs = vec![bad, x];
        let error = DefaultDenseExecutor::new()
            .solve_batch_into(&mut broken, &[a, a], &[b, b], &mut xs)
            .unwrap_err();
        assert!(
            matches!(
                error,
                DenseError::ShapeMismatch {
                    op: "solve_into",
                    ..
                }
            ),
            "{error:?}"
        );
        xs.swap(0, 1);
        let error = DefaultDenseExecutor::new()
            .solve_batch_into(&mut broken, &[a, a], &[b, b], &mut xs)
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
        assert!(
            matches!(
                error,
                DenseError::Backend {
                    op: "solve_into",
                    ..
                }
            ),
            "{error:?}"
        );
    }

    /// A `[2, 3]` destination, whose shape no `[2, 2]` right-hand side
    /// matches.
    fn wide_x(data: &mut [f64]) -> DenseWrite<'_> {
        DenseWrite::F64(DenseViewMut::new(data, &[2, 3], &[1, 2], 0).unwrap())
    }

    fn f64r<'a>(
        data: &'a [f64],
        shape: &'static [usize],
        strides: &'static [usize],
    ) -> DenseRead<'a> {
        DenseRead::F64(DenseView::new(data, shape, strides, 0).unwrap())
    }

    /// The route and reason substrings a delegated provider failure keeps:
    /// tenferro's direct validation names `solve_read_into`, its
    /// `solve_read` fallback (a batched input) `solve`.
    fn reasons(case: &str) -> Option<&'static [&'static str]> {
        match case {
            "non-square" | "row mismatch" => Some(&["solve_read_into", "incompatible shapes"]),
            "mixed dtype" => Some(&["solve_read_into", "dtype mismatch"]),
            "batched singular" => Some(&["solve is singular"]),
            _ => None,
        }
    }

    /// The `(variant, op)` of an error, its comparable context.
    fn context(error: &DenseError) -> (std::mem::Discriminant<DenseError>, Option<&'static str>) {
        let op = match error {
            DenseError::Unsupported { op, .. }
            | DenseError::DTypeMismatch { op, .. }
            | DenseError::ShapeMismatch { op, .. }
            | DenseError::NumericalFailure { op, .. }
            | DenseError::Backend { op, .. } => Some(*op),
            _ => None,
        };
        (std::mem::discriminant(error), op)
    }

    /// Purpose: every system outside the lane set keeps the per-matrix
    /// `solve_into` result — values for the successful vector right-hand
    /// side and padded destination, variant and op for the rank-three,
    /// empty, non-square, row-mismatched, mixed-dtype and destination
    /// dtype or shape errors.
    #[test]
    fn native_systems_outside_the_lane_set_keep_the_per_matrix_result() {
        let a2 = [2.0_f64, 1.0, 0.0, 3.0];
        let a3 = [2.0_f64, 1.0, 0.0, 3.0, 1.0, 1.0];
        let a_batched = [2.0_f64, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0];
        let a_batched_singular = [2.0_f64, 0.0, 0.0, 3.0, 1.0, 2.0, 2.0, 4.0];
        let b2 = [4.0_f64, 6.0];
        let b4 = [4.0_f64, 6.0, 1.0, 1.0];
        let b6 = [4.0_f64, 6.0, 1.0, 1.0, 2.0, 2.0];
        let b4_f32 = [4.0_f32, 6.0, 1.0, 1.0];
        type Case<'a> = (
            DenseRead<'a>,
            DenseRead<'a>,
            Vec<usize>,
            Vec<usize>,
            usize,
            bool,
        );
        let cases: Vec<(&str, Case<'_>)> = vec![
            (
                "vector rhs",
                (
                    f64r(&a2, &[2, 2], &[1, 2]),
                    f64r(&b2, &[2], &[1]),
                    vec![2],
                    vec![1],
                    2,
                    false,
                ),
            ),
            (
                "padded x",
                (
                    f64r(&a2, &[2, 2], &[1, 2]),
                    f64r(&b4, &[2, 2], &[1, 2]),
                    vec![2, 2],
                    vec![1, 3],
                    6,
                    false,
                ),
            ),
            (
                "rank three",
                (
                    f64r(&a_batched, &[2, 2, 2], &[1, 2, 4]),
                    f64r(&b4, &[2, 1, 2], &[1, 2, 2]),
                    vec![2, 1, 2],
                    vec![1, 2, 2],
                    4,
                    false,
                ),
            ),
            (
                "batched singular",
                (
                    f64r(&a_batched_singular, &[2, 2, 2], &[1, 2, 4]),
                    f64r(&b4, &[2, 1, 2], &[1, 2, 2]),
                    vec![2, 1, 2],
                    vec![1, 2, 2],
                    4,
                    false,
                ),
            ),
            (
                "empty",
                (
                    f64r(&[], &[0, 0], &[1, 1]),
                    f64r(&[], &[0, 2], &[1, 1]),
                    vec![0, 2],
                    vec![1, 1],
                    0,
                    false,
                ),
            ),
            (
                "non-square",
                (
                    f64r(&a3, &[2, 3], &[1, 2]),
                    f64r(&b6, &[2, 3], &[1, 2]),
                    vec![2, 3],
                    vec![1, 2],
                    6,
                    false,
                ),
            ),
            (
                "row mismatch",
                (
                    f64r(&a2, &[2, 2], &[1, 2]),
                    f64r(&b6, &[3, 2], &[1, 3]),
                    vec![3, 2],
                    vec![1, 3],
                    6,
                    false,
                ),
            ),
            (
                "mixed dtype",
                (
                    f64r(&a2, &[2, 2], &[1, 2]),
                    DenseRead::F32(DenseView::new(&b4_f32, &[2, 2], &[1, 2], 0).unwrap()),
                    vec![2, 2],
                    vec![1, 2],
                    4,
                    true,
                ),
            ),
            (
                "x dtype",
                (
                    f64r(&a2, &[2, 2], &[1, 2]),
                    f64r(&b4, &[2, 2], &[1, 2]),
                    vec![2, 2],
                    vec![1, 2],
                    4,
                    true,
                ),
            ),
            (
                "x shape",
                (
                    f64r(&a2, &[2, 2], &[1, 2]),
                    f64r(&b4, &[2, 2], &[1, 2]),
                    vec![2, 1],
                    vec![1, 2],
                    2,
                    false,
                ),
            ),
        ];
        for (name, (a, b, x_shape, x_strides, len, f32_x)) in cases {
            let run = |batch: bool| {
                let (mut x64, mut x32) = (vec![0.0_f64; len], vec![0.0_f32; len]);
                let x = if f32_x {
                    DenseWrite::F32(DenseViewMut::new(&mut x32, &x_shape, &x_strides, 0).unwrap())
                } else {
                    DenseWrite::F64(DenseViewMut::new(&mut x64, &x_shape, &x_strides, 0).unwrap())
                };
                let result = if batch {
                    DefaultDenseExecutor::new().solve_batch_into(&mut lanes(), &[a], &[b], &mut [x])
                } else {
                    DefaultDenseExecutor::new().solve_into(&mut CpuEngine::serial(), a, b, x)
                };
                (result, x64, x32)
            };
            let (expected, expected64, expected32) = run(false);
            let (actual, actual64, actual32) = run(true);
            assert!(
                reasons(name).is_none() || actual.is_err(),
                "{name} must fail"
            );
            match (&expected, &actual) {
                (Ok(()), Ok(())) => {
                    assert_eq!((expected64, expected32), (actual64, actual32), "{name}")
                }
                (Err(expected), Err(actual)) => {
                    assert_eq!(
                        context(expected),
                        context(actual),
                        "{name}: {expected:?} / {actual:?}"
                    );
                    if let Some(want) = reasons(name) {
                        let message = match actual {
                            DenseError::Backend { message, .. }
                            | DenseError::NumericalFailure { message, .. } => message.as_str(),
                            other => panic!("{name}: no provider message in {other:?}"),
                        };
                        assert!(
                            want.iter().all(|part| message.contains(part)),
                            "{name}: {message:?} lacks {want:?}"
                        );
                    }
                }
                _ => panic!("{name}: {expected:?} / {actual:?}"),
            }
        }
    }

    /// The lane setups (id, calls) and every system's solve allocation calls
    /// of one batch of `systems` `8 x 8` systems with three right-hand
    /// sides on four threads split into two lanes.
    fn owner_calls(systems: usize) -> (std::collections::BTreeMap<usize, u64>, Vec<Option<u64>>) {
        use crate::block_kernel::factor::owners;
        let split_lanes = BatchPolicy::new(BatchStrategy::Split {
            lanes: std::num::NonZeroUsize::new(2).unwrap(),
        });
        let mut engine = engine(4, split_lanes);
        let n = 8;
        let mut a = vec![0.0_f64; systems * n * n];
        for (index, value) in a.iter_mut().enumerate() {
            let (i, j) = (index % n, index / n % n);
            *value = if i == j {
                10.0
            } else {
                ((index * 7) % 5) as f64 * 0.1
            };
        }
        let b = vec![1.0_f64; n * 3];
        let mut x = vec![0.0_f64; systems * n * 3];
        let (square, wide, strides) = ([n, n], [n, 3], [1, n]);
        let reads: Vec<_> = a
            .chunks(n * n)
            .map(|chunk| DenseRead::F64(DenseView::new(chunk, &square, &strides, 0).unwrap()))
            .collect();
        let rhs: Vec<_> = (0..systems)
            .map(|_| DenseRead::F64(DenseView::new(&b, &wide, &strides, 0).unwrap()))
            .collect();
        let mut outs: Vec<_> = x
            .chunks_mut(n * 3)
            .map(|chunk| DenseWrite::F64(DenseViewMut::new(chunk, &wide, &strides, 0).unwrap()))
            .collect();
        let mark = owners::mark();
        DefaultDenseExecutor::new()
            .solve_batch_into(&mut engine, &reads, &rhs, &mut outs)
            .unwrap();
        let start = a.as_ptr() as usize;
        let records = owners::within(mark, start..start + std::mem::size_of_val(&a[..]));
        assert_eq!(records.len(), systems);
        let setups = records
            .iter()
            .map(|record| (record.lane, record.setup.unwrap()))
            .collect();
        (setups, records.iter().map(|record| record.solve).collect())
    }

    /// Purpose: the provider workspace is a lane's, never a system's: no
    /// system's solve allocates (provider call included), and a batch of
    /// twice the systems on the same two lanes sets them up with exactly as
    /// many allocation calls.
    #[test]
    fn native_lane_workspace_does_not_grow_with_systems() {
        let (small_setups, small_solves) = owner_calls(4);
        let (large_setups, large_solves) = owner_calls(8);
        assert_eq!((small_setups.len(), large_setups.len()), (2, 2));
        assert_eq!(
            small_setups.values().sum::<u64>(),
            large_setups.values().sum::<u64>()
        );
        assert!(
            small_solves
                .iter()
                .chain(&large_solves)
                .all(|&calls| calls == Some(0)),
            "{small_solves:?} {large_solves:?}"
        );
    }
}
