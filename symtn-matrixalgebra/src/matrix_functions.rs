//! Matrix functions of fusion tensors, built from spectral factorizations,
//! coupled-sector linear solves, or a blockwise polynomial approximant at the
//! dense boundary.

#[cfg(test)]
use std::hash::Hash;

#[cfg(test)]
use symtn_core::MultiplicityFreeRigidSymbols;
use symtn_dense::{DenseDotConfig, DenseExecutor, DenseView, DenseViewMut};
use symtn_tensors::OperationError;
#[cfg(test)]
use symtn_tensors::{
    TensorContractBackend, TensorContractFusionExecutionContext, TreeTransformBackend,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{
    compact_eigh_owned, inverse_by_sector_dyn_into, is_hermitian_endomorphism_dyn,
    map_square_sectors_dyn_into, pinv_adjoint_by_sector_dyn_into, pinv_by_sector_dyn_into,
    solve_left_by_sector_dyn_into, validate_real_eigenvalues, BoundDynFactor,
    BoundDynamicTensorRef, EndomorphismFamily, FactorScalar,
};
#[cfg(test)]
use crate::factorize::{typed_from_bound_factor, BoundTensorMap, BoundTensorMapRef};

#[cfg(test)]
/// Matrix exponential of any endomorphism (TensorKit `exp!`, which checks only
/// `domain == codomain`); see [`exp_direct_into_dyn`] for the two routes.
pub(crate) fn exp<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<BoundTensorMap<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = exp_into_mf(dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Matrix exponential into a caller-admitted output space with the input's
/// hom space, in every fusion mode.
///
/// TensorKit's `exp!` (`linalg.jl:420-428`) checks only that the map is an
/// endomorphism and copies `LinearAlgebra.exp!` into each block; Julia's
/// `exp!` (stdlib v1.11 `dense.jl:677`) takes `exp(Hermitian(A))` — `V
/// exp(D) Vᴴ` from one eigendecomposition, made exactly Hermitian — when the
/// block is Hermitian and scaling-and-squaring Padé otherwise. symtn asks the
/// question once for the whole tensor, at its fixed relative threshold
/// `||(A - A†)/2||_F <= 64 * eps(real(D)) * ||A||_F` in every coupled-sector
/// block: every block then takes the same algorithm (issue #577, #1799).
/// The threshold picks an algorithm, not eigh's admission tolerance, and is
/// not user-configurable.
///
/// # Errors
///
/// - [`OperationError::SpaceMismatch`] for a non-endomorphism or an output
///   whose hom space differs, and
///   [`OperationError::UnsupportedTensorContractScope`] for an output layout
///   without canonical coupled-sector storage;
/// - see [`exp_pade13_sector`] for the Padé route's value errors; nonfinite
///   input is never Hermitian, so it always reaches that route.
#[doc(hidden)]
pub fn exp_direct_into_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    // Asked here, at the one point both routes pass through, so the refusal
    // names `exp` rather than whichever helper would notice first.
    EndomorphismFamily::Exp.require_endomorphism(input.space().space().homspace())?;
    // Asked directly rather than inferred from a failed EIGH, so a backend
    // failure is never mistaken for non-hermiticity.
    let spectral = is_hermitian_endomorphism_dyn(input)?;
    map_square_sectors_dyn_into(
        dense,
        input,
        output_space,
        |order| ExpWorkspace::new(order, spectral),
        |dense, workspace, source, order, output| match workspace {
            ExpWorkspace::Spectral { scaled } => {
                exp_spectral_sector(dense, scaled, source, order, output)
            }
            ExpWorkspace::Pade(workspace) => {
                exp_pade13_sector(dense, workspace, source, order, output)
            }
        },
    )
}

/// Scratch of the route [`exp_direct_into_dyn`] chose, sized once to the
/// largest coupled sector.
enum ExpWorkspace<D> {
    /// `V exp(D)` of the sector in flight.
    Spectral {
        scaled: Vec<D>,
    },
    Pade(Box<Pade13Workspace<D>>),
}

impl<D: FactorScalar> ExpWorkspace<D> {
    fn new(order: usize, spectral: bool) -> Result<Self, OperationError> {
        if !spectral {
            return Pade13Workspace::new(order).map(|workspace| Self::Pade(Box::new(workspace)));
        }
        let elements = order
            .checked_mul(order)
            .ok_or(OperationError::ElementCountOverflow)?;
        Ok(Self::Spectral {
            scaled: vec![D::zero(); elements],
        })
    }
}

/// One Hermitian coupled sector of [`exp_direct_into_dyn`]: Julia's
/// `exp(::Hermitian)` (stdlib v1.11 `symmetric.jl:708-722`) followed by
/// `exp!`'s `copytri!(…, 'U', true)` (`dense.jl:680`). The eigenvector order
/// and phase gauge cancel in `V exp(D) Vᴴ`, so neither is applied.
fn exp_spectral_sector<E, D>(
    dense: &mut E,
    scaled: &mut [D],
    source: &[D],
    order: usize,
    output: &mut [D],
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (values, vectors) = compact_eigh_owned(dense, &source[..order * order], order)?;
    validate_real_eigenvalues(&values)?;
    // Fold exp(D) into a column scaling of V rather than a diagonal GEMM
    // operand (issue #46).
    for (column, &value) in values.iter().enumerate() {
        let factor = D::from_real(value.exp());
        for row in 0..order {
            scaled[row + order * column] = vectors[row + order * column] * factor;
        }
    }
    let shape = [order, order];
    let strides = [1usize, order];
    let adjoint_strides = [order, 1usize];
    let lhs = DenseView::new(&scaled[..order * order], &shape, &strides, 0)
        .map_err(OperationError::Dense)?;
    let rhs =
        DenseView::new(&vectors, &shape, &adjoint_strides, 0).map_err(OperationError::Dense)?;
    let destination =
        DenseViewMut::new(output, &shape, &strides, 0).map_err(OperationError::Dense)?;
    dense
        .dot_general_into(
            D::dense_write(destination),
            D::dense_read(lhs),
            D::dense_read(rhs),
            &DenseDotConfig::matmul().with_conjugation(false, true),
        )
        .map_err(OperationError::Dense)?;
    // Exactly Hermitian, as Julia publishes it: the upper triangle is kept,
    // the lower one is its adjoint and the diagonal is real.
    for column in 0..order {
        let diagonal = column + order * column;
        output[diagonal] = D::from_real(output[diagonal].widen_complex().re);
        for row in 0..column {
            output[column + order * row] = FactorScalar::adjoint(output[row + order * column]);
        }
    }
    Ok(())
}

/// Higham's `theta_13`: the largest `||A||_1` for which the [13/13] Padé
/// approximant of `exp(A)` has backward error below the double-precision unit
/// roundoff (Higham 2005, Table 2.3). Julia's `LinearAlgebra.exp!` rounds this
/// to `5.4`; the exact constant is used here because nothing downstream depends
/// on matching Julia's choice of squaring count, only on the accuracy it buys.
const PADE13_THETA: f64 = 5.371_920_351_148_152;

/// Padé [13/13] numerator/denominator coefficients `b_0 .. b_13`, the same
/// table Julia's `LinearAlgebra.exp!` carries. `exp(A) ≈ (V - U)^-1 (V + U)`
/// with `U = A (b13 A^6 + ... )` odd and `V` even in `A`.
const PADE13_B: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// Scratch for the Padé evaluation, sized once to the largest coupled sector.
///
/// Every matrix buffer is `max_c n_c²` and `balance` is `max_c n_c`; the sector
/// loop borrows them and allocates nothing. `image` and `square` are swapped
/// rather than copied during the squaring phase.
struct Pade13Workspace<D> {
    scaled: Vec<D>,
    power2: Vec<D>,
    power4: Vec<D>,
    power6: Vec<D>,
    odd: Vec<D>,
    even: Vec<D>,
    inner: Vec<D>,
    accumulator: Vec<D>,
    image: Vec<D>,
    square: Vec<D>,
    /// LAPACK `gebal`'s `scale` output for the sector in flight.
    balance: Vec<f64>,
}

impl<D: FactorScalar> Pade13Workspace<D> {
    fn new(order: usize) -> Result<Self, OperationError> {
        let elements = order
            .checked_mul(order)
            .ok_or(OperationError::ElementCountOverflow)?;
        let buffer = || vec![D::zero(); elements];
        Ok(Self {
            scaled: buffer(),
            power2: buffer(),
            power4: buffer(),
            power6: buffer(),
            odd: buffer(),
            even: buffer(),
            inner: buffer(),
            accumulator: buffer(),
            image: buffer(),
            square: buffer(),
            balance: vec![0.0; order],
        })
    }
}

/// One coupled sector of the matrix exponential by scaling-and-squaring Padé
/// [13/13].
///
/// The general-endomorphism arm of [`exp_direct_into_dyn`], and symtn's port of what
/// TensorKit's `exp!` gets from `LinearAlgebra.exp!`: N. J. Higham, "The
/// Scaling and Squaring Method for the Matrix Exponential Revisited", SIAM J.
/// Matrix Anal. Appl. 26(4), 2005.
///
/// Per coupled sector the block is first balanced — LAPACK `gebal('B')`, run
/// where Julia's `exp!` runs it and undone where Julia undoes it, see
/// [`balance_in_place`] — and then, with
/// `s = max(0, ceil(log2(||A||_1 / theta_13)))` over the *balanced* norm and
/// `B = A / 2^s`:
///
/// ```text
/// U = B (B^6 (b13 B^6 + b11 B^4 + b9 B^2) + b7 B^6 + b5 B^4 + b3 B^2 + b1 I)
/// V =    B^6 (b12 B^6 + b10 B^4 + b8 B^2) + b6 B^6 + b4 B^4 + b2 B^2 + b0 I
/// exp(B) = (V - U)^-1 (V + U),   exp(A) = exp(B)^(2^s)
/// ```
///
/// Why a single [13/13] degree while Julia switches to degree 3/5/7/9 below
/// `||A||_1 = 2.1`: the low-degree branch is a speed optimization at equal
/// accuracy, and one branch is one thing to get wrong. Values therefore agree
/// with Julia to approximant error, not bit for bit.
///
/// Complexity: `O(Σ_c n_c³)` time — six GEMMs, one solve and `s` squarings per
/// sector — and `O(Σ_c n_c²)` result, with no cross-sector coupling and no
/// allocation inside the sector loop. The Padé workspace is `O(max_c n_c²)`,
/// sized once to the largest sector; on the canonical direct-region layout that
/// is the whole of the scratch, while the packed fallback in
/// [`map_square_sectors_dyn_into`] matricizes every sector up front and so adds
/// `O(Σ_c n_c²)` of its own.
///
/// # Errors
///
/// - [`OperationError::InvalidArgument`] for a nonfinite block, which has no
///   exponential and would otherwise reach the backend as a silent NaN, or for
///   a block whose balanced column 1-norm overflows to infinity even though
///   every entry is finite — the scaling count derived from it is not
///   representable;
/// - [`OperationError::Dense`] from the backend, including
///   [`symtn_dense::DenseError::Unsupported`] when the selected executor has no
///   dense solve. Nothing is published unless every sector succeeded.
fn exp_pade13_sector<E, D>(
    dense: &mut E,
    workspace: &mut Pade13Workspace<D>,
    source: &[D],
    order: usize,
    output: &mut [D],
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let Pade13Workspace {
        scaled,
        power2,
        power4,
        power6,
        odd,
        even,
        inner,
        accumulator,
        image,
        square,
        balance,
    } = workspace;
    let elements = order * order;

    // The finiteness gate, before anything reads the block: a nonfinite entry
    // would otherwise pass through balancing and the whole approximant as a NaN
    // and come back as an opaque backend failure or a silently wrong tensor.
    for &value in &source[..elements] {
        let value = value.widen_complex();
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(OperationError::InvalidArgument {
                message: "exp requires finite coupled-sector blocks",
            });
        }
    }

    // Balance first, exactly where Julia's `exp!` does it (stdlib v1.11
    // `dense.jl:684`), and undo it after the squaring phase: the 1-norm the
    // squaring count comes from is the *balanced* norm, so a block that is
    // merely badly scaled — `[0 1e16; 1e-16 0]`, whose exponential is a
    // hyperbolic rotation — costs no squarings instead of fifty-one, and does
    // not lose the digits fifty-one squarings of an approximant cost.
    scaled[..elements].copy_from_slice(&source[..elements]);
    let (ilo, ihi) = balance_in_place(&mut scaled[..elements], order, &mut balance[..order]);

    let mut norm1 = 0.0_f64;
    for column in 0..order {
        let mut column_sum = 0.0_f64;
        for row in 0..order {
            column_sum += scaled[row + order * column].widen_complex().norm();
        }
        norm1 = norm1.max(column_sum);
    }
    // Finite entries do not imply a finite norm: two entries near `f64::MAX` in
    // one column sum to infinity. The squaring count below is a saturating cast
    // — `ceil(log2(inf / theta_13)) as u32` is `u32::MAX`, whose `i32` reading
    // is -1 — so an infinite norm would scale the block *up* and then square it
    // ~4.3e9 times, turning a finite input into a hang. symtn already refuses
    // nonfinite blocks by policy; a norm that is not representable is the same
    // refusal one derivation later.
    if !norm1.is_finite() {
        return Err(OperationError::InvalidArgument {
            message: "exp requires coupled-sector blocks with a finite 1-norm",
        });
    }

    let squarings = if norm1 > PADE13_THETA {
        (norm1 / PADE13_THETA).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    // An exact power of two, so the scaling is exact and the squaring phase
    // undoes it exactly.
    let scale = D::from_real(2.0_f64.powi(-(squarings as i32)));
    for value in &mut scaled[..elements] {
        *value = *value * scale;
    }

    gemm(dense, power2, scaled, scaled, order)?;
    gemm(dense, power4, power2, power2, order)?;
    gemm(dense, power6, power2, power4, order)?;

    let b = &PADE13_B;
    combine(
        inner,
        &[
            (b[13], &power6[..]),
            (b[11], &power4[..]),
            (b[9], &power2[..]),
        ],
        order,
    );
    gemm(dense, accumulator, power6, inner, order)?;
    add_terms(
        accumulator,
        &[
            (b[7], &power6[..]),
            (b[5], &power4[..]),
            (b[3], &power2[..]),
        ],
        b[1],
        order,
    );
    gemm(dense, odd, scaled, accumulator, order)?;

    combine(
        inner,
        &[
            (b[12], &power6[..]),
            (b[10], &power4[..]),
            (b[8], &power2[..]),
        ],
        order,
    );
    gemm(dense, even, power6, inner, order)?;
    add_terms(
        even,
        &[
            (b[6], &power6[..]),
            (b[4], &power4[..]),
            (b[2], &power2[..]),
        ],
        b[0],
        order,
    );

    // `inner` = V + U (right-hand side), `accumulator` = V - U (system matrix).
    // `D` carries no `Sub`, so the negation rides a real scalar multiply.
    let minus_one = D::from_real(-1.0);
    for index in 0..elements {
        inner[index] = even[index] + odd[index];
        accumulator[index] = even[index] + minus_one * odd[index];
    }
    solve(dense, accumulator, inner, image, order)?;

    for _ in 0..squarings {
        gemm(dense, square, image, image, order)?;
        std::mem::swap(image, square);
    }

    unbalance_in_place(&mut image[..elements], order, &balance[..order], ilo, ihi);

    output[..elements].copy_from_slice(&image[..elements]);
    Ok(())
}

/// `|Re z| + |Im z|`, LAPACK's `CABS1` — the magnitude `gebal` sums and
/// compares, and plain `abs` on a real block.
fn abs1<D: FactorScalar>(value: D) -> f64 {
    let value = value.widen_complex();
    value.re.abs() + value.im.abs()
}

/// `IDAMAX`/`IZAMAX` followed by `ABS` of the element it picked, which is how
/// `gebal` forms `CA`/`RA` (`dgebal.f:343-346`, `zgebal.f:348-351`).
///
/// The two magnitudes are deliberately different and must stay so: `IZAMAX`
/// *selects* by `CABS1 = |Re| + |Im|`, but `ABS` of the selected element is its
/// **modulus**. Collapsing both onto `abs1` would inflate every complex `CA`
/// and `RA` by up to a factor of `sqrt(2)`, which is a factor of the radix in
/// the loop that consumes them.
///
/// `IDAMAX` compares strictly, so ties keep the earliest element; starting the
/// incumbent key at `-inf` reproduces that, including LAPACK's zero for an
/// all-zero span.
fn iamax_modulus<D: FactorScalar>(values: impl Iterator<Item = D>) -> f64 {
    let mut key = f64::NEG_INFINITY;
    let mut modulus = 0.0;
    for value in values {
        let candidate = abs1(value);
        if candidate > key {
            key = candidate;
            modulus = value.widen_complex().norm();
        }
    }
    modulus
}

/// `DNRM2`/`DZNRM2`: the Euclidean norm, by Blue's three-accumulator scaling
/// (`dnrm2.f90:139-197`, `dznrm2.f90` identically over the interleaved real and
/// imaginary parts), so that a span containing entries near the overflow or
/// underflow threshold still gets an exact-to-rounding answer instead of `inf`
/// or `0`.
///
/// This is the norm `gebal` measures its rows and columns with
/// (`dgebal.f:341-342`), not the `abs1` sum: they order pairs of vectors
/// differently, so the `abs1` sum reaches a different scale vector — on
/// `[0 4 0; 1 0 1; 1 1 0]`, `[1, 1/2, 1]` instead of LAPACK's `[2, 1, 1]`.
///
/// The accumulation is `f64` for every `D`, one step better than the reference
/// (which works in the component type), because the bounds that gate the
/// balancing loop already come from the component type via
/// [`FactorScalar::safe_minimum`]: it is the *factor* that has to be
/// representable in `D`, not the norms it was derived from.
fn nrm2<D: FactorScalar>(values: impl Iterator<Item = D>) -> f64 {
    // Blue's constants for `f64`: `radix^ceiling((minexponent - 1) / 2)` and
    // friends, `dnrm2.f90:103-110` evaluated at `wp = real64`.
    let tsml = 2.0_f64.powi(-511);
    let tbig = 2.0_f64.powi(486);
    let ssml = 2.0_f64.powi(537);
    let sbig = 2.0_f64.powi(-538);

    let mut notbig = true;
    let mut asml = 0.0_f64;
    let mut amed = 0.0_f64;
    let mut abig = 0.0_f64;
    for value in values {
        let value = value.widen_complex();
        for ax in [value.re.abs(), value.im.abs()] {
            if ax > tbig {
                abig += (ax * sbig) * (ax * sbig);
                notbig = false;
            } else if ax < tsml {
                if notbig {
                    asml += (ax * ssml) * (ax * ssml);
                }
            } else {
                amed += ax * ax;
            }
        }
    }

    let (scale, sum_of_squares) = if abig > 0.0 {
        if amed > 0.0 || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        (1.0 / sbig, abig)
    } else if asml > 0.0 {
        if amed > 0.0 || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / ssml;
            let (smaller, larger) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            (1.0, larger * larger * (1.0 + (smaller / larger).powi(2)))
        } else {
            (1.0 / ssml, asml)
        }
    } else {
        (1.0, amed)
    };
    scale * sum_of_squares.sqrt()
}

/// LAPACK `dgebal`/`zgebal` with `job = 'B'`, in place on a column-major
/// `order x order` block: the permutation that pushes already-isolated
/// eigenvalues out of the active window, then the radix-2 diagonal similarity
/// that equalizes row and column norms inside it.
///
/// This is the balancing Julia's `LinearAlgebra.exp!` runs before its Padé
/// evaluation and undoes after it (stdlib v1.11 `dense.jl:684` and `769-780`),
/// which is what TensorKit's `exp!` inherits per block, so symtn's general arm
/// runs it too. Both halves of `'B'` are ported because both halves are what
/// `exp!` asks for and what it undoes.
///
/// Returns the **0-based, inclusive** active window `(ilo, ihi)`. `scale` is
/// LAPACK's dual-purpose output: inside the window, the diagonal factor applied
/// to that index; outside it, the **1-based** index the position was exchanged
/// with. The 1-based encoding is LAPACK's and is kept because
/// [`unbalance_in_place`] reads both meanings out of the one array — including
/// the corner where a fully isolated block leaves `ilo = ihi = 0` and the lone
/// window entry is the self-exchange `1`, which the undo then reads as the
/// harmless scaling factor `1.0`, exactly as Julia does.
pub(crate) fn balance_in_place<D: FactorScalar>(
    matrix: &mut [D],
    order: usize,
    scale: &mut [f64],
) -> (usize, usize) {
    if order == 0 {
        return (0, 0);
    }
    scale.fill(1.0);
    let mut k = 0usize;
    let mut l = order - 1;

    // Rows that isolate an eigenvalue, pushed down; then columns that do,
    // pushed left. The exchanges are partial, as in LAPACK: a column swap
    // touches rows `0..=l` and a row swap columns `k..`, leaving the already
    // isolated borders alone.
    loop {
        let isolated = (0..=l)
            .rev()
            .find(|&j| (0..=l).all(|i| i == j || abs1(matrix[j + order * i]) == 0.0));
        let Some(j) = isolated else { break };
        scale[l] = (j + 1) as f64;
        exchange(matrix, order, j, l, l, k);
        if l == 0 {
            return (0, 0);
        }
        l -= 1;
    }
    loop {
        let isolated =
            (k..=l).find(|&j| (k..=l).all(|i| i == j || abs1(matrix[i + order * j]) == 0.0));
        let Some(j) = isolated else { break };
        scale[k] = (j + 1) as f64;
        exchange(matrix, order, j, k, l, k);
        k += 1;
    }

    // The iterative scaling, transcribed from `dgebal.f`: at each index, walk
    // the row and column norms towards each other in factors of the radix while
    // neither the scaled quantities nor the accumulated factor can overflow or
    // underflow, and keep the step only if it shrinks their sum by more than
    // `FACTOR`.
    const RADIX: f64 = 2.0;
    const FACTOR: f64 = 0.95;
    // `xLAMCH('S') / xLAMCH('P')` of the *component* type (`dgebal.f:330-333`,
    // `sgebal.f:330-333`), not of `f64` unconditionally: these bounds are what
    // stops the radix loop, so they decide whether the factor it produces is
    // representable in `D`.
    let sfmin1 = D::safe_minimum() / D::epsilon();
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * RADIX;
    let sfmax2 = 1.0 / sfmin2;
    let mut converged = false;
    while !converged {
        converged = true;
        for i in k..=l {
            // `dgebal.f:341-346`. The norms span the whole window including the
            // diagonal — `DNRM2(L-K+1, A(K,I), 1)` is contiguous — while `CA`
            // and `RA` reach outside it, down every row and along every column
            // the similarity will touch.
            let mut c = nrm2((k..=l).map(|row| matrix[row + order * i]));
            let mut r = nrm2((k..=l).map(|column| matrix[i + order * column]));
            let mut ca = iamax_modulus((0..=l).map(|row| matrix[row + order * i]));
            let mut ra = iamax_modulus((k..order).map(|column| matrix[i + order * column]));
            // Guard against a row or column norm that underflowed to zero.
            if c == 0.0 || r == 0.0 {
                continue;
            }
            // `dgebal.f:354-358` bails out on a NaN to avoid an infinite loop.
            // This signature has no error channel to bail through, so the index
            // is skipped instead — same effect on termination, since only a
            // scaling that fired clears `converged`.
            if (c + ca + r + ra).is_nan() {
                continue;
            }
            let mut g = r / RADIX;
            let mut f = 1.0_f64;
            let s = c + r;
            while c < g && f.max(c).max(ca) < sfmax2 && r.min(g).min(ra) > sfmin2 {
                f *= RADIX;
                c *= RADIX;
                ca *= RADIX;
                r /= RADIX;
                g /= RADIX;
                ra /= RADIX;
            }
            g = c / RADIX;
            while g >= r && r.max(ra) < sfmax2 && f.min(c).min(g).min(ca) > sfmin2 {
                f /= RADIX;
                c /= RADIX;
                g /= RADIX;
                ca /= RADIX;
                r *= RADIX;
                ra *= RADIX;
            }
            if c + r >= FACTOR * s {
                continue;
            }
            if f < 1.0 && scale[i] < 1.0 && f * scale[i] <= sfmin1 {
                continue;
            }
            if f > 1.0 && scale[i] > 1.0 && scale[i] >= sfmax1 / f {
                continue;
            }
            scale[i] *= f;
            converged = false;
            // `f` is a power of the radix, so both factors are exact and the
            // similarity introduces no rounding of its own.
            let row_factor = D::from_real(1.0 / f);
            let column_factor = D::from_real(f);
            for column in k..order {
                matrix[i + order * column] = matrix[i + order * column] * row_factor;
            }
            for row in 0..=l {
                matrix[row + order * i] = matrix[row + order * i] * column_factor;
            }
        }
    }
    (k, l)
}

/// LAPACK's partial row/column exchange of `j` and `m`: columns over rows
/// `0..=rows`, rows over columns `first_column..`.
fn exchange<D: Copy>(
    matrix: &mut [D],
    order: usize,
    j: usize,
    m: usize,
    rows: usize,
    first_column: usize,
) {
    if j == m {
        return;
    }
    for row in 0..=rows {
        matrix.swap(row + order * j, row + order * m);
    }
    for column in first_column..order {
        matrix.swap(j + order * column, m + order * column);
    }
}

/// Undoes [`balance_in_place`] on an exponentiated block, in the order Julia's
/// `exp!` does it (stdlib v1.11 `dense.jl:769-780`): the diagonal similarity
/// inside the window first, then the exchanges below it in reverse order and
/// the ones above it in forward order — each the reverse of the order they were
/// made in.
fn unbalance_in_place<D: FactorScalar>(
    matrix: &mut [D],
    order: usize,
    scale: &[f64],
    ilo: usize,
    ihi: usize,
) {
    for j in ilo..=ihi {
        // Powers of the radix again, so `1 / s` is exact.
        let row_factor = D::from_real(scale[j]);
        let column_factor = D::from_real(1.0 / scale[j]);
        for i in 0..order {
            matrix[j + order * i] = matrix[j + order * i] * row_factor;
        }
        for i in 0..order {
            matrix[i + order * j] = matrix[i + order * j] * column_factor;
        }
    }
    for (j, &partner) in scale[..ilo].iter().enumerate().rev() {
        row_column_swap(matrix, order, j, partner as usize - 1);
    }
    for (j, &partner) in scale.iter().enumerate().skip(ihi + 1) {
        row_column_swap(matrix, order, j, partner as usize - 1);
    }
}

/// Julia's `rcswap!`: swap rows `i` and `j` and columns `i` and `j`.
fn row_column_swap<D: Copy>(matrix: &mut [D], order: usize, i: usize, j: usize) {
    if i == j {
        return;
    }
    for k in 0..order {
        matrix.swap(k + order * i, k + order * j);
    }
    for k in 0..order {
        matrix.swap(i + order * k, j + order * k);
    }
}

/// `destination = Σ_k coefficient_k * term_k` over `order x order` blocks.
fn combine<D: FactorScalar>(destination: &mut [D], terms: &[(f64, &[D])], order: usize) {
    let elements = order * order;
    for index in 0..elements {
        let mut value = D::zero();
        for (coefficient, term) in terms {
            value = value + D::from_real(*coefficient) * term[index];
        }
        destination[index] = value;
    }
}

/// `destination += Σ_k coefficient_k * term_k + diagonal * I`.
fn add_terms<D: FactorScalar>(
    destination: &mut [D],
    terms: &[(f64, &[D])],
    diagonal: f64,
    order: usize,
) {
    let elements = order * order;
    for index in 0..elements {
        let mut value = destination[index];
        for (coefficient, term) in terms {
            value = value + D::from_real(*coefficient) * term[index];
        }
        destination[index] = value;
    }
    let diagonal = D::from_real(diagonal);
    for index in 0..order {
        destination[index + order * index] = destination[index + order * index] + diagonal;
    }
}

/// `destination = lhs * rhs` on column-major `order x order` blocks.
fn gemm<E, D>(
    dense: &mut E,
    destination: &mut [D],
    lhs: &[D],
    rhs: &[D],
    order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let elements = order * order;
    let shape = [order, order];
    let strides = [1usize, order];
    let lhs =
        DenseView::new(&lhs[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let rhs =
        DenseView::new(&rhs[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let destination = DenseViewMut::new(&mut destination[..elements], &shape, &strides, 0)
        .map_err(OperationError::Dense)?;
    dense
        .matmul_into(
            D::dense_write(destination),
            D::dense_read(lhs),
            D::dense_read(rhs),
        )
        .map_err(OperationError::Dense)
}

/// Solves `matrix * solution = rhs` on column-major `order x order` blocks.
fn solve<E, D>(
    dense: &mut E,
    matrix: &[D],
    rhs: &[D],
    solution: &mut [D],
    order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let elements = order * order;
    let shape = [order, order];
    let strides = [1usize, order];
    let matrix =
        DenseView::new(&matrix[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let rhs =
        DenseView::new(&rhs[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let solution = DenseViewMut::new(&mut solution[..elements], &shape, &strides, 0)
        .map_err(OperationError::Dense)?;
    dense
        .solve_into(
            D::dense_read(matrix),
            D::dense_read(rhs),
            D::dense_write(solution),
        )
        .map_err(OperationError::Dense)
}

#[cfg(test)]
/// Thresholded pseudo-inverse via the compact SVD with an
/// `rcond * sigma_max` cutoff: `t^+ = V S^+ U^H`.
///
/// This is the exact Moore-Penrose inverse of the hard-thresholded
/// effective-rank tensor. It is the Moore-Penrose inverse of the original
/// tensor only when the cutoff discards no genuinely nonzero singular value.
pub(crate) fn pinv<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
    rcond: f64,
) -> Result<BoundTensorMap<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = pinv_into_mf(dense, &input.dynamic(), rcond)?;
    typed_from_bound_factor(out)
}

fn validate_pinv_rcond(rcond: f64) -> Result<(), OperationError> {
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(OperationError::InvalidArgument {
            message: "pinv rcond must be finite and non-negative",
        });
    }
    Ok(())
}

#[cfg(test)]
/// True inverse of a nonsingular map between isomorphic spaces.
///
/// The context parameter is retained for source compatibility. Inverse itself
/// is context-free and performs one dense solve per nonempty coupled sector.
pub(crate) fn inv<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<BoundTensorMap<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + symtn_tensors::RecouplingCoefficientAction<f64>,
{
    let _ = context;
    let out = inv_into_mf(dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Context-free inverse execution into a caller-admitted swapped output space,
/// in every fusion mode: the sector routing and one dense LU solve per
/// coupled sector.
///
/// # Preconditions
///
/// The caller must have checked `codomain ≅ domain` per coupled sector,
/// through the mode's factor-space authority
/// ([`FactorSpaceAuthority::isomorphic`](crate::seam::FactorSpaceAuthority::isomorphic));
/// the facade is that admitting caller. A violation is not reliably detected
/// here: a non-square stored block is refused, but a coupled sector stored
/// on only one side is not, and the result is then the inverse on the stored
/// sector intersection rather than an error.
#[doc(hidden)]
pub fn inv_direct_into_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    inverse_by_sector_dyn_into(dense, input, output_space)
}

/// Context-free pseudo-inverse into a caller-admitted swapped output space, in
/// every fusion mode.
///
/// # Preconditions
///
/// None on the operand's shape: a pseudo-inverse is defined for every map, so
/// no isomorphism is assumed. `output_space` must be the operand's swapped
/// space on its provider, which this seam validates.
#[doc(hidden)]
pub fn pinv_direct_into_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    validate_pinv_rcond(rcond)?;
    pinv_by_sector_dyn_into(dense, input, output_space, rcond)
}

/// Context-free pseudo-inverse of the logical adjoint of `parent`, read in
/// place, into a caller-admitted output space (`parent`'s hom space).
#[doc(hidden)]
pub fn pinv_adjoint_parent_direct_into_dyn<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    output_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    validate_pinv_rcond(rcond)?;
    pinv_adjoint_by_sector_dyn_into(dense, parent, output_space, rcond)
}

/// Context-free dynamic-rank left solve into a caller-admitted output space.
///
/// Categorical preflight and destination admission belong to the caller; this
/// seam validates the rule identities, the codomains and the complete storage
/// route before allocating output or executing any dense solve.
///
/// # Preconditions
///
/// The caller must have checked that the divisor's `codomain ≅ domain` per
/// coupled sector, through the mode's factor-space authority
/// ([`FactorSpaceAuthority::isomorphic`](crate::seam::FactorSpaceAuthority::isomorphic));
/// the facade is that admitting caller. A violation is not reliably detected
/// here: a non-square stored divisor block is refused, but a coupled sector
/// stored on only one side is not, and the result is then the solve on the
/// stored sector intersection rather than an error.
#[doc(hidden)]
pub fn solve_left_direct_into_dyn<E, R, D>(
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
    output_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    solve_left_by_sector_dyn_into(dense, divisor, rhs, output_space)
}

// Test fixtures: each seam into the multiplicity-free layout derived from the
// operation's hom space, with no categorical preflight — that belongs to the
// facade (#1995); the `_into` seams validate their own operands and output.

#[cfg(test)]
fn mf_output<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    homspace: symtn_core::FusionTreeHomSpace,
) -> Result<symtn_tensors::BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    crate::factorize::multiplicity_free_output_space(input.space(), homspace)
}

#[cfg(test)]
fn swapped(homspace: &symtn_core::FusionTreeHomSpace) -> symtn_core::FusionTreeHomSpace {
    symtn_core::FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone())
}

#[cfg(test)]
pub(crate) fn exp_into_mf<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let output = mf_output(input, input.space().space().homspace().clone())?;
    exp_direct_into_dyn(dense, input, output)
}

#[cfg(test)]
pub(crate) fn inv_into_mf<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let output = mf_output(input, swapped(input.space().space().homspace()))?;
    inv_direct_into_dyn(dense, input, output)
}

#[cfg(test)]
pub(crate) fn pinv_into_mf<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let output = mf_output(input, swapped(input.space().space().homspace()))?;
    pinv_direct_into_dyn(dense, input, output, rcond)
}

#[cfg(test)]
pub(crate) fn pinv_adjoint_parent_into_mf<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let output = mf_output(parent, parent.space().space().homspace().clone())?;
    pinv_adjoint_parent_direct_into_dyn(dense, parent, output, rcond)
}

#[cfg(test)]
pub(crate) fn solve_left_into_mf<E, R, D>(
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let output = mf_output(
        divisor,
        symtn_core::FusionTreeHomSpace::new(
            divisor.space().space().homspace().domain().clone(),
            rhs.space().space().homspace().domain().clone(),
        ),
    )?;
    solve_left_direct_into_dyn(dense, divisor, rhs, output)
}
