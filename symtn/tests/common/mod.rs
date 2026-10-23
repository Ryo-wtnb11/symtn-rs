//! Literal lazy-adjoint oracles shared by the multiplicity-free and checked
//! Generic integration tests.
//!
//! Everything here works on public snapshots (fusion trees, block geometry and
//! the flat payload) so the expected values never pass through the production
//! adjoint kernel under test.

#![allow(dead_code)]

use symtn::typed::BlockFusionTrees;

pub struct BlockGeometry {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

pub struct Snapshot<S, D> {
    pub nout: usize,
    pub blocks: Vec<(BlockFusionTrees<S>, BlockGeometry)>,
    pub data: Vec<D>,
}

/// Snapshots a `TensorMap` through its public block API.
#[macro_export]
macro_rules! snapshot {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        $crate::common::Snapshot {
            nout: tensor.codomain_rank(),
            blocks: (0..tensor.subblock_count())
                .map(|index| {
                    let block = tensor.subblock(index).unwrap();
                    (
                        tensor.subblock_fusion_trees(index).unwrap(),
                        $crate::common::BlockGeometry {
                            shape: block.shape().to_vec(),
                            strides: block.strides().to_vec(),
                            offset: block.offset(),
                        },
                    )
                })
                .collect(),
            data: tensor.materialize().unwrap().dense_data().unwrap().to_vec(),
        }
    }};
}

/// Explicit odometer over a dense rectangle, innermost axis first.
pub fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        visit(&index);
        let mut axis = 0;
        loop {
            if axis == shape.len() {
                return;
            }
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}

pub fn linear(geometry: &BlockGeometry, index: &[usize]) -> usize {
    geometry.offset
        + index
            .iter()
            .zip(&geometry.strides)
            .map(|(&i, &s)| i * s)
            .sum::<usize>()
}

/// `parent` is the dagger of `lazy`'s block: codomain and domain trees swap.
pub fn is_adjoint_pair<S: PartialEq>(
    lazy: &BlockFusionTrees<S>,
    parent: &BlockFusionTrees<S>,
) -> bool {
    lazy.coupled() == parent.coupled()
        && lazy.codomain_uncoupled() == parent.domain_uncoupled()
        && lazy.codomain_innerlines() == parent.domain_innerlines()
        && lazy.codomain_vertices() == parent.domain_vertices()
        && lazy.domain_uncoupled() == parent.codomain_uncoupled()
        && lazy.domain_innerlines() == parent.codomain_innerlines()
        && lazy.domain_vertices() == parent.codomain_vertices()
}

/// The pre-#1201 adjoint kernel as a literal formula:
/// `L[b, x] = conj(P[s(b), pi(x)])` where `s` swaps the codomain/domain trees
/// and `pi` swaps the dense axis halves (`x[..nin]` addresses the parent's
/// domain axes, `x[nin..]` its codomain axes).
pub fn literal_adjoint_payload<S: PartialEq, D: Copy + Default>(
    parent: &Snapshot<S, D>,
    lazy: &Snapshot<S, D>,
    conj: impl Fn(D) -> D,
) -> Vec<D> {
    // The lazy codomain is the parent domain: the split of `x` is at the
    // parent's `nin == lazy.nout`.
    let parent_nout = parent.nout;
    let split = lazy.nout;
    let mut result = vec![D::default(); lazy.data.len()];
    for (trees, geometry) in &lazy.blocks {
        let (_, source) = parent
            .blocks
            .iter()
            .find(|(candidate, _)| is_adjoint_pair(trees, candidate))
            .expect("every lazy block has exactly one swapped parent block");
        for_each_index(&geometry.shape, |x| {
            let source_position = source.offset
                + x[split..]
                    .iter()
                    .zip(&source.strides[..parent_nout])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + x[..split]
                    .iter()
                    .zip(&source.strides[parent_nout..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            result[linear(geometry, x)] = conj(parent.data[source_position]);
        });
    }
    result
}

/// Blockwise equality by fusion trees, independent of block order or layout.
pub fn assert_same_tensor<S: PartialEq + std::fmt::Debug, D: Copy + std::fmt::Debug>(
    actual: &Snapshot<S, D>,
    expected: &Snapshot<S, D>,
    close: impl Fn(D, D) -> bool,
) {
    assert_eq!(actual.nout, expected.nout);
    assert_eq!(actual.blocks.len(), expected.blocks.len());
    for (trees, geometry) in &actual.blocks {
        let (_, other) = expected
            .blocks
            .iter()
            .find(|(candidate, _)| candidate == trees)
            .unwrap_or_else(|| panic!("missing block {trees:?}"));
        assert_eq!(geometry.shape, other.shape, "block {trees:?}");
        for_each_index(&geometry.shape, |x| {
            let a = actual.data[linear(geometry, x)];
            let e = expected.data[linear(other, x)];
            assert!(close(a, e), "block {trees:?} at {x:?}: {a:?} != {e:?}");
        });
    }
}

/// TensorKit `tr`: `sum_c dim(c) * tr(b_c)` over the blocks whose codomain and
/// domain trees coincide, pairing codomain axis `i` with domain axis `nout+i`.
pub fn literal_weighted_trace<S: PartialEq, D: Copy>(
    tensor: &Snapshot<S, D>,
    dim: impl Fn(&S) -> f64,
    mut accumulate: impl FnMut(D, f64),
) {
    for (trees, geometry) in &tensor.blocks {
        if trees.codomain_uncoupled() != trees.domain_uncoupled()
            || trees.codomain_innerlines() != trees.domain_innerlines()
            || trees.codomain_vertices() != trees.domain_vertices()
        {
            continue;
        }
        let weight = dim(trees.coupled());
        for_each_index(&geometry.shape[..tensor.nout], |i| {
            let position = geometry.offset
                + i.iter()
                    .enumerate()
                    .map(|(axis, &c)| {
                        c * (geometry.strides[axis] + geometry.strides[tensor.nout + axis])
                    })
                    .sum::<usize>();
            accumulate(tensor.data[position], weight);
        });
    }
}

/// Permutes the axes of a row-major dense array: axis `k` of the output is
/// axis `perm[k]` of the input.
///
/// This is the whole of the physical-basis oracle for a bosonic permute that
/// stays within the codomain and within the domain — the reduced-block replay
/// may have to recouple to produce it, but the physical array only moves.
/// Shared so the Host pin (`typed_transform_host_side.rs`) and the device
/// gate (`typed_cuda_transform.rs`) compare against one definition.
pub fn permute_dense<D: Copy>(shape: &[usize], data: &[D], perm: &[usize]) -> (Vec<usize>, Vec<D>) {
    let out_shape: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
    let strides = |shape: &[usize]| {
        let mut strides = vec![1usize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        strides
    };
    let in_strides = strides(shape);
    let out_strides = strides(&out_shape);
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        let mut source = 0usize;
        for (axis, &source_axis) in perm.iter().enumerate() {
            let index = (flat / out_strides[axis]) % out_shape[axis].max(1);
            source += index * in_strides[source_axis];
        }
        out.push(data[source]);
    }
    (out_shape, out)
}

// ---------------------------------------------------------------------------
// Device payload harness (`cuda` only)
// ---------------------------------------------------------------------------

/// Every admitted device payload dtype, with the comparisons the payload
/// suites need.
///
/// It lives here rather than in one test binary because several suites are
/// generic over exactly this set — the device single-precision suites (#1336,
/// #1341), the device general contraction (#1345) and its ungated Host oracle
/// pin — and a second copy is a second place for `EPS` or a conversion to
/// drift. It needs no device, so a Host pin can share it; [`DevicePayload`]
/// adds the device bound.
///
/// `entry` never receives an un-suffixed float literal in a single-precision
/// context: the caller passes `f64` and each impl performs its own conversion,
/// which is also what keeps the fixtures dyadic and therefore exact.
pub trait Payload:
    symtn::typed::TensorScalar + Copy + PartialEq + std::fmt::Debug + crate::numerics::Numeric
{
    const NAME: &'static str;

    fn entry(real: f64, imaginary: f64) -> Self;
    fn parts(self) -> (f64, f64);

    fn distance(self, other: Self) -> f64 {
        let (ar, ai) = self.parts();
        let (br, bi) = other.parts();
        ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt()
    }

    fn magnitude(self) -> f64 {
        let (re, im) = self.parts();
        (re * re + im * im).sqrt()
    }
}

impl Payload for f64 {
    const NAME: &'static str = "f64";

    fn entry(real: f64, _imaginary: f64) -> Self {
        real
    }

    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

impl Payload for num_complex::Complex64 {
    const NAME: &'static str = "c64";

    fn entry(real: f64, imaginary: f64) -> Self {
        Self::new(real, imaginary)
    }

    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

impl Payload for f32 {
    const NAME: &'static str = "f32";

    fn entry(real: f64, _imaginary: f64) -> Self {
        real as f32
    }

    fn parts(self) -> (f64, f64) {
        (f64::from(self), 0.0)
    }
}

impl Payload for num_complex::Complex32 {
    const NAME: &'static str = "c32";

    fn entry(real: f64, imaginary: f64) -> Self {
        Self::new(real as f32, imaginary as f32)
    }

    fn parts(self) -> (f64, f64) {
        (f64::from(self.re), f64::from(self.im))
    }
}

/// A [`Payload`] the device admits.
#[cfg(feature = "cuda")]
pub trait DevicePayload: Payload + symtn::typed::CudaPayload {}

#[cfg(feature = "cuda")]
impl<D: Payload + symtn::typed::CudaPayload> DevicePayload for D {}

/// Symmetry providers the device generic bodies are instantiated over.
#[cfg(feature = "cuda")]
pub trait DeviceRule:
    symtn::sector::TypedSectorAdmission<
        Sector = <Self as symtn::sector::SectorCodec>::Sector,
        Error = symtn::typed::FusionAlgebraError,
        Mode = symtn::sector::MultiplicityFreeAdmissionMode,
    > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
    + symtn::sector::CheckedFusionAlgebra
    + symtn::sector::SectorCodec
{
}

#[cfg(feature = "cuda")]
impl<R> DeviceRule for R where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec
{
}
