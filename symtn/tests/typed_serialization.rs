use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use symtn::typed::Engine;
use symtn::typed::HermitianTol;

use symtn::sector::TypedSectorAdmission;
use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::__network::NetworkReuseClass;
use symtn::typed::{Complex32, Complex64, Runtime};
use symtn::typed::{
    DecodeError, DecodeLimits, GradedSpace, PersistedScalar, SectorSpectrum, TensorMap,
    TypedPersistenceCodec,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodecError {
    MissingProvider,
    InvalidSector,
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CodecError {}

struct Su2Codec {
    provider: Arc<SU2FusionRule>,
    resolve_calls: AtomicUsize,
    missing_provider: bool,
    accept_foreign_key: bool,
}

impl Su2Codec {
    fn new(provider: Arc<SU2FusionRule>) -> Self {
        Self {
            provider,
            resolve_calls: AtomicUsize::new(0),
            missing_provider: false,
            accept_foreign_key: false,
        }
    }

    fn missing(provider: Arc<SU2FusionRule>) -> Self {
        Self {
            missing_provider: true,
            ..Self::new(provider)
        }
    }

    fn mismatched(provider: Arc<SU2FusionRule>) -> Self {
        Self {
            accept_foreign_key: true,
            ..Self::new(provider)
        }
    }
}

impl TypedPersistenceCodec<SU2FusionRule> for Su2Codec {
    type Error = CodecError;

    fn provider_key(&self, _provider: &SU2FusionRule) -> Result<Vec<u8>, Self::Error> {
        Ok(b"su2".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<SU2FusionRule>, Self::Error> {
        self.resolve_calls.fetch_add(1, Ordering::Relaxed);
        if self.missing_provider || (key != b"su2" && !self.accept_foreign_key) {
            Err(CodecError::MissingProvider)
        } else {
            Ok(Arc::clone(&self.provider))
        }
    }

    fn encode_sector(
        &self,
        _provider: &SU2FusionRule,
        sector: &SU2Irrep,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![sector.twice_spin() as u8])
    }

    fn decode_sector(
        &self,
        _provider: &SU2FusionRule,
        bytes: &[u8],
    ) -> Result<SU2Irrep, Self::Error> {
        let &[twice_spin] = bytes else {
            return Err(CodecError::InvalidSector);
        };
        SU2Irrep::try_from_twice_spin(twice_spin as usize).ok_or(CodecError::InvalidSector)
    }
}

struct U1Codec {
    provider: Arc<U1FusionRule>,
}

impl TypedPersistenceCodec<U1FusionRule> for U1Codec {
    type Error = CodecError;

    fn provider_key(&self, _provider: &U1FusionRule) -> Result<Vec<u8>, Self::Error> {
        Ok(b"u1".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<U1FusionRule>, Self::Error> {
        if key == b"u1" {
            Ok(Arc::clone(&self.provider))
        } else {
            Err(CodecError::MissingProvider)
        }
    }

    fn encode_sector(
        &self,
        _provider: &U1FusionRule,
        sector: &U1Irrep,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(sector.charge().to_le_bytes().to_vec())
    }

    fn decode_sector(
        &self,
        _provider: &U1FusionRule,
        bytes: &[u8],
    ) -> Result<U1Irrep, Self::Error> {
        let bytes: [u8; 4] = bytes.try_into().map_err(|_| CodecError::InvalidSector)?;
        U1Irrep::try_new(i32::from_le_bytes(bytes)).ok_or(CodecError::InvalidSector)
    }
}

use crate::toy_rules;

use toy_rules::{GenericError, GenericLabel, GenericToy};

struct GenericCodec {
    provider: Arc<GenericToy>,
}

impl TypedPersistenceCodec<GenericToy> for GenericCodec {
    type Error = GenericError;

    fn provider_key(&self, _provider: &GenericToy) -> Result<Vec<u8>, Self::Error> {
        Ok(b"generic-toy".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<GenericToy>, Self::Error> {
        if key == b"generic-toy" {
            Ok(Arc::clone(&self.provider))
        } else {
            Err(GenericError)
        }
    }

    fn encode_sector(
        &self,
        _provider: &GenericToy,
        sector: &GenericLabel,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![match sector {
            GenericLabel::Vacuum => 0,
            GenericLabel::X => 1,
        }])
    }

    fn decode_sector(
        &self,
        _provider: &GenericToy,
        bytes: &[u8],
    ) -> Result<GenericLabel, Self::Error> {
        match bytes {
            [0] => Ok(GenericLabel::Vacuum),
            [1] => Ok(GenericLabel::X),
            _ => Err(GenericError),
        }
    }
}

use crate::fixtures;

use fixtures::host_runtime;

fn su2_leg(provider: &Arc<SU2FusionRule>, dual: bool) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .and_then(|space| if dual { space.try_dual() } else { Ok(space) })
    .unwrap()
}

fn u64_at(bytes: &[u8], offset: usize) -> usize {
    usize::try_from(u64::from_le_bytes(
        bytes[offset..offset + 8].try_into().unwrap(),
    ))
    .unwrap()
}

fn header_end(bytes: &[u8]) -> usize {
    20 + u64_at(bytes, 12)
}

fn dense_leg_degeneracy_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut position = header_end(bytes) + 1;
    let rank = u64_at(bytes, position) + u64_at(bytes, position + 8);
    position += 16;
    let mut offsets = Vec::new();
    for _ in 0..rank {
        position += 1;
        let sectors = u64_at(bytes, position);
        position += 8;
        for _ in 0..sectors {
            let label_len = u64_at(bytes, position);
            position += 8 + label_len;
            offsets.push(position);
            position += 8;
        }
    }
    offsets
}

#[test]
fn checked_generic_multiplicity_keys_payload_and_resolver_arc_roundtrip() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(GenericToy);
    let codec = GenericCodec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(GenericLabel::X, 1)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |trees, indices| {
            let coupled = usize::from(*trees.coupled() == GenericLabel::X);
            let codomain_mu = trees.codomain_vertices()[0].get();
            let domain_mu = trees.domain_vertices()[0].get();
            (1000 * coupled + 100 * codomain_mu + 10 * domain_mu + indices.iter().sum::<usize>())
                as f64
        },
    )
    .unwrap();

    let keys = (0..source.subblock_count())
        .map(|index| source.subblock_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    let mu_one = keys
        .iter()
        .find(|key| {
            key.coupled() == &GenericLabel::X
                && key.codomain_vertices()[0].get() == 1
                && key.domain_vertices()[0].get() == 1
        })
        .unwrap();
    let mu_two = keys
        .iter()
        .find(|key| {
            key.coupled() == &GenericLabel::X
                && key.codomain_vertices()[0].get() == 2
                && key.domain_vertices()[0].get() == 1
        })
        .unwrap();
    assert_eq!(mu_one.coupled(), mu_two.coupled());
    assert_eq!(mu_one.codomain_uncoupled(), mu_two.codomain_uncoupled());
    assert_eq!(mu_one.codomain_innerlines(), mu_two.codomain_innerlines());
    assert_eq!(mu_one.domain_uncoupled(), mu_two.domain_uncoupled());
    assert_eq!(mu_one.domain_innerlines(), mu_two.domain_innerlines());
    assert_eq!(mu_one.domain_vertices(), mu_two.domain_vertices());
    assert_eq!(mu_one.codomain_vertices()[0].get(), 1);
    assert_eq!(mu_two.codomain_vertices()[0].get(), 2);

    let bytes = source.to_bytes_with(&codec).unwrap();
    let restored = TensorMap::<GenericToy, f64>::from_bytes_with(
        engine,
        &runtime,
        &bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    for space in restored.codomain().iter().chain(restored.domain().iter()) {
        assert!(std::ptr::eq(space.provider(), provider.as_ref()));
    }
    assert_eq!(
        restored.dense_data().unwrap().len(),
        source.dense_data().unwrap().len()
    );
    assert!(restored
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    assert_eq!(restored.subblock_count(), source.subblock_count());
    for index in 0..source.subblock_count() {
        assert_eq!(
            restored.subblock_fusion_trees(index).unwrap(),
            source.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            restored.subblock(index).unwrap(),
            source.subblock(index).unwrap()
        );
    }

    let complex = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, -0.25))
        .unwrap();
    let restored_complex = TensorMap::<GenericToy, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &complex.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(restored_complex.provider(), provider.as_ref()));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_complex, false),
        NetworkReuseClass::OwnedDense
    ));
    assert!(restored_complex
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));

    let complex_lazy = complex.adjoint(engine).unwrap();
    let restored_complex_lazy = TensorMap::<GenericToy, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &complex_lazy.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(
        restored_complex_lazy.provider(),
        provider.as_ref()
    ));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_complex_lazy, false),
        NetworkReuseClass::LazyAdjoint
    ));
    assert!(restored_complex_lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(
            complex_lazy
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        )
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));
    let source_adjoint = &source.adjoint(engine).unwrap();

    let hermitian = source.axpby(engine, 1.0, source_adjoint, 1.0).unwrap();
    let factor = hermitian
        .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap()
        .d;
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&factor, false),
        NetworkReuseClass::Compact
    ));
    // TensorKit `adjoint(::DiagonalTensorMap)` is again a diagonal.
    let factor_adjoint = factor.adjoint(engine).unwrap();
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&factor_adjoint, false),
        NetworkReuseClass::Compact
    ));
    let restored_adjoint = TensorMap::<GenericToy, f64>::from_bytes_with(
        engine,
        &runtime,
        &factor_adjoint.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(restored_adjoint.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::network_reuse_class(&restored_adjoint, false)
            == symtn::typed::__network::network_reuse_class(&factor_adjoint, false)
    );
    assert_eq!(
        restored_adjoint
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        factor_adjoint
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(
            &restored_adjoint.adjoint(engine).unwrap(),
            false
        ),
        NetworkReuseClass::Compact
    ));
    let complex_adjoint = &complex.adjoint(engine).unwrap();

    let complex_hermitian = complex
        .axpby(
            engine,
            Complex64::new(1.0, 0.0),
            complex_adjoint,
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
    let complex_factor = complex_hermitian
        .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap()
        .d;
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&complex_factor, false),
        NetworkReuseClass::Compact
    ));
    let restored_complex_factor = TensorMap::<GenericToy, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &complex_factor.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(
        restored_complex_factor.provider(),
        provider.as_ref()
    ));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_complex_factor, false),
        NetworkReuseClass::Compact
    ));
    assert!(restored_complex_factor
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(
            complex_factor
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        )
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));

    let complex_factor_lazy = complex_factor.adjoint(engine).unwrap();
    let restored_complex_factor_lazy = TensorMap::<GenericToy, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &complex_factor_lazy.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(
        restored_complex_factor_lazy.provider(),
        provider.as_ref()
    ));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_complex_factor_lazy, false),
        NetworkReuseClass::Compact
    ));
    assert!(restored_complex_factor_lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(
            complex_factor_lazy
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        )
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(
            &restored_complex_factor_lazy.adjoint(engine).unwrap(),
            false
        ),
        NetworkReuseClass::Compact
    ));
}

#[test]
fn admitted_shape_limit_precedes_dense_payload_allocation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(GenericToy);
    let codec = GenericCodec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(GenericLabel::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |_, _| 1.0)
            .unwrap();
    let mut forged = source.to_bytes_with(&codec).unwrap();
    let offsets = dense_leg_degeneracy_offsets(&forged);
    assert_eq!(offsets.len(), 4);
    for offset in offsets {
        forged[offset..offset + 8].copy_from_slice(&1024u64.to_le_bytes());
    }

    let limit = source.dense_data().unwrap().len();
    assert!(matches!(
        TensorMap::<GenericToy, f64>::from_bytes_with(engine,
            &runtime,
            &forged,
            DecodeLimits {
                max_elements: limit,
                ..DecodeLimits::default()
            },
            &codec,
        ),
        Err(DecodeError::LimitExceeded {
            resource: "admitted dense elements",
            actual,
            limit: observed_limit,
        }) if actual > observed_limit && observed_limit == limit
    ));
}

#[test]
fn graded_space_roundtrip_keeps_duality_exact_arc_and_deterministic_bytes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let source = su2_leg(&provider, true);

    let bytes = source.to_bytes_with(&codec).unwrap();
    assert_eq!(source.to_bytes_with(&codec).unwrap(), bytes);
    let restored = GradedSpace::<SU2FusionRule>::from_bytes_with(
        engine,
        &bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert!(restored.is_dual());
    assert_eq!(restored.sectors().unwrap(), source.sectors().unwrap());
    assert_eq!(restored.degeneracies(), source.degeneracies());
}

#[test]
fn non_self_dual_u1_space_roundtrip_does_not_dualize_twice() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let codec = U1Codec {
        provider: Arc::clone(&provider),
    };
    let source = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(3), 7)])
        .and_then(|space| space.try_dual())
        .unwrap();
    assert_eq!(source.sectors().unwrap(), [U1Irrep::new(-3)]);

    let restored = GradedSpace::<U1FusionRule>::from_bytes_with(
        engine,
        &source.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert!(restored.is_dual());
    assert_eq!(restored.sectors().unwrap(), [U1Irrep::new(-3)]);
    assert_eq!(restored.degeneracies(), [7]);
}

#[test]
fn dense_su2_f64_and_c64_roundtrip_exact_bits_and_semantic_blocks() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let codomain = su2_leg(&provider, false);
    let domain = su2_leg(&provider, true);
    let real_bits = [
        0x8000_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        0x7ff8_0000_0000_0042,
    ];
    let real: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&codomain, &codomain, &codomain],
        [&domain],
        |trees, indices| {
            f64::from_bits(
                real_bits[(trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3],
            )
        },
    )
    .unwrap();
    let complex: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&codomain, &codomain, &codomain],
        [&domain],
        |trees, indices| {
            let index = (trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3;
            Complex64::new(
                f64::from_bits(real_bits[index]),
                f64::from_bits(real_bits[(index + 1) % 3]),
            )
        },
    )
    .unwrap();
    assert!((0..real.subblock_count()).any(|index| {
        !real
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_innerlines()
            .is_empty()
    }));

    let real_bytes = real.to_bytes_with(&codec).unwrap();
    let restored_real = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        engine,
        &runtime,
        &real_bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    let complex_bytes = complex.to_bytes_with(&codec).unwrap();
    let restored_complex = TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &complex_bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert_eq!(real.to_bytes_with(&codec).unwrap(), real_bytes);
    assert_eq!(complex.to_bytes_with(&codec).unwrap(), complex_bytes);
    assert_eq!(
        restored_real
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        real.dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        restored_complex
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| (x.re.to_bits(), x.im.to_bits()))
            .collect::<Vec<_>>(),
        complex
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| (x.re.to_bits(), x.im.to_bits()))
            .collect::<Vec<_>>()
    );
    assert!(std::ptr::eq(restored_real.provider(), provider.as_ref()));
    assert_eq!(restored_real.subblock_count(), real.subblock_count());
    for index in 0..real.subblock_count() {
        assert_eq!(
            restored_real.subblock_fusion_trees(index).unwrap(),
            real.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            restored_real.subblock(index).unwrap().shape(),
            real.subblock(index).unwrap().shape()
        );
    }
}

#[test]
fn compact_and_lazy_representations_survive_roundtrip() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let leg = su2_leg(&provider, false);
    let spectrum = vec![
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(0),
            values: vec![f64::from_bits(0x8000_0000_0000_0000)],
        },
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(1),
            values: vec![2.0, f64::from_bits(0x7ff8_0000_0000_0007)],
        },
    ];
    let compact =
        TensorMap::<SU2FusionRule, f64>::diagonal(engine, &runtime, &leg, spectrum).unwrap();
    let restored_compact = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        engine,
        &runtime,
        &compact.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_compact, false),
        NetworkReuseClass::Compact
    ));
    let actual = symtn::expert::diagonal_spectrum(engine, &restored_compact)
        .unwrap()
        .unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].values[0].to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(actual[1].values[1].to_bits(), 0x7ff8_0000_0000_0007);

    let complex_spectrum = vec![
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(0),
            values: vec![Complex64::new(f64::from_bits(0x8000_0000_0000_0000), 1.0)],
        },
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(1),
            values: vec![
                Complex64::new(2.0, -3.0),
                Complex64::new(4.0, f64::from_bits(0x7ff8_0000_0000_0007)),
            ],
        },
    ];
    let compact_complex =
        TensorMap::<SU2FusionRule, Complex64>::diagonal(engine, &runtime, &leg, complex_spectrum)
            .unwrap();
    let restored_compact_complex = TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &compact_complex.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(
        restored_compact_complex.provider(),
        provider.as_ref()
    ));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_compact_complex, false),
        NetworkReuseClass::Compact
    ));
    assert!(restored_compact_complex
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(
            compact_complex
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        )
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));

    // Multiplicity-free compact adjoint is an owned conjugated compact value,
    // not a lazy parent-backed view.
    let compact_complex_adjoint = compact_complex.adjoint(engine).unwrap();
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&compact_complex_adjoint, false),
        NetworkReuseClass::Compact
    ));
    let restored_compact_adjoint = TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &compact_complex_adjoint.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(
        restored_compact_adjoint.provider(),
        provider.as_ref()
    ));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_compact_adjoint, false),
        NetworkReuseClass::Compact
    ));
    assert!(restored_compact_adjoint
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(
            compact_complex_adjoint
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        )
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));

    let dense: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            (1 + indices.iter().sum::<usize>()) as f64
        })
        .unwrap();
    let lazy = dense.adjoint(engine).unwrap();
    let restored_lazy = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        engine,
        &runtime,
        &lazy.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_lazy, false),
        NetworkReuseClass::LazyAdjoint
    ));
    assert_eq!(
        restored_lazy
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        lazy.materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );

    let dense_complex = dense
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.5))
        .unwrap();
    let lazy_complex = dense_complex.adjoint(engine).unwrap();
    let restored_lazy_complex = TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &lazy_complex.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(
        restored_lazy_complex.provider(),
        provider.as_ref()
    ));
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&restored_lazy_complex, false),
        NetworkReuseClass::LazyAdjoint
    ));
    assert!(restored_lazy_complex
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(
            lazy_complex
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        )
        .all(|(actual, expected)| {
            (actual.re.to_bits(), actual.im.to_bits())
                == (expected.re.to_bits(), expected.im.to_bits())
        }));
}

/// A lazy-adjoint record names only a dense parent: the encoder never writes a
/// diagonal parent, because the adjoint of a compact diagonal is itself an
/// owned compact diagonal.
#[test]
fn lazy_adjoint_record_with_a_diagonal_parent_is_rejected() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let diagonal = TensorMap::<_, Complex64>::diagonal(
        engine,
        &runtime,
        &su2_leg(&provider, false),
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![Complex64::new(2.0, 1.5)],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(1),
                values: vec![Complex64::new(2.0, 1.5), Complex64::new(-3.0, -0.5)],
            },
        ],
    )
    .unwrap();
    let mut bytes = diagonal.to_bytes_with(&codec).unwrap();
    let repr = header_end(&bytes);
    assert_eq!(bytes[repr], 2, "diagonal record tag");
    bytes.insert(repr, 3);
    assert!(
        matches!(TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(engine,
            &runtime,
            &bytes,
            DecodeLimits::default(),
            &codec,
        ), Err(DecodeError::InvalidFormat(m)) if m.contains("unknown lazy-adjoint parent representation"))
    );
}

#[test]
fn malformed_framing_limits_and_provider_failures_are_typed_and_ordered() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let bytes = su2_leg(&provider, true).to_bytes_with(&codec).unwrap();
    let mut decode = |input: &[u8], limits, codec: &Su2Codec| {
        GradedSpace::<SU2FusionRule>::from_bytes_with(engine, input, limits, codec)
    };

    for (offset, value) in [(0, b'X'), (10, 2), (11, 1)] {
        let mut malformed = bytes.clone();
        malformed[offset] = value;
        assert!(
            matches!(decode(&malformed, DecodeLimits::default(), &codec), Err(DecodeError::InvalidFormat(m)) if m.contains(match offset { 0 => "bad magic", 10 => "wrong object kind", 11 => "wrong scalar kind", _ => unreachable!() }))
        );
    }
    let mut unsupported = bytes.clone();
    unsupported[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert!(matches!(
        decode(&unsupported, DecodeLimits::default(), &codec),
        Err(DecodeError::UnsupportedVersion {
            actual: 2,
            supported: 1
        })
    ));
    let mut bad_dual = bytes.clone();
    let dual_offset = header_end(&bad_dual);
    bad_dual[dual_offset] = 2;
    assert!(
        matches!(decode(&bad_dual, DecodeLimits::default(), &codec), Err(DecodeError::InvalidFormat(m)) if m.contains("duality flag is not 0 or 1"))
    );
    assert!(
        matches!(decode(&bytes[..bytes.len() - 1], DecodeLimits::default(), &codec), Err(DecodeError::InvalidFormat(m)) if m.contains("truncated input"))
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(
        matches!(decode(&trailing, DecodeLimits::default(), &codec), Err(DecodeError::InvalidFormat(m)) if m.contains("trailing bytes"))
    );

    let limits = DecodeLimits {
        max_bytes: bytes.len() - 1,
        ..DecodeLimits::default()
    };
    codec.resolve_calls.store(0, Ordering::Relaxed);
    assert!(matches!(
        decode(&bytes, limits, &codec),
        Err(DecodeError::LimitExceeded { .. })
    ));
    assert_eq!(codec.resolve_calls.load(Ordering::Relaxed), 0);

    let missing = Su2Codec::missing(Arc::clone(&provider));
    let mut bad_magic = bytes.clone();
    bad_magic[0] = 0;
    assert!(matches!(
        decode(&bad_magic, DecodeLimits::default(), &missing),
        Err(DecodeError::InvalidFormat(_))
    ));
    assert_eq!(missing.resolve_calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        decode(&bytes, DecodeLimits::default(), &missing),
        Err(DecodeError::Codec(CodecError::MissingProvider))
    ));

    let mut mismatched_key = bytes.clone();
    mismatched_key[20..23].copy_from_slice(b"bad");
    let mismatched = Su2Codec::mismatched(Arc::clone(&provider));
    assert!(matches!(
        decode(&mismatched_key, DecodeLimits::default(), &mismatched),
        Err(DecodeError::ProviderMismatch)
    ));
}

#[test]
fn malformed_tensor_tags_and_duplicate_or_missing_blocks_are_rejected() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(SU2Irrep::from_twice_spin(0), 1)]).unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [], |_, _| 3.0).unwrap();
    let bytes = tensor.to_bytes_with(&codec).unwrap();
    let mut decode = |input: &[u8]| {
        TensorMap::<SU2FusionRule, f64>::from_bytes_with(
            engine,
            &runtime,
            input,
            DecodeLimits::default(),
            &codec,
        )
    };

    for (offset, value) in [(10, 1), (11, 0), (11, 99), (header_end(&bytes), 99)] {
        let mut malformed = bytes.clone();
        malformed[offset] = value;
        assert!(
            matches!(decode(&malformed), Err(DecodeError::InvalidFormat(m)) if m.contains(match offset { 10 => "wrong object kind", 11 => "wrong scalar kind", _ => "unknown tensor representation" }))
        );
    }

    // Header + repr + ranks + one leg (dual, count, one-byte label, degeneracy).
    let block_count_offset = header_end(&bytes) + 1 + 16 + 1 + 8 + 8 + 1 + 8;
    assert_eq!(u64_at(&bytes, block_count_offset), 1);
    let first_block = block_count_offset + 8;

    let mut missing = bytes[..first_block].to_vec();
    missing[block_count_offset..first_block].copy_from_slice(&0u64.to_le_bytes());
    assert!(
        matches!(decode(&missing), Err(DecodeError::InvalidFormat(m)) if m.contains("missing an admitted block"))
    );

    let mut duplicate = bytes.clone();
    duplicate[block_count_offset..first_block].copy_from_slice(&2u64.to_le_bytes());
    duplicate.extend_from_slice(&bytes[first_block..]);
    assert!(
        matches!(decode(&duplicate), Err(DecodeError::InvalidFormat(m)) if m.contains("duplicate semantic block key"))
    );

    for limits in [
        DecodeLimits {
            max_rank: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_sectors_per_leg: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_blocks: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_elements: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_blob_bytes: 0,
            ..DecodeLimits::default()
        },
    ] {
        assert!(matches!(
            TensorMap::<SU2FusionRule, f64>::from_bytes_with(
                engine, &runtime, &bytes, limits, &codec
            ),
            Err(DecodeError::LimitExceeded { .. })
        ));
    }
}

const GOLDEN_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/typed_serialization"
);
const GOLDEN_F64_BITS: [u64; 3] = [
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0001,
    0x7ff8_0000_0000_0042,
];

fn golden_dense_f64(
    runtime: &Runtime,
    provider: &Arc<SU2FusionRule>,
) -> TensorMap<SU2FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let codomain = su2_leg(provider, false);
    let domain = su2_leg(provider, true);
    TensorMap::from_subblock_fn(
        engine,
        runtime,
        [&codomain, &codomain],
        [&domain],
        |trees, indices| {
            f64::from_bits(
                GOLDEN_F64_BITS[(trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3],
            )
        },
    )
    .unwrap()
}

fn golden_diagonal_c64(
    runtime: &Runtime,
    provider: &Arc<SU2FusionRule>,
) -> TensorMap<SU2FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let spectrum = vec![
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(0),
            values: vec![Complex64::new(
                f64::from_bits(GOLDEN_F64_BITS[0]),
                f64::from_bits(GOLDEN_F64_BITS[2]),
            )],
        },
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(1),
            values: vec![
                Complex64::new(2.0, -3.0),
                Complex64::new(f64::from_bits(GOLDEN_F64_BITS[1]), -0.0),
            ],
        },
    ];
    TensorMap::diagonal(engine, runtime, &su2_leg(provider, false), spectrum).unwrap()
}

#[test]
fn version_one_golden_files_decode_and_reencode_byte_for_byte() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let golden = |name: &str| std::fs::read(format!("{GOLDEN_DIR}/{name}")).unwrap();
    let mut decode_f64 = |bytes: &[u8]| {
        TensorMap::<SU2FusionRule, f64>::from_bytes_with(
            engine,
            &runtime,
            bytes,
            DecodeLimits::default(),
            &codec,
        )
        .unwrap()
    };

    let dense_bytes = golden("su2_dense_f64_v1.bin");
    let dense = decode_f64(&dense_bytes);
    let expected = golden_dense_f64(&runtime, &provider);
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&dense, false),
        NetworkReuseClass::OwnedDense
    ));
    assert_eq!(
        dense.dense_data().unwrap().bits(),
        expected.dense_data().unwrap().bits()
    );
    assert_eq!(dense.to_bytes_with(&codec).unwrap(), dense_bytes);
    assert_eq!(expected.to_bytes_with(&codec).unwrap(), dense_bytes);

    let adjoint_bytes = golden("su2_adjoint_f64_v1.bin");
    let adjoint = decode_f64(&adjoint_bytes);
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&adjoint, false),
        NetworkReuseClass::LazyAdjoint
    ));
    assert_eq!(
        adjoint
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .bits(),
        expected
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .bits()
    );
    assert_eq!(adjoint.to_bytes_with(&codec).unwrap(), adjoint_bytes);

    let diagonal_bytes = golden("su2_diagonal_c64_v1.bin");
    let diagonal = TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &diagonal_bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    let expected = golden_diagonal_c64(&runtime, &provider);
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&diagonal, false),
        NetworkReuseClass::Compact
    ));
    assert_eq!(
        diagonal
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .bits(),
        expected
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .bits()
    );
    assert_eq!(diagonal.to_bytes_with(&codec).unwrap(), diagonal_bytes);
    assert_eq!(expected.to_bytes_with(&codec).unwrap(), diagonal_bytes);

    let space_bytes = golden("su2_space_v1.bin");
    let space = GradedSpace::<SU2FusionRule>::from_bytes_with(
        engine,
        &space_bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(space.is_dual());
    assert_eq!(space.to_bytes_with(&codec).unwrap(), space_bytes);
}

/// Scalar bit patterns compared exactly: persistence is data movement.
trait Bits {
    fn bits(&self) -> Vec<(u64, u64)>;
}

macro_rules! impl_bits {
    ($real:ty) => {
        impl Bits for [$real] {
            fn bits(&self) -> Vec<(u64, u64)> {
                self.iter().map(|x| (u64::from(x.to_bits()), 0)).collect()
            }
        }
    };
    ($complex:ty, complex) => {
        impl Bits for [$complex] {
            fn bits(&self) -> Vec<(u64, u64)> {
                self.iter()
                    .map(|x| (u64::from(x.re.to_bits()), u64::from(x.im.to_bits())))
                    .collect()
            }
        }
    };
}

impl_bits!(f64);
impl_bits!(f32);
impl_bits!(Complex64, complex);
impl_bits!(Complex32, complex);

/// `-0`, a value one ulp above 1, a quiet NaN with a payload, and a signalling
/// NaN with a payload, as `f32` bits.
const SINGLE_BITS: [u32; 4] = [0x8000_0000, 0x3f80_0001, 0x7fc0_0042, 0x7f80_0007];

fn single_bits(index: usize) -> f32 {
    f32::from_bits(SINGLE_BITS[index % SINGLE_BITS.len()])
}

struct FermionCodec {
    provider: Arc<FermionParityFusionRule>,
}

impl TypedPersistenceCodec<FermionParityFusionRule> for FermionCodec {
    type Error = CodecError;

    fn provider_key(&self, _provider: &FermionParityFusionRule) -> Result<Vec<u8>, Self::Error> {
        Ok(b"fermion-parity".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<FermionParityFusionRule>, Self::Error> {
        if key == b"fermion-parity" {
            Ok(Arc::clone(&self.provider))
        } else {
            Err(CodecError::MissingProvider)
        }
    }

    fn encode_sector(
        &self,
        _provider: &FermionParityFusionRule,
        sector: &Z2Irrep,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![sector.parity()])
    }

    fn decode_sector(
        &self,
        _provider: &FermionParityFusionRule,
        bytes: &[u8],
    ) -> Result<Z2Irrep, Self::Error> {
        match bytes {
            [parity @ (0 | 1)] => Ok(Z2Irrep::new(*parity)),
            _ => Err(CodecError::InvalidSector),
        }
    }
}

/// Round-trips dense, compact-diagonal and lazy-adjoint forms of one payload
/// dtype and checks the restored representation class and every bit.
macro_rules! single_precision_roundtrip {
    ($name:ident, $scalar:ty, $value:expr) => {
        #[test]
        fn $name() {
            let engine = &mut symtn::typed::CpuEngine::serial();
            let runtime = host_runtime();
            let value: fn(usize) -> $scalar = $value;

            fn check<R, C>(
                runtime: &Runtime,
                codec: &C,
                source: &TensorMap<R, $scalar>,
                class: NetworkReuseClass,
            ) where
                R: TypedSectorAdmission,
                C: TypedPersistenceCodec<R>,
                TensorMap<R, $scalar>: RoundTrip<R, C>,
            {
                let engine = &mut symtn_dense::CpuEngine::serial();
                assert!(symtn::typed::__network::network_reuse_class(&source, false) == class);
                let bytes = source.encode(codec);
                assert_eq!(bytes[11], <$scalar as Tag>::TAG);
                let restored = TensorMap::<R, $scalar>::decode(engine, runtime, &bytes, codec);
                assert!(std::ptr::eq(restored.provider(), source.provider()));
                assert!(symtn::typed::__network::network_reuse_class(&restored, false) == class);
                assert_eq!(
                    restored
                        .materialize(engine)
                        .unwrap()
                        .dense_data()
                        .unwrap()
                        .bits(),
                    source
                        .materialize(engine)
                        .unwrap()
                        .dense_data()
                        .unwrap()
                        .bits()
                );
                // The snapshot carries every block's fusion-tree key, so equal
                // re-encoded bytes also prove the block keys were restored.
                assert_eq!(restored.encode(codec), bytes);
            }

            // U(1): non-self-dual legs, so a dualization error would move sectors.
            let provider = Arc::new(U1FusionRule);
            let codec = U1Codec {
                provider: Arc::clone(&provider),
            };
            let leg = GradedSpace::try_new(
                Arc::clone(&provider),
                [
                    (U1Irrep::new(-1), 2),
                    (U1Irrep::new(0), 1),
                    (U1Irrep::new(2), 3),
                ],
            )
            .unwrap();
            let dual = leg.try_dual().unwrap();
            let dense: TensorMap<_, $scalar> = TensorMap::from_subblock_fn(
                engine,
                &runtime,
                [&leg, &dual],
                [&leg],
                |trees, indices| {
                    value(
                        trees.coupled().charge().unsigned_abs() as usize
                            + indices.iter().sum::<usize>(),
                    )
                },
            )
            .unwrap();
            check(&runtime, &codec, &dense, NetworkReuseClass::OwnedDense);
            check(
                &runtime,
                &codec,
                &dense.adjoint(engine).unwrap(),
                NetworkReuseClass::LazyAdjoint,
            );
            let diagonal = TensorMap::<_, $scalar>::diagonal(
                engine,
                &runtime,
                &leg,
                [
                    SectorSpectrum {
                        sector: U1Irrep::new(-1),
                        values: vec![value(0), value(1)],
                    },
                    SectorSpectrum {
                        sector: U1Irrep::new(0),
                        values: vec![value(2)],
                    },
                    SectorSpectrum {
                        sector: U1Irrep::new(2),
                        values: vec![value(3), value(4), value(5)],
                    },
                ],
            )
            .unwrap();
            check(&runtime, &codec, &diagonal, NetworkReuseClass::Compact);
            // Multiplicity-free compact adjoints are owned compact values.
            check(
                &runtime,
                &codec,
                &diagonal.adjoint(engine).unwrap(),
                NetworkReuseClass::Compact,
            );

            // SU(2): nontrivial inner lines.
            let provider = Arc::new(SU2FusionRule);
            let codec = Su2Codec::new(Arc::clone(&provider));
            let codomain = su2_leg(&provider, false);
            let domain = su2_leg(&provider, true);
            let dense: TensorMap<_, $scalar> = TensorMap::from_subblock_fn(
                engine,
                &runtime,
                [&codomain, &codomain, &codomain],
                [&domain],
                |trees, indices| {
                    value(trees.coupled().twice_spin() + indices.iter().sum::<usize>())
                },
            )
            .unwrap();
            assert!((0..dense.subblock_count()).any(|index| {
                !dense
                    .subblock_fusion_trees(index)
                    .unwrap()
                    .codomain_innerlines()
                    .is_empty()
            }));
            check(&runtime, &codec, &dense, NetworkReuseClass::OwnedDense);
            check(
                &runtime,
                &codec,
                &dense.adjoint(engine).unwrap(),
                NetworkReuseClass::LazyAdjoint,
            );
            let diagonal = TensorMap::<_, $scalar>::diagonal(
                engine,
                &runtime,
                &codomain,
                [
                    SectorSpectrum {
                        sector: SU2Irrep::from_twice_spin(0),
                        values: vec![value(0)],
                    },
                    SectorSpectrum {
                        sector: SU2Irrep::from_twice_spin(1),
                        values: vec![value(2), value(3)],
                    },
                ],
            )
            .unwrap();
            check(&runtime, &codec, &diagonal, NetworkReuseClass::Compact);
            check(
                &runtime,
                &codec,
                &diagonal.adjoint(engine).unwrap(),
                NetworkReuseClass::Compact,
            );

            // Fermion parity: odd sectors carry fermionic twists.
            let provider = Arc::new(FermionParityFusionRule);
            let codec = FermionCodec {
                provider: Arc::clone(&provider),
            };
            let leg = GradedSpace::try_new(
                Arc::clone(&provider),
                [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
            )
            .unwrap();
            let dual = leg.try_dual().unwrap();
            let dense: TensorMap<_, $scalar> = TensorMap::from_subblock_fn(
                engine,
                &runtime,
                [&leg, &dual],
                [&leg],
                |trees, indices| {
                    value(usize::from(trees.coupled().parity()) + indices.iter().sum::<usize>())
                },
            )
            .unwrap();
            check(&runtime, &codec, &dense, NetworkReuseClass::OwnedDense);
            check(
                &runtime,
                &codec,
                &dense.adjoint(engine).unwrap(),
                NetworkReuseClass::LazyAdjoint,
            );
            let diagonal = TensorMap::<_, $scalar>::diagonal(
                engine,
                &runtime,
                &leg,
                [
                    SectorSpectrum {
                        sector: Z2Irrep::EVEN,
                        values: vec![value(0), value(1)],
                    },
                    SectorSpectrum {
                        sector: Z2Irrep::ODD,
                        values: vec![value(2), value(3), value(4)],
                    },
                ],
            )
            .unwrap();
            check(&runtime, &codec, &diagonal, NetworkReuseClass::Compact);
            check(
                &runtime,
                &codec,
                &diagonal.adjoint(engine).unwrap(),
                NetworkReuseClass::Compact,
            );

            // Checked Generic: vertex multiplicity; compact adjoints are owned
            // compact values here too.
            let provider = Arc::new(GenericToy);
            let codec = GenericCodec {
                provider: Arc::clone(&provider),
            };
            let leg = GradedSpace::try_new(Arc::clone(&provider), [(GenericLabel::X, 2)]).unwrap();
            let dense: TensorMap<_, $scalar> = TensorMap::from_subblock_fn(
                engine,
                &runtime,
                [&leg, &leg],
                [&leg, &leg],
                |trees, indices| {
                    value(
                        trees.codomain_vertices()[0].get()
                            + trees.domain_vertices()[0].get()
                            + indices.iter().sum::<usize>(),
                    )
                },
            )
            .unwrap();
            assert!((0..dense.subblock_count()).any(|index| {
                dense
                    .subblock_fusion_trees(index)
                    .unwrap()
                    .codomain_vertices()[0]
                    .get()
                    == 2
            }));
            check(&runtime, &codec, &dense, NetworkReuseClass::OwnedDense);
            check(
                &runtime,
                &codec,
                &dense.adjoint(engine).unwrap(),
                NetworkReuseClass::LazyAdjoint,
            );
            let diagonal = TensorMap::<_, $scalar>::diagonal(
                engine,
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: GenericLabel::X,
                    values: vec![value(2), value(3)],
                }],
            )
            .unwrap();
            check(&runtime, &codec, &diagonal, NetworkReuseClass::Compact);
            check(
                &runtime,
                &codec,
                &diagonal.adjoint(engine).unwrap(),
                NetworkReuseClass::Compact,
            );
        }
    };
}

/// Wire tag of each payload dtype, spelled out independently of the library.
trait Tag {
    const TAG: u8;
    const SCALAR: PersistedScalar;
}

impl Tag for f64 {
    const TAG: u8 = 1;
    const SCALAR: PersistedScalar = PersistedScalar::F64;
}
impl Tag for Complex64 {
    const TAG: u8 = 2;
    const SCALAR: PersistedScalar = PersistedScalar::Complex64;
}
impl Tag for f32 {
    const TAG: u8 = 3;
    const SCALAR: PersistedScalar = PersistedScalar::F32;
}
impl Tag for Complex32 {
    const TAG: u8 = 4;
    const SCALAR: PersistedScalar = PersistedScalar::Complex32;
}

/// Unwrapping front end over the per-dtype inherent persistence methods.
trait RoundTrip<R: TypedSectorAdmission, C: TypedPersistenceCodec<R>>: Sized {
    fn encode(&self, codec: &C) -> Vec<u8>;
    fn decode(engine: &mut dyn Engine, runtime: &Runtime, bytes: &[u8], codec: &C) -> Self;
}

macro_rules! impl_round_trip {
    ($scalar:ty, $($rule:ty),+) => {$(
        impl<C: TypedPersistenceCodec<$rule>> RoundTrip<$rule, C> for TensorMap<$rule, $scalar> {
            fn encode(&self, codec: &C) -> Vec<u8> {
                self.to_bytes_with(codec).unwrap()
            }

            fn decode(engine: &mut dyn Engine, runtime: &Runtime, bytes: &[u8], codec: &C) -> Self {
                Self::from_bytes_with(engine, runtime, bytes, DecodeLimits::default(), codec).unwrap()
            }
        }
    )+};
}

impl_round_trip!(
    f32,
    U1FusionRule,
    SU2FusionRule,
    FermionParityFusionRule,
    GenericToy
);
impl_round_trip!(
    Complex32,
    U1FusionRule,
    SU2FusionRule,
    FermionParityFusionRule,
    GenericToy
);

single_precision_roundtrip!(
    f32_roundtrips_every_representation_and_provider_bit_exactly,
    f32,
    single_bits
);
single_precision_roundtrip!(
    complex32_roundtrips_every_representation_and_provider_bit_exactly,
    Complex32,
    |index| Complex32::new(single_bits(index), single_bits(index + 1))
);

/// Decodes `bytes`, written as `$stored`, as `$requested`: only the stored
/// dtype succeeds, and every other request is a typed scalar mismatch rather
/// than a conversion.
macro_rules! assert_decode_as {
    ($engine:ident, $runtime:expr, $codec:expr, $bytes:expr, $stored:ty, $requested:ty) => {{
        let result = TensorMap::<U1FusionRule, $requested>::from_bytes_with($engine,
            $runtime,
            $bytes,
            DecodeLimits::default(),
            $codec,
        );
        if <$requested as Tag>::TAG == <$stored as Tag>::TAG {
            assert!(result.is_ok());
        } else {
            let error = result.err().unwrap();
            assert!(
                matches!(
                    error,
                    DecodeError::ScalarMismatch { actual, expected }
                        if actual == <$stored as Tag>::SCALAR
                            && expected == <$requested as Tag>::SCALAR
                ),
                "{error}"
            );
        }
    }};
}

macro_rules! assert_only_decodes_as {
    ($engine:ident, $runtime:expr, $codec:expr, $bytes:expr, $stored:ty) => {{
        assert_decode_as!($engine, $runtime, $codec, $bytes, $stored, f64);
        assert_decode_as!($engine, $runtime, $codec, $bytes, $stored, Complex64);
        assert_decode_as!($engine, $runtime, $codec, $bytes, $stored, f32);
        assert_decode_as!($engine, $runtime, $codec, $bytes, $stored, Complex32);
    }};
}

#[test]
fn every_scalar_mismatch_direction_is_typed_and_precedes_provider_resolution() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(U1FusionRule);
    let codec = U1Codec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let f64_bytes =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 1.0)
            .unwrap()
            .to_bytes_with(&codec)
            .unwrap();
    let c64_bytes =
        TensorMap::<_, Complex64>::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex64::new(1.0, -0.0)
        })
        .unwrap()
        .to_bytes_with(&codec)
        .unwrap();
    let f32_bytes =
        TensorMap::<_, f32>::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 1.0)
            .unwrap()
            .to_bytes_with(&codec)
            .unwrap();
    let c32_bytes =
        TensorMap::<_, Complex32>::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex32::new(1.0, -0.0)
        })
        .unwrap()
        .to_bytes_with(&codec)
        .unwrap();

    assert_only_decodes_as!(engine, &runtime, &codec, &f64_bytes, f64);
    assert_only_decodes_as!(engine, &runtime, &codec, &c64_bytes, Complex64);
    assert_only_decodes_as!(engine, &runtime, &codec, &f32_bytes, f32);
    assert_only_decodes_as!(engine, &runtime, &codec, &c32_bytes, Complex32);

    // The mismatch is reported from the header, before the resolver runs.
    let missing = Su2Codec::missing(Arc::new(SU2FusionRule));
    let su2_leg = su2_leg(&Arc::new(SU2FusionRule), false);
    let su2: TensorMap<_, f32> =
        TensorMap::from_subblock_fn(engine, &runtime, [&su2_leg], [&su2_leg], |_, _| 1.0).unwrap();
    let su2_bytes = su2
        .to_bytes_with(&Su2Codec::new(Arc::new(SU2FusionRule)))
        .unwrap();
    let error = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        engine,
        &runtime,
        &su2_bytes,
        DecodeLimits::default(),
        &missing,
    )
    .err()
    .unwrap();
    assert!(matches!(
        error,
        DecodeError::ScalarMismatch {
            actual: PersistedScalar::F32,
            expected: PersistedScalar::F64,
        }
    ));
    assert_eq!(missing.resolve_calls.load(Ordering::Relaxed), 0);

    // A tensor snapshot is never a graded space, whatever its scalar tag.
    assert!(
        matches!(GradedSpace::<U1FusionRule>::from_bytes_with(engine, &f32_bytes, DecodeLimits::default(), &codec), Err(DecodeError::InvalidFormat(m)) if m.contains("wrong object kind"))
    );
}

#[test]
fn dual_compact_factor_roundtrip_preserves_space_and_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(U1FusionRule);
    let codec = U1Codec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(2), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-2),
            values: vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
        }],
    )
    .unwrap();
    let qr = input.qr_compact(engine, &[0], &[1]).unwrap();
    let lq = input.lq_full(engine, &[0], &[1]).unwrap();
    let left = input.left_polar(engine, &[0], &[1]).unwrap();
    let right = input.right_polar(engine, &[0], &[1]).unwrap();
    for factor in [qr.q, qr.r, lq.l, lq.q, left.w, left.p, right.p, right.wh] {
        let bytes = factor.to_bytes_with(&codec).unwrap();
        let restored: TensorMap<_, Complex64> =
            TensorMap::<U1FusionRule, Complex64>::from_bytes_with(
                engine,
                &runtime,
                &bytes,
                DecodeLimits::default(),
                &codec,
            )
            .unwrap();
        assert_eq!(restored.codomain(), input.codomain());
        assert_eq!(restored.domain(), input.domain());
        assert_eq!(
            restored.diagview(engine).unwrap(),
            factor.diagview(engine).unwrap()
        );
        assert!(restored.dense_data().is_err());
        assert_eq!(restored.to_bytes_with(&codec).unwrap(), bytes);
    }
}

#[test]
fn dual_diagonal_full_svd_s_roundtrip_preserves_bond_and_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let provider = Arc::new(U1FusionRule);
    let codec = U1Codec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(2), 3)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-2),
            values: vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(-2.0, 0.0),
            ],
        }],
    )
    .unwrap();
    let factor = input.svd_full(engine, &[0], &[1]).unwrap().s;
    let bytes = factor.to_bytes_with(&codec).unwrap();
    let restored = TensorMap::<U1FusionRule, Complex64>::from_bytes_with(
        engine,
        &runtime,
        &bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert_eq!(restored.codomain(), factor.codomain());
    assert_eq!(restored.domain(), factor.domain());
    assert_ne!(restored.domain(), input.domain());
    assert_eq!(
        restored.diagview(engine).unwrap()[0].values,
        vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0)
        ]
    );
    assert!(restored.dense_data().is_err());
    assert_eq!(restored.to_bytes_with(&codec).unwrap(), bytes);
}
