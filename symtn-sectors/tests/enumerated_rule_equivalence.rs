//! The Z_N, fermion-parity and CU(1) rules over the enumerated codec answer
//! exactly as the dedicated codecs they replaced, for every sector of the
//! fixture domain the other suites use.
//!
//! Why recorded values rather than an independent oracle: this is a
//! representation change with no intended semantic change, so the contract is
//! bitwise agreement with the previous rules. The values were recorded from
//! those rules; the suite goes once the SU(2) rule is migrated as well.

use symtn_sectors::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule, FermionParityIrrep,
    FusionRule, MultiplicityFreeRigidSymbols, SectorCodec, SectorId, ZNFusionRule, ZNIrrep,
};

/// FNV-1a 64 over a stream of words.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn word(&mut self, word: u64) {
        for byte in word.to_le_bytes() {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn scalar(&mut self, value: f64) {
        self.word(value.to_bits());
    }
}

trait Rule: FusionRule + CheckedFusionAlgebra + MultiplicityFreeRigidSymbols<Scalar = f64> {}
impl<R: FusionRule + CheckedFusionAlgebra + MultiplicityFreeRigidSymbols<Scalar = f64>> Rule for R {}

/// Styles, vacuum, and per sector: dual, the four dimensions, twist,
/// Frobenius–Schur phase, and the order key, as a readable transcript.
fn unary<R: Rule>(rule: &R, ids: &[usize]) -> String {
    let mut out = format!(
        "{:?} {:?} {} v{}",
        rule.fusion_style(),
        rule.braiding_style(),
        rule.has_trivial_associator_gauge(),
        rule.vacuum().id()
    );
    for &id in ids {
        let a = SectorId::new(id);
        out += &format!(
            "|{id}:d{} {:x} {:x} {:x} {:x} t{:x} fs{:x} k{:?}",
            rule.try_dual_sector(a).unwrap().id(),
            rule.dim_scalar(a).to_bits(),
            rule.inv_dim_scalar(a).to_bits(),
            rule.sqrt_dim_scalar(a).to_bits(),
            rule.inv_sqrt_dim_scalar(a).to_bits(),
            rule.twist_scalar(a).to_bits(),
            rule.frobenius_schur_phase_scalar(a).to_bits(),
            rule.sector_order_key(a).positions(),
        );
    }
    out
}

/// The fusion channels of every ordered pair, in the rule's order.
fn channels<R: Rule>(rule: &R, ids: &[usize]) -> String {
    let mut out = String::new();
    for &a in ids {
        for &b in ids {
            let got = rule
                .try_fusion_channels(SectorId::new(a), SectorId::new(b))
                .unwrap();
            let got: Vec<usize> = got.iter().map(|c| c.id()).collect();
            out += &format!("|{a}x{b}={got:?}");
        }
    }
    out
}

/// Digests of N, R, A, B over every triple and of F over every sextuple.
fn symbols<R: Rule>(rule: &R, ids: &[usize]) -> [u64; 5] {
    let s = |id: usize| SectorId::new(id);
    let [mut n, mut r, mut a, mut b, mut f] = std::array::from_fn(|_| Digest::new());
    for &x in ids {
        for &y in ids {
            for &z in ids {
                n.word(rule.try_nsymbol(s(x), s(y), s(z)).unwrap() as u64);
                r.scalar(rule.r_symbol_scalar(s(x), s(y), s(z)));
                a.scalar(rule.a_symbol_scalar(s(x), s(y), s(z)));
                b.scalar(rule.b_symbol_scalar(s(x), s(y), s(z)));
                for &u in ids {
                    for &v in ids {
                        for &w in ids {
                            f.scalar(rule.f_symbol_scalar(s(x), s(y), s(z), s(u), s(v), s(w)));
                        }
                    }
                }
            }
        }
    }
    [n.0, r.0, a.0, b.0, f.0]
}

fn transcript<R: Rule>(name: &str, rule: &R, ids: &[usize]) -> [String; 3] {
    [
        format!("{name} unary {}", unary(rule, ids)),
        format!("{name} channels {}", channels(rule, ids)),
        format!("{name} symbols {:x?}", symbols(rule, ids)),
    ]
}

/// The previous rules' transcripts over the fixture domain: Z2 (the former
/// `Z2FusionRule`) and Z_N for N = 2, 3, 5 and `u32::MAX`, fermion parity,
/// CU(1) through `2j = 5`, and the wide CU(1) charges `2j = 2^31` and
/// `2j = u32::MAX - 1`.
const RECORDED: &str = r#"
z2 unary Unique Bosonic true v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[1]
z2 channels |0x0=[0]|0x1=[1]|1x0=[1]|1x1=[0]
z2 symbols [7208d241c34e0fa5, b76949cd1b2617a5, b76949cd1b2617a5, b76949cd1b2617a5, 71ea9aba57cb7325]
zn2 unary Unique Bosonic true v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[1]
zn2 channels |0x0=[0]|0x1=[1]|1x0=[1]|1x1=[0]
zn2 symbols [7208d241c34e0fa5, b76949cd1b2617a5, b76949cd1b2617a5, b76949cd1b2617a5, 71ea9aba57cb7325]
zn3 unary Unique Bosonic true v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d2 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[1]|2:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[2]
zn3 channels |0x0=[0]|0x1=[1]|0x2=[2]|1x0=[1]|1x1=[2]|1x2=[0]|2x0=[2]|2x1=[0]|2x2=[1]
zn3 symbols [f300b47f7f922cc4, 5f02ba591bac2958, 5f02ba591bac2958, 5f02ba591bac2958, 9dcd2f82add2ca38]
zn5 unary Unique Bosonic true v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d4 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[1]|2:d3 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[2]|3:d2 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[3]|4:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[4]
zn5 channels |0x0=[0]|0x1=[1]|0x2=[2]|0x3=[3]|0x4=[4]|1x0=[1]|1x1=[2]|1x2=[3]|1x3=[4]|1x4=[0]|2x0=[2]|2x1=[3]|2x2=[4]|2x3=[0]|2x4=[1]|3x0=[3]|3x1=[4]|3x2=[0]|3x3=[1]|3x4=[2]|4x0=[4]|4x1=[0]|4x2=[1]|4x3=[2]|4x4=[3]
zn5 symbols [8693008c64e8c2e4, 9cc1adc5c9ccb78, 9cc1adc5c9ccb78, 9cc1adc5c9ccb78, 983247a2a3f8cd38]
znmax unary Unique Bosonic true v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d4294967294 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[1]|4294967294:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[4294967294]
znmax channels |0x0=[0]|0x1=[1]|0x4294967294=[4294967294]|1x0=[1]|1x1=[2]|1x4294967294=[0]|4294967294x0=[4294967294]|4294967294x1=[0]|4294967294x4294967294=[4294967293]
znmax symbols [bda4cf873c3e01e4, 3e7d900a672e2a78, 3e7d900a672e2a78, 3e7d900a672e2a78, 42a494012b543e78]
fz2 unary Unique Fermionic true v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 tbff0000000000000 fs3ff0000000000000 k[1]
fz2 channels |0x0=[0]|0x1=[1]|1x0=[1]|1x1=[0]
fz2 symbols [7208d241c34e0fa5, 6c93f89768a12125, b76949cd1b2617a5, b76949cd1b2617a5, 71ea9aba57cb7325]
cu1 unary Simple Bosonic false v0|0:d0 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[0]|1:d1 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 t3ff0000000000000 fs3ff0000000000000 k[1]|2:d2 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[2]|3:d3 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[3]|4:d4 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[4]|5:d5 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[5]|6:d6 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[6]
cu1 channels |0x0=[0]|0x1=[1]|0x2=[2]|0x3=[3]|0x4=[4]|0x5=[5]|0x6=[6]|1x0=[1]|1x1=[0]|1x2=[2]|1x3=[3]|1x4=[4]|1x5=[5]|1x6=[6]|2x0=[2]|2x1=[2]|2x2=[0, 1, 3]|2x3=[2, 4]|2x4=[3, 5]|2x5=[4, 6]|2x6=[5, 7]|3x0=[3]|3x1=[3]|3x2=[2, 4]|3x3=[0, 1, 5]|3x4=[2, 6]|3x5=[3, 7]|3x6=[4, 8]|4x0=[4]|4x1=[4]|4x2=[3, 5]|4x3=[2, 6]|4x4=[0, 1, 7]|4x5=[2, 8]|4x6=[3, 9]|5x0=[5]|5x1=[5]|5x2=[4, 6]|5x3=[3, 7]|5x4=[2, 8]|5x5=[0, 1, 9]|5x6=[2, 10]|6x0=[6]|6x1=[6]|6x2=[5, 7]|6x3=[4, 8]|6x4=[3, 9]|6x5=[2, 10]|6x6=[0, 1, 11]
cu1 symbols [791ac13a13029325, 95da423ca53de425, b5284d578b497205, 91324f2019764505, bc62c3935d04fed]
cu1wide unary Simple Bosonic false v0|2147483649:d2147483649 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[2147483649]|4294967295:d4294967295 4000000000000000 3fe0000000000000 3ff6a09e667f3bcd 3fe6a09e667f3bcc t3ff0000000000000 fs3ff0000000000000 k[4294967295]
cu1wide channels |2147483648x2147483648=[0, 1, 4294967295]
"#;

#[test]
fn enumerated_rules_reproduce_the_recorded_answers_bitwise() {
    let max = u32::MAX as usize;
    let cu1 = CU1FusionRule::new();
    let zn = |modulus| ZNFusionRule::new(modulus).unwrap();
    let half = (max - 1) / 2;
    let mut lines: Vec<String> = [
        transcript("z2", &zn(2), &[0, 1]),
        transcript("zn2", &zn(2), &[0, 1]),
        transcript("zn3", &zn(3), &[0, 1, 2]),
        transcript("zn5", &zn(5), &[0, 1, 2, 3, 4]),
        transcript("znmax", &zn(u32::MAX), &[0, 1, max - 1]),
        transcript("fz2", &FermionParityFusionRule::new(), &[0, 1]),
        transcript("cu1", &cu1, &[0, 1, 2, 3, 4, 5, 6]),
    ]
    .into_iter()
    .flatten()
    .collect();
    lines.push(format!(
        "cu1wide unary {}",
        unary(&cu1, &[(1 << 31) + 1, max])
    ));
    lines.push(format!("cu1wide channels {}", channels(&cu1, &[half + 1])));
    let recorded: Vec<&str> = RECORDED.trim().lines().collect();
    assert_eq!(lines.len(), recorded.len());
    for (got, want) in lines.iter().zip(recorded) {
        assert_eq!(got, want);
    }
}

#[test]
fn fusion_symbols_labels_encode_to_the_recorded_ids() {
    let s = SectorId::new;
    let z5 = ZNFusionRule::new(5).unwrap();
    assert_eq!(z5.encode_sector(&ZNIrrep::new(-1, 5).unwrap()), Ok(s(4)));
    let fz2 = FermionParityFusionRule::new();
    assert_eq!(fz2.encode_sector(&FermionParityIrrep::EVEN), Ok(s(0)));
    assert_eq!(fz2.encode_sector(&FermionParityIrrep::ODD), Ok(s(1)));
    let cu1 = CU1FusionRule::new();
    for (label, id) in [
        (CU1Irrep::VACUUM, 0),
        (CU1Irrep::PSEUDOSCALAR, 1),
        (CU1Irrep::from_twice_charge(1), 2),
        (CU1Irrep::from_twice_charge(u32::MAX - 1), u32::MAX as usize),
    ] {
        assert_eq!(cu1.encode_sector(&label), Ok(s(id)));
        assert_eq!(cu1.decode_sector(s(id)), Ok(label));
    }
}
