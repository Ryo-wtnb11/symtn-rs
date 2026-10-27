//! SU(3) F/R symbols are a deterministic function of their labels: fresh
//! processes (independent hash seeds and cold fusion-symbols caches) must produce
//! bit-identical blocks (issue #1932; fusion-symbols, formerly racah >= 0.2.3).
//!
//! The parent test re-runs this test binary as two child processes, each of
//! which hashes every block's bit pattern, and compares the children with each
//! other and with the parent. No hash value is committed: bits may legitimately
//! differ across platforms and dependency builds; only run-to-run variation is
//! the defect.

#![cfg(feature = "cgc-gen")]

use std::process::Command;

use symtn_sectors::{CheckedGenericFusion, CheckedGenericRigidSymbols, LieGroupFusionRule};

const CHILD_ENV: &str = "SYMTN_LIE_GROUP_DETERMINISM_CHILD";
const TEST_NAME: &str =
    "lie_group_cross_process_determinism::su3_f_and_r_symbols_are_bit_identical_across_processes";

fn fnv(hash: &mut u64, bits: u64) {
    for byte in bits.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

/// Hashes every F block over `{3, 3̄, 8}` (including the multiplicity-2
/// channel `8 ⊗ 8 → 8`) and every R block among them; returns
/// `(f_blocks, r_blocks, hash)`.
fn su3_symbol_hash() -> (usize, usize, u64) {
    let rule = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let labels = [[1, 1], [1, 0], [0, 1]].map(|l| rule.encode_irrep(&l, &[]).unwrap());
    let channels = |x, y| rule.try_fusion_channels(x, y).unwrap();
    let (mut f_blocks, mut r_blocks, mut hash) = (0, 0, 0xcbf2_9ce4_8422_2325u64);
    for &a in &labels {
        for &b in &labels {
            for c in channels(a, b) {
                for &x in rule.try_r_symbol_generic(a, b, c).unwrap().data() {
                    fnv(&mut hash, x.to_bits());
                }
                r_blocks += 1;
            }
            for &c in &labels {
                for e in channels(a, b) {
                    for f in channels(b, c) {
                        let a_f = channels(a, f);
                        for d in channels(e, c).into_iter().filter(|d| a_f.contains(d)) {
                            let block = rule.try_f_symbol_generic(a, b, c, d, e, f).unwrap();
                            for &x in block.data() {
                                fnv(&mut hash, x.to_bits());
                            }
                            f_blocks += 1;
                        }
                    }
                }
            }
        }
    }
    (f_blocks, r_blocks, hash)
}

#[test]
fn su3_f_and_r_symbols_are_bit_identical_across_processes() {
    let (f_blocks, r_blocks, hash) = su3_symbol_hash();
    let line = format!("su3-symbol-hash {f_blocks} {r_blocks} {hash:016x}");
    if std::env::var_os(CHILD_ENV).is_some() {
        // libtest prints `test <name> ... ` without a newline first.
        println!("\n{line}");
        return;
    }
    assert!(f_blocks > 0 && r_blocks > 0);
    let exe = std::env::current_exe().unwrap();
    for _ in 0..2 {
        let output = Command::new(&exe)
            .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "child failed: {output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let child = stdout
            .lines()
            .find(|l| l.starts_with("su3-symbol-hash "))
            .unwrap_or_else(|| panic!("child printed no hash: {stdout}"));
        assert_eq!(child, line);
    }
}
