# Issue #1007: current-main Generic-path performance review

## Scope and provenance

This review measures the completed Host GenericSymmetry provider path at main
`5f9e3b45b55220a2560698e262551cf3c3b1e3aa`. It does not change production
algorithms, add caches or prepared APIs, or extend the B/C/D providers.

> Historical measurement only; not current performance authority. See the
> [benchmark index](README.md).

- Host: Apple M4 Max, arm64, Darwin 24.5.0
- Toolchain: rustc/cargo 1.96.0
- Dense backend: faer
- Threads: `RAYON_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- P1 operation-matrix and network times are medians of three fresh-process
  samples. P0 uses the sampling scope stated in its section.

The operation-matrix wrapper emits authority, backend, threads, cache and
allocation scopes, raw samples, and the complete row selected by median time.
The cached-network example emits its CSV, raw samples, cold/cache scope, and the
complete median row; its environment is recorded above rather than emitted by
the example. Caller allocation counters exclude worker threads, native BLAS
allocations, frees, and peak/live bytes.

Historical commands (the cached-network program has been retired):

```sh
cargo test -p tenet-tensors --features racah-generated \
  measure_checked_generic_transform_phases -- --ignored --nocapture --test-threads=1
cargo test -p tenet-rs --features racah-generated \
  --test permute_overwrite_allocations \
  checked_generic_public_transform_measurement -- --ignored --nocapture --test-threads=1
OP_MATRIX_MIN_MS=0 OP_MATRIX_OPERATION=permute OP_MATRIX_FORM=owned \
  benchmarks/operation_matrix.sh
OP_MATRIX_DEGENERACY=2 OP_MATRIX_MIN_MS=5 \
  OP_MATRIX_OPERATION=contract_input_swap OP_MATRIX_FORM=owned \
  benchmarks/operation_matrix.sh
```

## P0: checked-Generic tree transforms

The existing public-transform measurement uses a tiny outer-multiplicity-two
fixture. Each case runs in one fresh child, records one cold call, and reports the
median of seven warm calls within that child. Allocation values below are
caller-thread allocation calls and requested bytes for one warm public operation.

| Symmetry / operation | Warm time | Allocations | Requested bytes |
|---|---:|---:|---:|
| SU2 permute control | 14.750 us | 14 | 1,128 |
| SU3 permute | 103.708 us | 268 | 18,370 |
| SU3 braid | 113.833 us | 269 | -- |
| SU3 repartition | 120.958 us | 270 | -- |
| SU4 permute | 148.125 us | 305 | 21,490 |
| SU4 braid | 147.708 us | 306 | -- |
| SU4 repartition | 150.250 us | 307 | -- |

Cold SU3 permute was 46.46 ms and 32,001 allocations; cold SU4 permute was
55.55 ms and 60,067 allocations. On this fixture, the warm checked-Generic
permute is therefore about 7--10 times the SU2 time and 19--22 times its caller
allocation count.

The deeper instrumented test observed first plan construction at 22--48 ms,
repeated plan/preflight work at 0.26--1.10 ms, preallocated replay at 33--75 us,
and 1,896 retained payload bytes. Provider-call vectors repeat on warm owned
paths. These spy-instrumented timings include auxiliary instrumentation overhead
and must not be compared numerically with the public wall times above.

## P1: public operation matrix

The focused permute smoke used degeneracy 8. The wrapper run preceded a later
wording-only commit; its executable behavior was unchanged. A subsequent
final-head contract smoke verified the complete wrapper and oracle.

| Symmetry | Permute cold / warm | Cold / warm allocations per iteration |
|---|---:|---:|
| U1 | 380.041 / 4.104 us | 685 / 15 |
| SU2 | 301.708 / 5.812 us | 1,281 / 15 |
| SU3 | 4,015.667 / 14.812 us | 28,577 / 271 |
| SU4 | 3,332.125 / 19.229 us | 53,543 / 308 |

The degeneracy-2 `contract_input_swap` final-head smoke gave:

| Symmetry | Cold | Warm | Warm allocations per iteration |
|---|---:|---:|---:|
| U1 | 1,740.625 us | 33.501 us | 229 |
| SU2 | 1,209.041 us | 46.177 us | 229 |
| SU3 | 22,326.750 us | 1,519.531 us | 17,345 |
| SU4 | 25,435.709 us | 1,478.114 us | 19,450 |

This smoke exposed an exact-float comparison in the benchmark oracle, not a
production arithmetic failure. The oracle now keeps provider, space, block, and
fusion-tree checks exact and compares finite payload entries with `64 eps`
absolute plus `256 eps` relative tolerance. The failing case then passed.

## P1: cached network

Each symmetry/cache/sample runs in its own child. The timer starts after the
mandatory provider, space, and tensor fixture construction, so construction-time
Racah work is excluded. The direct-contract oracle runs after timing. Each enabled
case retained one idle workspace and reused it 22 times.

| Symmetry | Enabled cold / warm | Disabled cold / warm | Retained bytes |
|---|---:|---:|---:|
| U1 | 1,771.500 / 1,223.312 us | 1,642.459 / 1,214.694 us | 2,656 |
| SU3 | 1,379.917 / 1,054.156 us | 1,244.667 / 1,085.202 us | 2,945 |
| SU4 | 1,307.250 / 1,061.942 us | 1,365.875 / 1,080.338 us | 2,945 |

The first U1 enabled-cold raw sample was 8,326.792 us; the three-child median is
reported above. With only three samples and differences of a few percent, this
tiny fixture does not establish a material steady-state speedup from retained
network workspaces. It does establish reuse and bounded retained storage.

## Conclusions

Measured fact: checked-Generic warm transforms and the representative swapped
contraction have substantially higher caller allocation counts and latency than
the U1/SU2 controls on these small fixtures. Measured fact: warm owned transforms
still perform provider/preflight work after a plan hit.

Inference: repeated provider/preflight/allocation work is the narrowest evidenced
candidate for the warm gap. The measurements do not prove which subphase will
dominate after one part is removed, and they do not justify a broad ordinary-
operation cache, a prepared public API, or zero-provider-call replay contract.

Create one follow-up issue to reduce and remeasure checked-Generic transform
provider/preflight/allocation work after a plan hit. Keep optimization behind the
existing provider-neutral path and require the same SU2/SU3/SU4 public and deep
measurements before accepting it. No network-cache optimization is justified by
this fixture.
