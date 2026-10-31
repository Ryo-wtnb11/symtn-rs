# symtn agent instructions

Read [REPOSITORY_RULES.md](REPOSITORY_RULES.md) for the authoritative symtn
quality gates. The parent workspace `AGENTS.md`, when present, also applies.
Use [docs/README.md](docs/README.md) to find current design and testing
documents. Historical audits and issue notes are evidence for their recorded
revision, not current capability claims.

Before changing code, fetch `origin/main`, record its commit, inspect the
current issue and implementation, and trace the complete public-to-backend
path. For mathematical or representation changes, record the corresponding
TensorKit and QSpace production paths and build an independent oracle.

Keep the public API as small symmetric tensor operations that applications can
compose. symtn owns categorical semantics and block structure; Tenferro owns
dense kernels and backend resources. Reuse existing execution seams. Do not
add a general work graph, a second CPU scheduler, a symmetry-specific engine
path, or a cache to hide an avoidably slow eager operation.

For production code, use distinct supervisor and implementer actors. Add an
independent reviewer for mathematics, representation, core algorithms, unsafe
code, or performance-sensitive changes. Review the exact committed diff and
complete every applicable local and CI gate before merge. Do not edit or
update a dependency without explicit user approval.

State supported providers, scalars, storage, placement, errors, and resource
costs precisely. Keep code, tests, rustdoc, examples, README, `docs/`, and
feature claims consistent in the same commit: before committing, search
`docs/`, `README.md`, `examples/`, and rustdoc for every public name, error,
option, and behavior the change touches, and update each stale statement.
Reviewers treat a stale document as a finding.

Pin performance evidence to a revision, provider, build, thread count,
workload, and raw result; report regressions and inconclusive measurements
too.

## Tests

A test exists to protect a stated purpose; write the purpose first, then the
assertion that checks exactly it. Reviewers judge every test by that pair.

1. One purpose per test, named in the test name; the assertion checks that
   purpose and nothing incidental.
2. Correctness oracles are independent of the code under test: dense
   expansion, hand values, TensorKit reference steps, or F/R-symbol
   composition. Never compare against a recorded earlier output of the same
   code.
3. Expected values are written inline. Opaque fixture files are allowed only
   for wire-format goldens.
4. Allocation, byte, upload and kernel-call tests assert structural
   invariants only: zero on a warm path, equality between a small and a large
   fixture (no growth), one allocation for the one output payload, retained
   bytes at most the charge, build or miss counters at zero when warm.
5. Never assert a recorded count, ceiling or byte figure. If an exact count
   is the contract, compute it from the fixture.
6. No wall-clock timing, sleeps for ordering, or exact `size_of` checks.
   Write an upper bound only when a memory bound is the documented contract.
7. Probe and spy counts are zero or derived from the fixture. Do not assert
   internal staging order or step counts that a valid refactor may change.
8. A forced-route flag is not a substitute for a public fixture; a branch no
   public input reaches is deleted.
9. Error tests assert the variant. Match message text only for a stringly
   variant, and then only the reason substring; at most one `Display` test per
   error type.
10. Do not re-test dependency contracts (fusion-symbols mathematics, Tenferro
    or vendor kernels); test symtn's boundary only.
11. Cover the relevant semantic dimensions (non-Abelian, multiplicity,
    fermionic, dual legs, real and complex) once per contract in one matrix,
    not again per file, storage mode or engine.
12. Backend- or hardware-dependent tests are `#[ignore = "<reason>"]`; never
    return early to pass silently.
13. Benchmarks and measurements live in `benches/`, not in ignored tests.
14. No issue numbers, commit hashes, leaf or stage names, or "measured on
    main" notes in test names or comments; explain why, not the history.
15. Reuse the shared fixtures in `tests/support/` (legs, spaces, counting
    allocator, export scanner) instead of adding local copies.
