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
