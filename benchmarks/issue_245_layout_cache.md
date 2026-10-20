# Issue #245 fusion-layout cache benchmark

> Historical measurement only; not current performance authority. The exact
> compared revisions are recorded below; see the [benchmark index](README.md).

This report records an exploratory non-regression check for commit
`e42a34138b2a06482cf18213f9a2042eb59eff17` against `origin/main` at
`7825e085de7ed3d2c3f5921721f7701e4d637a86`.

The machine was not quiet: Zoom, Dia, WindowServer, and `coreaudiod` were using
measurable CPU during the run. The samples therefore do not constitute the
formal machine-quiet <=3% proof. They are retained as reproducible evidence
that did not expose a regression under the available conditions.

## Source and commands

Both revisions used the exact source in
`tenet-core/examples/issue_245_layout_cache.rs`. The source was copied into the
detached main worktree without changing the main revision.

The retired program was built in release mode for each revision.
The main executable ran before the branch executable for each of seven samples.
The program reports two warm paths in nanoseconds per operation for U(1)
spaces with 1, 8, and 64 sectors: direct `fusion_tree_keys`/layout lookup, and
coupled-block lookup including shape cloning and coupled-structure lookup.

## Raw samples

| sample | revision | direct 1 | direct 8 | direct 64 | coupled 1 | coupled 8 | coupled 64 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | main | 65 | 90 | 309 | 1150 | 5221 | 35740 |
| 1 | branch | 61 | 73 | 179 | 1157 | 5319 | 36754 |
| 2 | main | 66 | 90 | 325 | 1140 | 5189 | 36122 |
| 2 | branch | 61 | 75 | 178 | 1163 | 5398 | 37297 |
| 3 | main | 69 | 91 | 326 | 1178 | 5363 | 36513 |
| 3 | branch | 61 | 88 | 183 | 1227 | 5392 | 37584 |
| 4 | main | 67 | 99 | 323 | 1206 | 5525 | 37380 |
| 4 | branch | 76 | 81 | 182 | 1206 | 5359 | 38269 |
| 5 | main | 74 | 94 | 332 | 1232 | 5843 | 38054 |
| 5 | branch | 67 | 74 | 182 | 1273 | 5742 | 41552 |
| 6 | main | 66 | 91 | 335 | 1221 | 6076 | 40961 |
| 6 | branch | 83 | 72 | 182 | 1293 | 5386 | 38732 |
| 7 | main | 77 | 94 | 323 | 1234 | 7211 | 38479 |
| 7 | branch | 78 | 88 | 187 | 1210 | 5933 | 38525 |

## Medians

Direct `fusion_tree_keys`/layout lookup:

| sectors | main (ns) | branch (ns) | change |
|---:|---:|---:|---:|
| 1 | 67 | 67 | 0.0% |
| 8 | 91 | 75 | -17.6% |
| 64 | 325 | 182 | -44.0% |

Coupled-block lookup including shape cloning:

| sectors | main (ns) | branch (ns) | change |
|---:|---:|---:|---:|
| 1 | 1206 | 1210 | +0.3% |
| 8 | 5525 | 5392 | -2.4% |
| 64 | 37380 | 38269 | +2.4% |

## Provenance

```text
OS: macOS 15.5 (24F74), Darwin 24.5.0, arm64
rustc: 1.96.0 (ac68faa20 2026-05-25), LLVM 22.1.2
cargo: 1.96.0 (30a34c682 2026-05-25)
main: 7825e085de7ed3d2c3f5921721f7701e4d637a86
branch code: e42a34138b2a06482cf18213f9a2042eb59eff17
machine-quiet: no
```
