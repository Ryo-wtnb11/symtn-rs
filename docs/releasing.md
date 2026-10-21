# Releasing symtn

This is the checklist for a future registry release. The facade is
`symtn` 0.1.1; the workspace pins the Tenferro 0.7.1 line. Its Rust library
target remains `symtn`, so
downstream code imports from `symtn::typed` and `symtn::sector`.

## Publication order

Publish only after the previous layer is visible in the crates.io index. Each
package is packaged and inspected before it is published.

1. `symtn-sectors` (after a compatible fusion-symbols release is available)
2. `symtn-core`
3. `symtn-dense`
4. `symtn-operations`
5. `symtn-tensors`
6. `symtn-matrixalgebra`
7. `symtn`
8. `symtn-macros` (no symtn dependencies; any time before `symtn-network`)
9. `symtn-network`

## Checks for each package

Run from a clean checkout with the tracked `Cargo.lock` (CI also uses `--locked`) and the intended toolchain:

```sh
cargo package -p <package> --locked
cargo publish --dry-run -p <package> --locked
```

Inspect the packaged manifest and reject the package if a normal dependency
still points to a local path or a mutable git branch. Path dependencies with a
matching version are allowed during workspace development; the registry
manifest must resolve them from crates.io.

After the complete closure is published, build a clean downstream fixture:

```toml
[dependencies]
symtn = "0.1.1"
```

The fixture must compile `use symtn::typed::{Runtime, TensorMap}` without a sibling checkout,
Tenferro source override, or fusion-symbols git override.

## External prerequisites

The workspace manifests pin the `tenferro-*` crates at 0.7.1 (`symtn-dense/Cargo.toml`),
`t4a-cubecl-runtime` at `=0.10.1` (kept in lockstep with Tenferro), and `fusion-symbols` at
0.5.1 (`symtn-sectors/Cargo.toml`, `symtn-tensors/Cargo.toml`).
The registry-only gate must still verify compatible resolved versions before a
future release.
