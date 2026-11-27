# symtn-dense

Internal dense-backend boundary of symtn. The symmetric layers use
`DenseExecutor`, `DenseView`, `DenseViewMut`, `DenseTensor`, and `DenseError`
without depending on a concrete kernel provider. Applications use the `symtn`
facade and pass their tenferro engine to each operation; they do not name
these types.

`tenferro` supplies the default CPU executor (faer-backed); `cuda` exports
CUDA context, storage, and kernels.
