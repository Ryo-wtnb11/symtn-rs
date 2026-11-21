# symtn-dense

Expert dense-backend boundary. Symmetric layers use `DenseExecutor`,
`DenseView`, `DenseViewMut`, `DenseTensor`, and `DenseError`
without depending on a concrete kernel provider. Ordinary applications choose
their backend through `symtn::typed::RuntimeBuilder` instead.

`tenferro` supplies the default CPU executor (faer-backed); `cuda` exports
CUDA context, storage, and kernels.
