# symtn-operations

Symmetry-free execution layer beneath `symtn-tensors`; it deliberately does
not consume fusion rules or enumerate fusion trees. Expert entry points include
operation specifications, `OperationError`, backend traits, replay structures,
and `DenseTreeTransformOperations`.

Enable the host backend `cpu-faer` (the default). The `cuda`
feature still requires it for replay;
raw kernels are not a user-level API.
