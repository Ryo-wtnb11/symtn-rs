# symtn-tensors

Compiler and executor for symmetric operations over `symtn-core` fusion-tree
layouts. It exposes expert `*_into` operations, execution contexts, transform
and contraction plans, and backend traits; ordinary code should use
`symtn::typed::TensorMap` methods.

`cuda` enables device seams and `cgc-gen` forwards generated SUN
support. User examples live in the `symtn` crate.
