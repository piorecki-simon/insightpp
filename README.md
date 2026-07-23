# insightpp

This is my own ML library, work in progress.

---

## About

This library is meant to be a simple ML library to dive deep into code optimization and maybe beat [libtorch](https://pytorch.org/) someday in 
special cases.

In the current state it is slower than using libtorch using SGD and a very simple network with linear layers and ReLU activation 
on the MNIST dataset.

## Dependencies

- OpenMP

### Currently implemented

- Simple SGD
- ReLU
- MSE

### Missing basics

- Quick start "How to"
- Unit tests
- Documentation

### Planned features

- Adam
- Advanced memory management (alignment etc.)
- Advanced kernels for GEMM and other computations
- OpenBLAS choice
- He initialization

### Much later planned features

- CNNs
- Transformer
- Diffusion
- Autograd
