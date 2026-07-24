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
- OpenBLAS
- libtorch

### Currently implemented

- Simple SGD
- ReLU
- MSE
- OpenBLAS choice

### Missing basics

- Quick start "How to"
- Unit tests
- Documentation
- Comparison between PyTorch (Python), libtorch and insightpp

### Planned features

- Adam
- Advanced memory management (alignment etc.) \[in progress\]
- Advanced kernels for GEMM and other computations
- He initialization

### Much later planned features

- CNNs
- Transformer
- Diffusion
- Autograd
- Data loading