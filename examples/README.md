# Examples

---

## MNIST

### Dependencies

- Unzipped MNIST files
  - ``./mnist/train-images-idx3-ubyte``
  - ``./mnist/train-labels-idx1-ubyte``
  - ``./mnist/t10k-images-idx3-ubyte``
  - ``./mnist/t10k-labels-idx1-ubyte``

- libtorch (CPU only)
- intel MKL if CMake option: ``USE_MKL_BLAS=ON``
- OpenMP

### How to run

- ``./mnist_sgd_01 <path_to_mnist_folder>``
- ``./mnist_sgd_01_libtorch <path_to_mnist_folder>``