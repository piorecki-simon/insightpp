#ifndef INSIGHTPP_COMPUTATION_H
#define INSIGHTPP_COMPUTATION_H

#include "../data/container.h"

namespace inpp {
namespace math {

template <typename T, std::size_t B, std::size_t N, std::size_t K>
requires std::floating_point<T>
void batched_mat_vec_mul(const data::Tensor<T, B, K>& sample_batch_t, const data::Tensor<T, K, N>& weight_matrix_t, data::Tensor<T, B, N>& out)
{
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t n = 0; n < N; ++n) {
            T sum_tmp = static_cast<T>(0.0);
            for (std::size_t k = 0; k < K; ++k) {
                sum_tmp += weight_matrix_t(k, n) * sample_batch_t(b, k);
            }
            out(b, n) = sum_tmp;
        }
    }
}

template <typename T, std::size_t B, std::size_t N, std::size_t K>
requires std::floating_point<T>
void batched_mat_vec_mul_w_bias(const data::Tensor<T, B, K>& sample_batch_t, const data::Tensor<T, K, N>& weight_matrix_t, const data::Tensor<T, N>& bias, data::Tensor<T, B, N>& out)
{
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t n = 0; n < N; ++n) {
            T sum_tmp = static_cast<T>(0.0);
            for (std::size_t k = 0; k < K; ++k) {
                sum_tmp += weight_matrix_t(k, n) * sample_batch_t(b, k);
            }
            out(b, n) = sum_tmp + bias(n);
        }
    }
}

} // namespace math
} // namespace inpp

#endif //INSIGHTPP_COMPUTATION_H
