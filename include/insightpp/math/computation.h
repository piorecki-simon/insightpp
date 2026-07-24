#ifndef INSIGHTPP_COMPUTATION_H
#define INSIGHTPP_COMPUTATION_H

#include "../data/container.h"

#include <cblas.h>
#include <omp.h>

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
static void batched_mat_vec_mul_w_bias_kernel(T* __restrict__ out_ptr, const T* __restrict__ sample_batch_t_ptr, const T* __restrict__ weight_matrix_t_ptr, const T* __restrict__ bias_ptr) noexcept
{

#ifndef USE_OPENBLAS
    if constexpr (B > 64) {

        #pragma omp parallel for num_threads(4)
        for (std::size_t b = 0; b < B; ++b) {
            const std::size_t b_s1 = data::Tensor<T, B, K>::strides[0] * b;
            const std::size_t o_s1 = data::Tensor<T, B, N>::strides[0] * b;

            for (std::size_t n = 0; n < N; ++n) {
                out_ptr[o_s1 + n] = bias_ptr[n];
            }

            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t w_s2 = data::Tensor<T, K, N>::strides[0] * k;
                const T x = sample_batch_t_ptr[b_s1 + k];

                for (std::size_t n = 0; n < N; ++n) {
                    out_ptr[o_s1 + n] += x * weight_matrix_t_ptr[w_s2 + n];
                }
            }
        }

    } else {
        for (std::size_t b = 0; b < B; ++b) {
            const std::size_t b_s1 = data::Tensor<T, B, K>::strides[0] * b;
            const std::size_t o_s1 = data::Tensor<T, B, N>::strides[0] * b;

            for (std::size_t n = 0; n < N; ++n) {
                out_ptr[o_s1 + n] = bias_ptr[n];
            }

            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t w_s2 = data::Tensor<T, K, N>::strides[0] * k;
                const T x = sample_batch_t_ptr[b_s1 + k];

                for (std::size_t n = 0; n < N; ++n) {
                    out_ptr[o_s1 + n] += x * weight_matrix_t_ptr[w_s2 + n];
                }
            }
        }
    }

#else
    for (std::size_t b = 0; b < B; ++b) {
        cblas_scopy(
            N,
            bias_ptr,
            1,
            out_ptr + b * N,
            1
        );
    }

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        B,
        N,
        K,
        1.0f,
        sample_batch_t_ptr,
        K,
        weight_matrix_t_ptr,
        N,
        1.0f,
        out_ptr,
        N
    );

#endif

}

template <typename T, std::size_t B, std::size_t N, std::size_t K>
requires std::floating_point<T>
void batched_mat_vec_mul_w_bias(const data::Tensor<T, B, K>& sample_batch_t, const data::Tensor<T, K, N>& weight_matrix_t, const data::Tensor<T, N>& bias, data::Tensor<T, B, N>& out)
{
    T* __restrict__ out_ptr = out.data_ptr();
    const T* __restrict__ sample_batch_t_ptr = sample_batch_t.data_ptr();
    const T* __restrict__ weight_matrix_t_ptr = weight_matrix_t.data_ptr();
    const T* __restrict__ bias_ptr = bias.data_ptr();

    batched_mat_vec_mul_w_bias_kernel<T, B, N, K>(out_ptr, sample_batch_t_ptr, weight_matrix_t_ptr, bias_ptr);
}

} // namespace math
} // namespace inpp

#endif //INSIGHTPP_COMPUTATION_H
