#ifndef INSIGHTPP_LINEAR_H
#define INSIGHTPP_LINEAR_H

#include <cstddef>

#include "../data/container.h"
#include "../data/initialization.h"
#include "../math/computation.h"

#include <chrono>
#include <cblas.h>
#include <omp.h>

namespace inpp {
namespace nn {

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy = init::RandomFromTo<-0.1f, 0.1f>>
requires std::floating_point<T>
class Linear {
    using WeightTensor = data::Tensor<T, inp_dim, out_dim>;
    using BiasTensor = data::Tensor<T, out_dim>;

/* API */
public:
    explicit Linear(const T learning_rate) : learning_rate(learning_rate) {}

    T w_grad_access(std::size_t idx) const noexcept { return w_grad[idx]; }
    T w_access(std::size_t idx) const noexcept { return w[idx]; }
    T forward_in_access(std::size_t idx) const noexcept { return forward_in[idx]; }

    const data::Tensor<T, batch_size, out_dim>& forward(const data::Tensor<T, batch_size, inp_dim>& x) noexcept;
    const data::Tensor<T, batch_size, inp_dim>& backward(const data::Tensor<T, batch_size, out_dim>& d_z) noexcept;
    void update() noexcept;

/* Routines */
private:
    static void update_weights_kernel(T* __restrict__ w_ptr, const T* __restrict__ w_grad_ptr, T learning_rate) noexcept;
    static void update_bias_kernel(T* __restrict__ b_ptr, const T* __restrict__ b_grad_ptr, T learning_rate) noexcept;

    static void backward_weight_grad_kernel(T* __restrict__ w_grad_ptr, const T* __restrict__ fwd_in_ptr, const T* __restrict__ d_z_ptr) noexcept;
    static void backward_bias_grad_kernel(T* __restrict__ b_grad_ptr, const T* __restrict__ d_z_ptr) noexcept;
    static void backward_out_kernel(T* __restrict__ bwd_out_ptr, const T* __restrict__ w_ptr, const T* __restrict__ d_z_ptr) noexcept;

/* Private member */
private:

    WeightTensor w = InitPolicy::template create<WeightTensor>();
    WeightTensor w_grad = data::Tensor<T, inp_dim, out_dim>::zeros();
    BiasTensor b = InitPolicy::template create<BiasTensor>();
    BiasTensor b_grad = data::Tensor<T, out_dim>::zeros();
    data::Tensor<T, batch_size, out_dim> forward_out = data::Tensor<T, batch_size, out_dim>::zeros();
    data::Tensor<T, batch_size, inp_dim> forward_in = data::Tensor<T, batch_size, inp_dim>::zeros();
    data::Tensor<T, batch_size, inp_dim> backward_out = data::Tensor<T, batch_size, inp_dim>::zeros();

    const T learning_rate;
};

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
const data::Tensor<T, batch_size, out_dim>& Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::forward(const data::Tensor<T, batch_size, inp_dim>& x) noexcept
{
    forward_in = x;

    T* __restrict__ fwd_out_ptr = forward_out.data_ptr();
    const T* __restrict__ fwd_in_ptr = forward_in.data_ptr();
    const T* __restrict__ w_ptr = w.data_ptr();
    const T* __restrict__ b_ptr = b.data_ptr();

    math::batched_mat_vec_mul_w_bias_kernel<T, batch_size, out_dim, inp_dim>(fwd_out_ptr, fwd_in_ptr, w_ptr, b_ptr);

    return forward_out;
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
const data::Tensor<T, batch_size, inp_dim>& Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::backward(const data::Tensor<T, batch_size, out_dim>& d_z) noexcept
{
    T* __restrict__ w_grad_ptr = w_grad.data_ptr();
    const T* __restrict__ fwd_in_ptr = forward_in.data_ptr();
    const T* __restrict__ d_z_ptr = d_z.data_ptr();
    backward_weight_grad_kernel(w_grad_ptr, fwd_in_ptr, d_z_ptr);

    T* __restrict__ b_grad_ptr = b_grad.data_ptr();
    backward_bias_grad_kernel(b_grad_ptr, d_z_ptr);

    T* __restrict__ bwd_out_ptr = backward_out.data_ptr();
    const T* __restrict w_ptr = w.data_ptr();
    backward_out_kernel(bwd_out_ptr, w_ptr, d_z_ptr);

    return backward_out;
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::backward_weight_grad_kernel(T* __restrict__ w_grad_ptr, const T* __restrict__ fwd_in_ptr, const T* __restrict__ d_z_ptr) noexcept
{
    constexpr auto n = inp_dim * out_dim;
    for (std::size_t i = 0; i < n; ++i) {
        w_grad_ptr[i] = static_cast<T>(0.0);
    }

    if constexpr (batch_size > 64) {

        #pragma omp parallel for num_threads(4)
        for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            const std::size_t fwd_s1 = data::Tensor<T, batch_size, inp_dim>::strides[0] * b_idx;
            const std::size_t d_s1 = data::Tensor<T, batch_size, out_dim>::strides[0] * b_idx;

            for (std::size_t inp_col = 0; inp_col < inp_dim; ++inp_col) {
                const T f_in = fwd_in_ptr[fwd_s1 + inp_col];
                const std::size_t w_s1 = data::Tensor<T, inp_dim, out_dim>::strides[0] * inp_col;

                for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
                    w_grad_ptr[w_s1 + out_col] += f_in * d_z_ptr[d_s1 + out_col];
                }
            }
        }

    } else {
        for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            const std::size_t fwd_s1 = data::Tensor<T, batch_size, inp_dim>::strides[0] * b_idx;
            const std::size_t d_s1 = data::Tensor<T, batch_size, out_dim>::strides[0] * b_idx;

            for (std::size_t inp_col = 0; inp_col < inp_dim; ++inp_col) {
                const T f_in = fwd_in_ptr[fwd_s1 + inp_col];
                const std::size_t w_s1 = data::Tensor<T, inp_dim, out_dim>::strides[0] * inp_col;

                for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
                    w_grad_ptr[w_s1 + out_col] += f_in * d_z_ptr[d_s1 + out_col];
                }
            }
        }
    }
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::backward_bias_grad_kernel(T* __restrict__ b_grad_ptr, const T* __restrict__ d_z_ptr) noexcept
{
    for (std::size_t i = 0; i < out_dim; ++i) {
        b_grad_ptr[i] = static_cast<T>(0.0);
    }

    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        const std::size_t d_s1 = data::Tensor<T, batch_size, out_dim>::strides[0] * b_idx;

        for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
            b_grad_ptr[out_col] += d_z_ptr[d_s1 + out_col];
        }
    }
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::backward_out_kernel(T* __restrict__ bwd_out_ptr, const T* __restrict__ w_ptr, const T* __restrict__ d_z_ptr) noexcept
{

    if constexpr (batch_size > 64) {

        #pragma omp parallel for num_threads(4)
        for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            const std::size_t d_s1 = data::Tensor<T, batch_size, out_dim>::strides[0] * b_idx;
            const std::size_t b_s1 = data::Tensor<T, batch_size, inp_dim>::strides[0] * b_idx;

            for (std::size_t inp_col = 0; inp_col < inp_dim; ++inp_col) {
                T sum = static_cast<T>(0.0);

                const std::size_t w_s1 = data::Tensor<T, inp_dim, out_dim>::strides[0] * inp_col;
                for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
                    sum += d_z_ptr[d_s1 + out_col] * w_ptr[w_s1 + out_col];
                }

                bwd_out_ptr[b_s1 + inp_col] = sum;
            }
        }

    } else {
        for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            const std::size_t d_s1 = data::Tensor<T, batch_size, out_dim>::strides[0] * b_idx;
            const std::size_t b_s1 = data::Tensor<T, batch_size, inp_dim>::strides[0] * b_idx;

            for (std::size_t inp_col = 0; inp_col < inp_dim; ++inp_col) {
                T sum = static_cast<T>(0.0);

                const std::size_t w_s1 = data::Tensor<T, inp_dim, out_dim>::strides[0] * inp_col;
                for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
                    sum += d_z_ptr[d_s1 + out_col] * w_ptr[w_s1 + out_col];
                }

                bwd_out_ptr[b_s1 + inp_col] = sum;
            }
        }
    }
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::update() noexcept
{
    T* __restrict__ w_ptr = w.data_ptr();
    const T* __restrict__ w_grad_ptr = w_grad.data_ptr();
    T* __restrict__ b_ptr = b.data_ptr();
    const T* __restrict__ b_grad_ptr = b_grad.data_ptr();

    update_weights_kernel(w_ptr, w_grad_ptr, learning_rate);
    update_bias_kernel(b_ptr, b_grad_ptr, learning_rate);
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::update_weights_kernel(T* __restrict__ w_ptr, const T* __restrict__ w_grad_ptr, T learning_rate) noexcept
{
    constexpr auto n = inp_dim * out_dim;
    for (std::size_t i = 0; i < n; ++i) {
        w_ptr[i] -= learning_rate * w_grad_ptr[i];
    }
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::update_bias_kernel(T* __restrict__ b_ptr, const T* __restrict__ b_grad_ptr, T learning_rate) noexcept
{
    for (std::size_t i = 0; i < out_dim; ++i) {
        b_ptr[i] -= learning_rate * b_grad_ptr[i];
    }
}

} // namespace nn
} // namespace inpp

#endif //INSIGHTPP_LINEAR_H
