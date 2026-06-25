#ifndef INSIGHTPP_LINEAR_H
#define INSIGHTPP_LINEAR_H

#include <cstddef>

#include "../data/container.h"
#include "../data/initialization.h"
#include "../math/computation.h"

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
    math::batched_mat_vec_mul_w_bias(x, w, b, forward_out);

    return forward_out;
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
const data::Tensor<T, batch_size, inp_dim>& Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::backward(const data::Tensor<T, batch_size, out_dim>& d_z) noexcept
{
    // TODO: improve transpose access
    for (std::size_t inp_col = 0; inp_col < inp_dim; ++inp_col) {
        for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {

            T sum = static_cast<T>(0.0);
            for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
                sum += forward_in(b_idx, inp_col) * d_z(b_idx, out_col);
            }

            w_grad(inp_col, out_col) = sum;
        }
    }

    // TODO: improve transpose access
    for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
        T sum = static_cast<T>(0.0);
        for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            sum += d_z(b_idx, out_col);
        }
        b_grad(out_col) = sum;
    }

    // TODO: improve transpose access
    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (std::size_t inp_col = 0; inp_col < inp_dim; ++inp_col) {
            T sum = static_cast<T>(0.0);
            for (std::size_t out_col = 0; out_col < out_dim; ++out_col) {
                sum += d_z(b_idx, out_col) * w(inp_col, out_col);
            }
            backward_out(b_idx, inp_col) = sum;
        }
    }

    return backward_out;
}

template <typename T, std::size_t batch_size, std::size_t inp_dim, std::size_t out_dim, typename InitPolicy>
requires std::floating_point<T>
void Linear<T, batch_size, inp_dim, out_dim, InitPolicy>::update() noexcept
{
    for (std::size_t i = 0; i < inp_dim * out_dim; ++i) {
        w[i] -= learning_rate * w_grad[i];
    }

    for (std::size_t row = 0; row < out_dim; ++row) {
        b[row] -= learning_rate * b_grad[row];
    }
}

} // namespace nn
} // namespace inpp

#endif //INSIGHTPP_LINEAR_H
