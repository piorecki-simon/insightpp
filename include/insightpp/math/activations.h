#ifndef INSIGHTPP_ACTIVATIONS_H
#define INSIGHTPP_ACTIVATIONS_H

#include "../data/container.h"

namespace inpp {
namespace math {

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
class ReLU {

/* API */
public:
    const std::unique_ptr<data::Tensor<T, batch_size, Rows_m>>& forward(const data::Tensor<T, batch_size, Rows_m>& x);
    const std::unique_ptr<data::Tensor<T, batch_size, Rows_m>>& backward(const data::Tensor<T, batch_size, Rows_m>& d_z);

/* Routines */
private:
    static void forward_kernel(T* __restrict__ out_ptr, const T* __restrict__ fwd_in_ptr) noexcept;
    static void backward_kernel(T* __restrict__ out_ptr, const T* __restrict__ fwd_in_ptr, const T* __restrict__ d_z_ptr) noexcept;

/* Private member */
private:
    const data::Tensor<T, batch_size, Rows_m>* forward_in = nullptr;
    std::unique_ptr<data::Tensor<T, batch_size, Rows_m>> last_out = std::make_unique<data::Tensor<T, batch_size, Rows_m>>(data::Tensor<T, batch_size, Rows_m>::zeros());
    std::unique_ptr<data::Tensor<T, batch_size, Rows_m>> backward_out = std::make_unique<data::Tensor<T, batch_size, Rows_m>>(data::Tensor<T, batch_size, Rows_m>::zeros());
};


template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
const std::unique_ptr<data::Tensor<T, batch_size, Rows_m>>& ReLU<T, batch_size, Rows_m>::forward(const data::Tensor<T, batch_size, Rows_m>& x)
{
    forward_in = &x;

    T* __restrict__ out_ptr = (*last_out).data_ptr();
    const T* __restrict__ fwd_in_ptr = (*forward_in).data_ptr();

    forward_kernel(out_ptr, fwd_in_ptr);

    return last_out;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
void ReLU<T, batch_size, Rows_m>::forward_kernel(T* __restrict__ out_ptr, const T* __restrict__ fwd_in_ptr) noexcept
{
    constexpr auto n = batch_size * Rows_m;
    for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = std::max(fwd_in_ptr[i], T(0));
    }
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
const std::unique_ptr<data::Tensor<T, batch_size, Rows_m>>& ReLU<T, batch_size, Rows_m>::backward(const data::Tensor<T, batch_size, Rows_m>& d_z)
{
    T* __restrict__ bwd_out_ptr = (*backward_out).data_ptr();
    const T* __restrict__ d_z_ptr = d_z.data_ptr();
    const T* __restrict__ fwd_in_ptr = (*forward_in).data_ptr();

    backward_kernel(bwd_out_ptr, fwd_in_ptr, d_z_ptr);

    return backward_out;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m> requires std::floating_point<T>
void ReLU<T, batch_size, Rows_m>::backward_kernel(T* __restrict__ out_ptr, const T* __restrict__ fwd_in_ptr, const T* __restrict__ d_z_ptr) noexcept
{
    constexpr auto n = batch_size * Rows_m;
    for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = fwd_in_ptr[i] > T(0) ? d_z_ptr[i] : T(0);
    }
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
class MSE {

/* API */
public:
    T forward(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction);
    const std::unique_ptr<data::Tensor<T, batch_size, Rows_m>>& backward() noexcept { return gradient; }

/* Routines */
private:
    static T forward_kernel(const T* __restrict__ lbl_ptr, const T* __restrict__ pred_ptr) noexcept;
    static void backward_kernel(T* __restrict__ grad_ptr, const T* __restrict__ lbl_ptr, const T* __restrict__ pred_ptr) noexcept;

/* Private Member */
private:
    T loss = T(0);
    const std::unique_ptr<data::Tensor<T, batch_size, Rows_m>> gradient = std::make_unique<data::Tensor<T, batch_size, Rows_m>>(data::Tensor<T, batch_size, Rows_m>::zeros());

};

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
T MSE<T, batch_size, Rows_m>::forward(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction)
{
    const T* __restrict__ lbl_ptr = label.data_ptr();
    const T* __restrict__ pred_ptr = prediction.data_ptr();
    T* __restrict__ grad_ptr = (*gradient).data_ptr();

    loss = forward_kernel(lbl_ptr, pred_ptr);
    backward_kernel(grad_ptr, lbl_ptr, pred_ptr);
    return loss;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
T MSE<T, batch_size, Rows_m>::forward_kernel(const T* __restrict__ lbl_ptr, const T* __restrict__ pred_ptr) noexcept
{
    constexpr auto n = batch_size * Rows_m;
    T sum = 0;

    #pragma omp simd reduction(+:sum)
    for (std::size_t i = 0; i < n; ++i) {
        const T diff = pred_ptr[i] - lbl_ptr[i];
        sum += diff * diff;
    }
    return sum / n;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
void MSE<T, batch_size, Rows_m>::backward_kernel(T* __restrict__ grad_ptr, const T* __restrict__ lbl_ptr, const T* __restrict__ pred_ptr) noexcept
{
    constexpr T scale = static_cast<T>(2.0) / static_cast<T>(Rows_m * batch_size);
    constexpr auto n = batch_size * Rows_m;

    for (std::size_t i = 0; i < n; ++i) {
        grad_ptr[i] = scale * (pred_ptr[i] - lbl_ptr[i]);
    }
}


} // namespace math
} // namespace inpp

#endif //INSIGHTPP_ACTIVATIONS_H
