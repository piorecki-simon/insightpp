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
    const data::Tensor<T, batch_size, Rows_m>& forward(const data::Tensor<T, batch_size, Rows_m>& x);
    const data::Tensor<T, batch_size, Rows_m>& backward(const data::Tensor<T, batch_size, Rows_m>& d_z);

/* Routines */
private:
    static void forward_kernel(T* __restrict out_ptr, const T* __restrict fwd_in_ptr) noexcept;
    static void backward_kernel(T* __restrict out_ptr, const T* __restrict fwd_in_ptr, const T* __restrict d_z_ptr) noexcept;

/* Private member */
private:
    data::Tensor<T, batch_size, Rows_m> forward_in = data::Tensor<T, batch_size, Rows_m>::zeros();
    data::Tensor<T, batch_size, Rows_m> last_out = data::Tensor<T, batch_size, Rows_m>::zeros();
    data::Tensor<T, batch_size, Rows_m> backward_out = data::Tensor<T, batch_size, Rows_m>::zeros();
};


template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
const data::Tensor<T, batch_size, Rows_m>& ReLU<T, batch_size, Rows_m>::forward(const data::Tensor<T, batch_size, Rows_m>& x)
{
    forward_in = x;

    T* __restrict out_ptr = last_out.data_ptr();
    const T* __restrict fwd_in_ptr = forward_in.data_ptr();

    forward_kernel(out_ptr, fwd_in_ptr);

    return last_out;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
void ReLU<T, batch_size, Rows_m>::forward_kernel(T* __restrict out_ptr, const T* __restrict fwd_in_ptr) noexcept
{
    for (std::size_t i = 0; i < batch_size * Rows_m; ++i) {
        out_ptr[i] = fwd_in_ptr[i] > static_cast<T>(0.0) ? fwd_in_ptr[i] : static_cast<T>(0.0);
    }
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
const data::Tensor<T, batch_size, Rows_m>& ReLU<T, batch_size, Rows_m>::backward(const data::Tensor<T, batch_size, Rows_m>& d_z)
{
    T* __restrict bwd_out_ptr = backward_out.data_ptr();
    const T* __restrict d_z_ptr = d_z.data_ptr();
    const T* __restrict fwd_in_ptr = forward_in.data_ptr();

    backward_kernel(bwd_out_ptr, fwd_in_ptr, d_z_ptr);

    return backward_out;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m> requires std::floating_point<T>
void ReLU<T, batch_size, Rows_m>::backward_kernel(T* __restrict out_ptr, const T* __restrict fwd_in_ptr, const T* __restrict d_z_ptr) noexcept
{
    for (std::size_t i = 0; i < batch_size * Rows_m; ++i) {
        out_ptr[i] = fwd_in_ptr[i] > static_cast<T>(0.0) ? d_z_ptr[i] : static_cast<T>(0.0);
    }
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
class MSE {

/* API */
public:
    T forward(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction);
    const data::Tensor<T, batch_size, Rows_m>& backward() noexcept { return gradient; }

/* Routines */
private:
    static T forward_kernel(const T* __restrict lbl_ptr, const T* __restrict pred_ptr) noexcept;
    static void backward_kernel(T* __restrict grad_ptr, const T* __restrict lbl_ptr, const T* __restrict pred_ptr) noexcept;

/* Private Member */
private:
    T loss = static_cast<T>(0.0);
    data::Tensor<T, batch_size, Rows_m> gradient = data::Tensor<T, batch_size, Rows_m>::zeros();

};

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
T MSE<T, batch_size, Rows_m>::forward(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction)
{
    const T* __restrict lbl_ptr = label.data_ptr();
    const T* __restrict pred_ptr = prediction.data_ptr();
    T* __restrict grad_ptr = gradient.data_ptr();

    loss = forward_kernel(lbl_ptr, pred_ptr);
    backward_kernel(grad_ptr, lbl_ptr, pred_ptr);
    return loss;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
T MSE<T, batch_size, Rows_m>::forward_kernel(const T* __restrict lbl_ptr, const T* __restrict pred_ptr) noexcept
{
    T loss = static_cast<T>(0.0);

    for (std::size_t i = 0; i < batch_size * Rows_m; ++i) {
        const auto tmp = pred_ptr[i] - lbl_ptr[i];
        loss += tmp * tmp;
    }

    return loss / static_cast<T>(batch_size * Rows_m);
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
void MSE<T, batch_size, Rows_m>::backward_kernel(T* __restrict grad_ptr, const T* __restrict lbl_ptr, const T* __restrict pred_ptr) noexcept
{
    constexpr T scale = static_cast<T>(2.0) / static_cast<T>(Rows_m * batch_size);

    for (std::size_t i = 0; i < batch_size * Rows_m; ++i) {
        grad_ptr[i] = scale * (pred_ptr[i] - lbl_ptr[i]);
    }
}


} // namespace math
} // namespace inpp

#endif //INSIGHTPP_ACTIVATIONS_H
