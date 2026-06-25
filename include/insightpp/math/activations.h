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
    static T relu(const T x) { return x > static_cast<T>(0.0) ? x : static_cast<T>(0.0); }
    static T relu_d(const T x) { return x > static_cast<T>(0.0) ? static_cast<T>(1.0) : static_cast<T>(0.0); }

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

    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (std::size_t row = 0; row < Rows_m; ++row) {
            last_out(b_idx, row) = relu(x(b_idx, row));
        }
    }

    return last_out;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
const data::Tensor<T, batch_size, Rows_m>& ReLU<T, batch_size, Rows_m>::backward(const data::Tensor<T, batch_size, Rows_m>& d_z)
{
    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (std::size_t row = 0; row < Rows_m; ++row) {
            backward_out(b_idx, row) = d_z(b_idx, row) * relu_d(forward_in(b_idx, row));
        }
    }

    return backward_out;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
class MSE {


/* API */
public:
    T forward(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction);
    const data::Tensor<T, batch_size, Rows_m>& backward();

/* Routines */
private:
    void mse(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction);
    void mse_d(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction);

/* Private Member */
private:
    T loss = static_cast<T>(0.0);
    data::Tensor<T, batch_size, Rows_m> gradient = data::Tensor<T, batch_size, Rows_m>::zeros();

};


template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
T MSE<T, batch_size, Rows_m>::forward(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction)
{
    mse(label, prediction);
    mse_d(label, prediction);
    return loss;
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
const data::Tensor<T, batch_size, Rows_m>& MSE<T, batch_size, Rows_m>::backward() { return gradient; }

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
void MSE<T, batch_size, Rows_m>::mse(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction)
{
    loss = static_cast<T>(0.0);

    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (std::size_t row = 0; row < Rows_m; ++row) {
            const auto tmp = prediction(b_idx, row) - label(b_idx, row);
            loss += tmp * tmp;
        }
    }

    loss = loss / static_cast<T>(batch_size * Rows_m);
}

template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
void MSE<T, batch_size, Rows_m>::mse_d(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction)
{
    constexpr T scale = static_cast<T>(2.0) / static_cast<T>(Rows_m * batch_size);

    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (std::size_t row = 0; row < Rows_m; ++row) {
            gradient(b_idx, row) = scale * (prediction(b_idx, row) - label(b_idx, row));
        }
    }
}


} // namespace math
} // namespace inpp

#endif //INSIGHTPP_ACTIVATIONS_H
