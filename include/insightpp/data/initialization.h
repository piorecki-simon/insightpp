#ifndef INSIGHTPP_INITIALIZATION_H
#define INSIGHTPP_INITIALIZATION_H

#include "container.h"

#include <algorithm>
#include <sys/stat.h>

namespace inpp {
namespace init {

template <auto from, auto to>
struct RandomFromTo {
    template <data::TensorConcept T>
    static T create()
    {
        using Scalar = typename T::value_type;

        return T::random(static_cast<Scalar>(from), static_cast<Scalar>(to));
    }
};

struct Zeros {
    template <data::TensorConcept T>
    static T create()
    {
        return T::zeros();
    }
};

struct Ones {
    template <data::TensorConcept T>
    static T create()
    {
        return T::ones();
    }
};


template <typename T, std::size_t batch_size, std::size_t Rows_m>
requires std::floating_point<T>
data::Tensor<T, batch_size> is_correct_label(const data::Tensor<T, batch_size, Rows_m>& label, const data::Tensor<T, batch_size, Rows_m>& prediction)
{
    auto out = data::Tensor<T, batch_size>::zeros();

    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        T max_val = static_cast<T>(-1000000.0);
        std::size_t max_idx = 0;
        for (std::size_t row = 0; row < Rows_m; ++row) {
            if (prediction(b_idx, row) > max_val) {
                max_val = prediction(b_idx, row);
                max_idx = row;
            }
        }

        if (label(b_idx, max_idx) == static_cast<T>(1.0)) {
            out(b_idx) = static_cast<T>(1.0);
        }
    }

    return out;

    /*
     * const auto res = std::max_element(prediction.data.begin(), prediction.data.end());
     * const auto idx = std::distance(prediction.data.begin(), res);
     */
}


} // namespace init
} // namespace inpp

#endif //INSIGHTPP_INITIALIZATION_H
