#ifndef INSIGHTPP_CONTAINER_H
#define INSIGHTPP_CONTAINER_H

#include <array>
#include <random>
#include <iostream>

namespace inpp {
namespace data {

template <typename T>
concept TensorConcept =
requires
{
    typename T::value_type;

    { T::random(std::declval<typename T::value_type>(), std::declval<typename T::value_type>()) } -> std::same_as<T>;
    { T::zeros() } -> std::same_as<T>;
    { T::ones() }  -> std::same_as<T>;
};

template <typename T, std::size_t... Dims>
requires std::floating_point<T>
class Tensor {

/* Initialization */
public:

    Tensor() noexcept : data(total_size) {}
    explicit Tensor(T value) noexcept : data(total_size, value) {}

    static Tensor zeros() noexcept { return Tensor(static_cast<T>(0.0)); }
    static Tensor ones() noexcept { return Tensor(static_cast<T>(1.0)); }
    static Tensor random(T min = static_cast<T>(-0.01), T max = static_cast<T>(0.01)) noexcept
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::uniform_real_distribution<T> dist(min, max);

        Tensor out;

        for (auto& v : out.data) {
            v = dist(gen);
        }

        return out;
    }

/* Public Access */

    T& operator[](std::size_t i) noexcept { return data[i]; }
    const T& operator[](std::size_t i) const noexcept { return data[i]; }

    template <std::unsigned_integral... Indices>
    T& operator()(Indices... idx) noexcept;

    template <std::unsigned_integral... Indices>
    const T& operator()(Indices... idx) const noexcept;

    template <std::size_t... Indices>
    T& at() noexcept;

    template <std::size_t... Indices>
    const T& at() const noexcept;

    T* data_ptr() noexcept { return data.data(); }
    const T* data_ptr() const noexcept { return data.data(); }

/* Routines */
private:
    template <std::size_t N>
    static consteval std::array<std::size_t, N> compute_strides();

/* Compile time properties */
public:
    using value_type = T;

    static constexpr std::array<std::size_t, sizeof...(Dims)> shape = {Dims...};
    static constexpr auto strides = compute_strides<sizeof...(Dims)>();
    static constexpr auto rank = sizeof...(Dims);
    static constexpr std::size_t total_size = (Dims * ...);

/* Member */
private:
    std::vector<T> data;
};

template <typename T, std::size_t... Dims>
requires std::floating_point<T>
template <std::unsigned_integral... Indices>
T& Tensor<T, Dims...>::operator()(Indices... idx) noexcept
{
    static_assert(rank == sizeof...(Indices), "Rank mismatch");

    std::array<std::size_t, sizeof...(idx)> indices{static_cast<std::size_t>(idx)...};
    std::size_t offset = 0;

    for (std::size_t i = 0; i < rank; ++i) {
        offset += strides[i] * indices[i];
    }

    return data[offset];
}

template <typename T, std::size_t... Dims>
requires std::floating_point<T>
template <std::unsigned_integral... Indices>
const T& Tensor<T, Dims...>::operator()(Indices... idx) const noexcept
{
    static_assert(rank == sizeof...(Indices), "Rank mismatch!");

    std::array<std::size_t, sizeof...(idx)> indices{static_cast<std::size_t>(idx)...};
    std::size_t offset = 0;

    for (std::size_t i = 0; i < rank; ++i) {
        offset += strides[i] * indices[i];
    }

    return data[offset];
}

template <typename T, std::size_t... Dims>
requires std::floating_point<T>
template <std::size_t... Indices>
T& Tensor<T, Dims...>::at() noexcept
{
    static_assert(rank == sizeof...(Indices), "Rank mismatch!");

    static_assert([]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((Indices < shape[Is]) && ...);
    }(std::make_index_sequence<sizeof...(Indices)>{}), "Index out of bounds!");

    constexpr std::size_t offset = []<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((Indices * strides[Is]) + ...);
    }(std::make_index_sequence<rank>{});

    return data[offset];
}

template <typename T, std::size_t... Dims>
requires std::floating_point<T>
template <std::size_t... Indices>
const T& Tensor<T, Dims...>::at() const noexcept
{
    static_assert(rank == sizeof...(Indices), "Rank mismatch!");

    static_assert([]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((Indices < shape[Is]) && ...);
    }(std::make_index_sequence<sizeof...(Indices)>{}), "Index out of bounds!");

    constexpr std::size_t offset = []<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((Indices * strides[Is]) + ...);
    }(std::make_index_sequence<rank>{});

    return data[offset];
}

template <typename T, std::size_t... Dims>
requires std::floating_point<T>
template <std::size_t N>
consteval std::array<std::size_t, N> Tensor<T, Dims...>::compute_strides()
{
    std::array<std::size_t, N> strides_tmp;

    strides_tmp[N - 1] = 1;

    for (std::size_t i = N - 1; i > 0; --i) {
        strides_tmp[i - 1] = strides_tmp[i] * shape[i];
    }

    return strides_tmp;
}

} // namespace data
} // namespace inpp

#endif //INSIGHTPP_CONTAINER_H