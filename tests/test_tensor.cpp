#include "nn/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <iostream>
#include <type_traits>

static_assert(!std::is_default_constructible_v<nn::Tensor>,
              "a Tensor must always be given a shape");

TEST_CASE("Tensor reports its shape and size correctly", "[tensor]") {
    const nn::Tensor tensor({2, 3});

    REQUIRE(tensor.shape() == std::vector<size_t>({2, 3}));
    REQUIRE(tensor.size() == 6);
}

TEST_CASE("Tensor all elements are initialized to zero", "[tensor]") {
    const size_t rows = 2;
    const size_t cols = 3;
    const nn::Tensor tensor({rows, cols});
    const size_t size = tensor.size();

    for (size_t i = 0; i < size; ++i) {
        const float p = tensor.data()[i];
        std::cout << "Element " << i << ": " << p << std::endl;
        REQUIRE(p == 0.0f);
    }
}

TEST_CASE("Tensor can get elements by index", "[tensor]") {
    const size_t rows = 2;
    const size_t cols = 3;

    nn::Tensor tensor({2, 3});

    for (size_t row_idx = 0; row_idx < rows; ++row_idx) {
        for (size_t col_idx = 0; col_idx < cols; ++col_idx) {
            const size_t index = row_idx * cols + col_idx;
            const float input_value = float(row_idx * col_idx);

            tensor.data()[index] = input_value;
            const float stored_value = tensor.element({row_idx, col_idx});

            REQUIRE(stored_value == input_value);
        }
    }
}

TEST_CASE("Tensor can write an element with data method and read it back", "[tensor]") {
    nn::Tensor tensor({2, 3});
    const float value = 42.0f;

    for (size_t i = 0; i < tensor.size(); ++i) {
        REQUIRE(tensor.data()[i] == 0.0f);

        tensor.data()[i] = value;
        const float read_value = tensor.data()[i];
        REQUIRE(read_value == value);
    }
}

TEST_CASE("Tensor throws invalid argument exception for empty or zero dimensions", "[tensor]") {
    REQUIRE_THROWS_AS(nn::Tensor({}), std::invalid_argument);
    REQUIRE_THROWS_AS(nn::Tensor({0, 3}), std::invalid_argument);
    REQUIRE_THROWS_AS(nn::Tensor({2, 0}), std::invalid_argument);
}
