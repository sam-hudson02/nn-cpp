#include "nn/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <iostream>

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

TEST_CASE("Tensor can write an element with data method and read it back", "[tensor]") {
    nn::Tensor tensor({2, 3});
    const float value = 42.0f;
    const size_t index = 3;

    REQUIRE(tensor.data()[index] == 0.0f);

    tensor.data()[index] = value;
    const float read_value = tensor.data()[index];

    REQUIRE(read_value == value);
}

TEST_CASE("Tensor throws invalid argument exception for empty or zero dimensions", "[tensor]") {
    REQUIRE_THROWS_AS(nn::Tensor({}), std::invalid_argument);
    REQUIRE_THROWS_AS(nn::Tensor({0, 3}), std::invalid_argument);
    REQUIRE_THROWS_AS(nn::Tensor({2, 0}), std::invalid_argument);
}
