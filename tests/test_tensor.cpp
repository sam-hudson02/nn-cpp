#include "nn/tensor.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Tensor reports its shape and size correctly", "[tensor]") {
    const nn::Tensor tensor({2, 3});

    REQUIRE(tensor.shape() == std::vector<size_t>({2, 3}));
    REQUIRE(tensor.size() == 6);
}

TEST_CASE("Tensor all elements are initialized to zero", "[tensor]") {
    const size_t rows = 2;
    const size_t cols = 3;
    const nn::Tensor tensor({rows, cols});
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            REQUIRE(tensor.get_element({i, j}) == 0.0f);
        }
    }
}
