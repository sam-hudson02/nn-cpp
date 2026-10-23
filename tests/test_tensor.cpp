#include "nn/tensor.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Tensor reports its shape and size correctly", "[tensor]") {
    const nn::Tensor tensor({2, 3});

    REQUIRE(tensor.shape() == std::vector<size_t>({2, 3}));
    REQUIRE(tensor.size() == 6);
}
