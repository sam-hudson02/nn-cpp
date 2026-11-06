#include "nn/tensor.hpp"
#include "nn/transform.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Transpose can be applied to an mxn tensor giving output shape nxm", "[transform]") {
    const nn::Tensor input({2, 3});
    const nn::Tensor output = nn::transpose(input);
    REQUIRE(output.shape() == std::vector<size_t>({3, 2}));
}

TEST_CASE("Transpose moves each element to its swapped position", "[transform]") {
    nn::Tensor input({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = float(i);
    }

    const nn::Tensor output = nn::transpose(input);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(output.element({j, i}) == input.element({i, j}));
        }
    }
}

TEST_CASE("Transposing twice gives back the original tensor", "[transform]") {
    nn::Tensor input({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = float(i + 1);
    }

    const nn::Tensor twice = nn::transpose(nn::transpose(input));

    REQUIRE(twice.shape() == input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        REQUIRE(twice.data()[i] == input.data()[i]);
    }
}
