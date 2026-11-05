#include "nn/tensor.hpp"
#include "nn/transform.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Transpose can be applied to an mxn tensor giving output shape nxm", "[transform]") {
    const nn::Tensor input({2, 3});
    const nn::Tensor output = nn::transpose(input);
    REQUIRE(output.shape() == std::vector<size_t>({3, 2}));
}
