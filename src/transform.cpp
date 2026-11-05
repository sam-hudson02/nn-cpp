#include "nn/transform.hpp"
#include "nn/tensor.hpp"

namespace nn {

Tensor transpose(const Tensor& input) {
    const std::vector<size_t>& input_shape = input.shape();
    std::vector<size_t> output_shape(input_shape.rbegin(), input_shape.rend());
    return Tensor(output_shape);
}

} // namespace nn
