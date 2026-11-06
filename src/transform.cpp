#include "nn/transform.hpp"
#include "nn/tensor.hpp"

namespace nn {

Tensor transpose(const Tensor& input) {
    const std::vector<size_t>& input_shape = input.shape();
    std::vector<size_t> output_shape(input_shape.rbegin(), input_shape.rend());
    Tensor output = Tensor(output_shape);
    for (size_t i = 0; i < input_shape[0]; ++i) {
        for (size_t j = 0; j < input_shape[1]; ++j) {
            output.set_element({j, i}, input.element({i, j}));
        }
    }
    return output;
}

} // namespace nn
