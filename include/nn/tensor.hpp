#pragma once

#include <cstddef>
#include <vector>

namespace nn {

class Tensor {
  public:
    Tensor(std::vector<size_t> shape) : _shape(shape), _data(element_count(shape)) {}

    const std::vector<size_t>& shape() const { return _shape; }

    size_t size() const { return _data.size(); }

    const float& get_element(const std::vector<size_t>& indices) const {
        return (_data[compute_index(indices)]);
    }

  private:
    std::vector<size_t> _shape;
    std::vector<float> _data;

    size_t compute_index(const std::vector<size_t>& indices) const {
        return indices[0] * _shape[1] + indices[1];
    }

    static size_t element_count(const std::vector<size_t>& shape) {
        size_t count = 1;
        for (size_t dim : shape) {
            count *= dim;
        }
        return count;
    }
};

} // namespace nn
