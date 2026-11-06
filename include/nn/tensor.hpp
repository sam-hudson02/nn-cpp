#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nn {

class Tensor {
  public:
    Tensor(std::vector<size_t> shape) : _shape(shape), _data(element_count(shape)) {
        if (_shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty");
        }
        if (std::any_of(_shape.begin(), _shape.end(), [](size_t dim) { return dim == 0; })) {
            throw std::invalid_argument("Shape dimensions must be greater than zero");
        }
        _strides = compute_strides(_shape);
    }

    const std::vector<size_t>& shape() const { return _shape; }

    size_t size() const { return _data.size(); }

    const float& element(const std::vector<size_t>& indices) const {
        validate_indices(indices, _shape);

        return (_data[compute_index(indices)]);
    }

    void set_element(const std::vector<size_t>& indices, float value) {
        validate_indices(indices, _shape);

        _data[compute_index(indices)] = value;
    }

    const float* data() const { return _data.data(); }

    float* data() { return _data.data(); }

  private:
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    std::vector<float> _data;

    size_t compute_index(const std::vector<size_t>& indices) const {
        // return indices[0] * _shape[1] + indices[1];
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            index += indices[i] * _strides[i];
        }
        return index;
    }

    static void validate_indices(const std::vector<size_t>& indices,
                                 const std::vector<size_t>& shape) {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Number of indices does not match tensor dimensions");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of range for dimension " + std::to_string(i));
            }
        }
    }

    static size_t element_count(const std::vector<size_t>& shape) {
        size_t count = 1;
        for (size_t dim : shape) {
            count *= dim;
        }
        return count;
    }

    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
};

} // namespace nn
