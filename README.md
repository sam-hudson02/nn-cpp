# nn (C++)

A C++17 port of the Python neural network in [Neural-Network](../Neural-Network) (`~/code/Neural-Network`), with hand-written compute kernels
tested and benchmarked against Eigen.

## Layout

```
include/nn/   public headers (Tensor, gemm)
src/          kernel implementations
tests/        Catch2 tests; eigen_ref.hpp is for Eigen reference helpers
bench/        Google Benchmark suite
```

Dependencies (Catch2, Eigen, Google Benchmark) are downloaded by CMake on first configure.

## Build and test

```sh
# Debug build with AddressSanitizer + UBSan
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

# Release build for benchmarks
cmake --preset release
cmake --build --preset release
./build/release/nn_bench
```

Run a subset with `./build/debug/nn_tests "[gemm]"` or `./build/release/nn_bench --benchmark_filter=nn`.

Format with `clang-format -i src/*.cpp include/nn/*.hpp tests/*.cpp bench/*.cpp`.
