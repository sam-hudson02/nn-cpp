cmake_minimum_required(VERSION 3.24)
project(nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd / clang-tidy

option(NN_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
option(NN_BUILD_TESTS "Build the Catch2 test suite" ON)
option(NN_BUILD_BENCH "Build the Google Benchmark suite" ON)

# Warnings applied to our own targets only, so third-party code can't break the build.
add_library(nn_warnings INTERFACE)
target_compile_options(nn_warnings INTERFACE -Wall -Wextra -Wshadow -Wconversion -Werror)

if(NN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- the library -----------------------------------------------------------
add_library(nn
  src/gemm.cpp
)
target_include_directories(nn PUBLIC include)
target_link_libraries(nn PRIVATE nn_warnings)

# ---- third-party (only needed for tests and benchmarks) --------------------
include(FetchContent)

if(NN_BUILD_TESTS OR NN_BUILD_BENCH)
  # Eigen is the reference implementation: correctness oracle and speed baseline.
  # SOURCE_SUBDIR points nowhere so CMake only downloads the headers instead of
  # running Eigen's own (large) build.
  FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    SOURCE_SUBDIR headers-only)
  FetchContent_MakeAvailable(eigen)
  add_library(eigen_headers INTERFACE)
  target_include_directories(eigen_headers SYSTEM INTERFACE ${eigen_SOURCE_DIR})
endif()

if(NN_BUILD_TESTS)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(Catch2)

  enable_testing()
  add_executable(nn_tests
    tests/test_tensor.cpp
    tests/test_gemm.cpp
  )
  target_link_libraries(nn_tests PRIVATE nn nn_warnings eigen_headers Catch2::Catch2WithMain)

  include(Catch)
  catch_discover_tests(nn_tests)
endif()

if(NN_BUILD_BENCH)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(benchmark)
  # benchmark builds itself with -pedantic-errors; clang 22 flags its use of __COUNTER__.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
  endif()

  add_executable(nn_bench bench/bench_gemm.cpp)
  target_link_libraries(nn_bench PRIVATE nn nn_warnings eigen_headers benchmark::benchmark_main)
endif()
