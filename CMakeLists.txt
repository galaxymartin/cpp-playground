cmake_minimum_required(VERSION 3.31)
project(cpp-playground VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_LINKER_TYPE MOLD)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG True)

option(ENABLE_ADDRESS_SANITIZER "Build with AddressSanitizer for leak/memory checking" OFF)
if(ENABLE_ADDRESS_SANITIZER)
  message(STATUS "Building with AddressSanitizer enabled")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O1 -mavx512f -fopenmp)
  add_link_options(-fsanitize=address -fopenmp)
else()
  add_compile_options(-O3 -s -fdata-sections -ffunction-sections -mavx512f -fopenmp)
  add_link_options(-Wl,--gc-sections -Wl,--strip-all -fopenmp)
endif()

find_package(GTest CONFIG QUIET)
if (NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()
find_package(Threads REQUIRED)
find_package(Arrow CONFIG QUIET)
find_package(ArrowCompute CONFIG QUIET)
find_package(Parquet CONFIG QUIET)
find_package(boost_json CONFIG QUIET)
find_package(boost_asio CONFIG QUIET)
find_package(boost_program_options CONFIG QUIET)
find_package(boost_headers CONFIG QUIET)
find_package(boost_bind CONFIG QUIET)
find_package(boost_log CONFIG QUIET)
find_package(PkgConfig)

include(GoogleTest)
add_executable(tests test.cpp matrix.cpp simd/avx512/matrix_avx512.cpp simd/sse41/matrix_sse41.cpp)
target_link_libraries(tests PRIVATE
    GTest::gtest_main
    GTest::gmock_main
)

gtest_discover_tests(tests)

add_executable(cpp-playground main.cpp my_class.cpp printer.cpp matrix.cpp simd/avx512/matrix_avx512.cpp simd/sse41/matrix_sse41.cpp)

# Note: linking to Boost targets removed to allow building tests without external Boost

if (Arrow_FOUND)
  add_executable(arrow_demo arrow_main.cpp)
  target_link_libraries(arrow_demo PRIVATE
      "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>"
      "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,ArrowCompute::arrow_compute_static,ArrowCompute::arrow_compute_shared>"
      "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>"
      Boost::program_options
  )
else()
  message(STATUS "Arrow not found; skipping arrow_demo target")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
