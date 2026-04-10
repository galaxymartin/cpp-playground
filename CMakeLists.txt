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

find_package(GTest REQUIRED CONFIG)
find_package(Threads REQUIRED)
find_package(Arrow REQUIRED CONFIG)
find_package(ArrowCompute CONFIG REQUIRED)
find_package(Parquet REQUIRED CONFIG)
find_package(boost_json REQUIRED CONFIG)
find_package(boost_asio REQUIRED CONFIG)
find_package(boost_program_options REQUIRED CONFIG)
find_package(boost_headers REQUIRED CONFIG)
find_package(boost_bind REQUIRED CONFIG)
find_package(boost_log REQUIRED CONFIG)
find_package(PkgConfig)

include(GoogleTest)
add_executable(tests test.cpp matrix.cpp)
target_link_libraries(tests PRIVATE
    GTest::gtest_main
    GTest::gmock_main
)

gtest_discover_tests(tests)

add_executable(cpp-playground main.cpp my_class.cpp printer.cpp matrix.cpp)
add_executable(arrow_demo arrow_main.cpp)
target_link_libraries(cpp-playground PRIVATE 
    Boost::json 
    Boost::headers
    Boost::asio
    Boost::bind
    Boost::log
)

target_link_libraries(arrow_demo PRIVATE
    "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>"
    "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,ArrowCompute::arrow_compute_static,ArrowCompute::arrow_compute_shared>"
    "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>"
    Boost::program_options
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
