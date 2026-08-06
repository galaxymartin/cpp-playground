cmake_minimum_required(VERSION 4.3)

# Use workspace-local vcpkg overlay ports to patch dependency build quirks.
set(VCPKG_OVERLAY_PORTS "${CMAKE_SOURCE_DIR}/vcpkg-overlay-ports")

project(cpp-playground VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_LINKER_TYPE MOLD)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

option(ENABLE_ADDRESS_SANITIZER "Build with AddressSanitizer for leak/memory checking" OFF)
set(PROJECT_COMPILE_OPTIONS
  -mavx512f
  -fopenmp
)
set(PROJECT_LINK_OPTIONS
  -fopenmp
)

if(ENABLE_ADDRESS_SANITIZER)
  message(STATUS "Building with AddressSanitizer enabled")
  list(APPEND PROJECT_COMPILE_OPTIONS -fsanitize=address -fno-omit-frame-pointer -g -O1)
  list(APPEND PROJECT_LINK_OPTIONS -fsanitize=address)
else()
  # Keep symbols and minimal optimization in Debug so source breakpoints bind reliably.
  list(APPEND PROJECT_COMPILE_OPTIONS
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g3>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    $<$<NOT:$<CONFIG:Debug>>:-O3>
    $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>
    $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
  )
  list(APPEND PROJECT_LINK_OPTIONS
    $<$<NOT:$<CONFIG:Debug>>:-Wl,--gc-sections>
    $<$<NOT:$<CONFIG:Debug>>:-Wl,--strip-all>
  )
endif()

function(apply_project_options target_name)
  target_compile_options(${target_name} PRIVATE ${PROJECT_COMPILE_OPTIONS})
  target_link_options(${target_name} PRIVATE ${PROJECT_LINK_OPTIONS})
endfunction()

find_package(GTest CONFIG QUIET)
if (NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

# Pre-discover utf8proc so Arrow's finder doesn't fall back to deprecated
# unofficial-utf8proc package names and emit a warning.
find_package(utf8proc CONFIG QUIET)

find_package(Arrow CONFIG QUIET)
find_package(ArrowCompute CONFIG QUIET)
find_package(Parquet CONFIG QUIET)
foreach(boost_pkg IN ITEMS json asio accumulators log)
  find_package(boost_${boost_pkg} CONFIG QUIET)
endforeach()

include(GoogleTest)
add_library(simd STATIC
  simd/avx512/matrix_avx512.cpp
  simd/sse41/matrix_sse41.cpp
  simd/dispatch.cpp
)
target_compile_options(simd PRIVATE ${PROJECT_COMPILE_OPTIONS})
target_include_directories(simd PUBLIC
  ${CMAKE_SOURCE_DIR}/simd/avx512
  ${CMAKE_SOURCE_DIR}/simd/sse41
  ${CMAKE_SOURCE_DIR}/simd
)

add_executable(tests test.cpp matrix.cpp)
apply_project_options(tests)
target_link_libraries(tests PRIVATE
    simd
    GTest::gtest_main
    GTest::gmock_main
)

gtest_discover_tests(tests)

add_executable(cpp-playground main.cpp my_class.cpp printer.cpp matrix.cpp)
apply_project_options(cpp-playground)
target_link_libraries(cpp-playground PRIVATE
    simd
    GTest::gmock
    $<$<BOOL:${boost_json_FOUND}>:Boost::json>
    $<$<BOOL:${boost_asio_FOUND}>:Boost::asio>
    $<$<BOOL:${boost_log_FOUND}>:Boost::log>
    $<$<BOOL:${boost_accumulators_FOUND}>:Boost::accumulators>
)

if (Arrow_FOUND)
  find_package(boost_program_options CONFIG QUIET)
  add_executable(arrow_demo arrow_main.cpp)
  apply_project_options(arrow_demo)
  target_link_libraries(arrow_demo PRIVATE
      "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>"
      "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,ArrowCompute::arrow_compute_static,ArrowCompute::arrow_compute_shared>"
      "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>"
      $<$<BOOL:${boost_program_options_FOUND}>:Boost::program_options>
  )
else()
  message(STATUS "Arrow not found; skipping arrow_demo target")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
