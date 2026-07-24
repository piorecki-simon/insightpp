cmake_minimum_required(VERSION 3.28)
project(insightpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENBLAS "Enable OpenBLAS for GEMMs" OFF)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(Torch REQUIRED)

# Enable testing
enable_testing()

#
# Library
#

add_library(insightpp
    STATIC
        src/insightpp.cpp
)

target_link_libraries(insightpp PRIVATE OpenMP::OpenMP_CXX BLAS::BLAS ${TORCH_LIBRARIES})

target_include_directories(insightpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/data
        ${CMAKE_CURRENT_SOURCE_DIR}/include/linear
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math
)

target_compile_features(insightpp
    PUBLIC
        cxx_std_20
)

# Optimization flags
target_compile_options(insightpp PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:Release>:-fopt-info-vec-all>
)

#
# Test executable
#

add_executable(testinsightpp app/main.cpp)

target_compile_features(testinsightpp
    PUBLIC
        cxx_std_20
)

target_compile_options(testinsightpp PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:Release>:-fopt-info-vec-all>
)

target_link_libraries(testinsightpp
    PRIVATE
        insightpp
        OpenMP::OpenMP_CXX
        ${TORCH_LIBRARIES}
)

add_executable(test_libtorch_insightpp app/libtorch_main_test.cpp)

target_compile_features(test_libtorch_insightpp
        PUBLIC
        cxx_std_20
)

target_compile_options(test_libtorch_insightpp PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-fopt-info-vec-all>
)

target_link_libraries(test_libtorch_insightpp
        PRIVATE
        insightpp
        OpenMP::OpenMP_CXX
        ${TORCH_LIBRARIES}
)

if(USE_OPENBLAS)
    target_compile_definitions(testinsightpp PRIVATE USE_OPENBLAS)
endif()

#
# Tests
#
add_subdirectory(tests)