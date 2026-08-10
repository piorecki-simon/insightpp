cmake_minimum_required(VERSION 3.28)
project(insightpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# BLAS
option(USE_MKL_BLAS "Enable Intel MKL BLAS for GEMMs" OFF)
if(USE_MKL_BLAS)
    set(BLA_VENDOR Intel10_64lp) # or Intel10_64lp_seq for single-threaded MKL
endif()
find_package(BLAS REQUIRED)
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")

find_package(OpenMP REQUIRED)
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

target_link_libraries(insightpp
    PRIVATE
        OpenMP::OpenMP_CXX
        BLAS::BLAS
)

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
target_compile_options(insightpp
    PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
)

#
# Tests
#
add_subdirectory(tests)

#
# Examples
#
add_subdirectory(examples)