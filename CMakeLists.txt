cmake_minimum_required(VERSION 3.28)
project(insightpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Enable testing
enable_testing()

#
# Library
#

add_library(insightpp
    STATIC
        src/insightpp.cpp
)

target_link_libraries(insightpp PRIVATE OpenMP::OpenMP_CXX)

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
)

target_link_libraries(testinsightpp
    PRIVATE
        insightpp
)

#
# Tests
#
add_subdirectory(tests)