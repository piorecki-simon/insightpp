add_executable(mnist_sgd_01 mnist/common.h mnist/mnist_sgd_01.cpp)
add_executable(mnist_sgd_01_libtorch mnist/common.h mnist/mnist_sgd_01_libtorch.cpp)

target_compile_features(mnist_sgd_01
    PUBLIC
        cxx_std_20
)

target_compile_features(mnist_sgd_01_libtorch
    PUBLIC
        cxx_std_20
)

target_compile_options(mnist_sgd_01
    PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
)

target_compile_options(mnist_sgd_01_libtorch
    PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
)

target_link_libraries(mnist_sgd_01
    PRIVATE
        insightpp
        OpenMP::OpenMP_CXX
)

target_link_libraries(mnist_sgd_01_libtorch
    PRIVATE
        insightpp
        OpenMP::OpenMP_CXX
)

if(USE_MKL_BLAS)
    target_compile_definitions(mnist_sgd_01 PRIVATE USE_MKL_BLAS)
    target_compile_definitions(mnist_sgd_01_libtorch PRIVATE USE_MKL_BLAS)
endif()