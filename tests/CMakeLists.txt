include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(insightpp_tests
        test_library.cpp
)

target_link_libraries(insightpp_tests
    PRIVATE
        insightpp
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(insightpp_tests)