# note for ubuntu users: gtest lib is no longer shipped as dep package (see http://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test)
add_subdirectory(lib/googletest)
include_directories(lib/googletest/include lib/googletest)

set(TEST_FILES bitmask_test.cpp bitmask_experiments.cpp)

add_executable(tester ${TEST_FILES})

target_link_libraries(tester gtest gtest_main pthread)
