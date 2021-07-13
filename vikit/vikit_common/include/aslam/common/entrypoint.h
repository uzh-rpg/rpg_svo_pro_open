#pragma once

#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#define VIKIT_UNITTEST_ENTRYPOINT\
  int main(int argc, char** argv) {\
  ::testing::InitGoogleTest(&argc, argv);\
  google::InitGoogleLogging(argv[0]);\
  google::InstallFailureSignalHandler();\
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";\
  FLAGS_alsologtostderr = true; \
  FLAGS_colorlogtostderr = true; \
  return RUN_ALL_TESTS();\
}

// Let the Eclipse parser see the macro.
#ifndef TEST
#define TEST(a, b) int Test_##a##_##b()
#endif

#ifndef TEST_F
#define TEST_F(a, b) int Test_##a##_##b()
#endif

#ifndef TEST_P
#define TEST_P(a, b) int Test_##a##_##b()
#endif

#ifndef TYPED_TEST
#define TYPED_TEST(a, b) int Test_##a##_##b()
#endif

#ifndef TYPED_TEST_P
#define TYPED_TEST_P(a, b) int Test_##a##_##b()
#endif

#ifndef TYPED_TEST_CASE
#define TYPED_TEST_CASE(a, b) int Test_##a##_##b()
#endif

#ifndef REGISTER_TYPED_TEST_CASE_P
#define REGISTER_TYPED_TEST_CASE_P(a, ...)  int Test_##a()
#endif

#ifndef INSTANTIATE_TYPED_TEST_CASE_P
#define INSTANTIATE_TYPED_TEST_CASE_P(a, ...) int Test_##a()
#endif
