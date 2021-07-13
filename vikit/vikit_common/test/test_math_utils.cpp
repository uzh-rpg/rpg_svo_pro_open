#include <gtest/gtest.h>
#include <vikit/math_utils.h>
#include <aslam/common/entrypoint.h>

namespace {

TEST(MathUtils, testProject2_double)
{
  Eigen::Vector3d x3(2.0, 2.0, 2.0);
  Eigen::Vector2d x2 = vk::project2(x3);
  EXPECT_EQ(x2(0), 1.0);
  EXPECT_EQ(x2(1), 1.0);
}

TEST(MathUtils, testProject2_float)
{
  Eigen::Vector3f x3(2.0, 2.0, 2.0);
  Eigen::Vector2f x2 = vk::project2(x3);
  EXPECT_EQ(x2(0), 1.0);
  EXPECT_EQ(x2(1), 1.0);
}

} // namespace

VIKIT_UNITTEST_ENTRYPOINT
