#include "svo/common/container_helpers.h"

#include <algorithm>

#include <aslam/common/entrypoint.h>
#include <gtest/gtest.h>

namespace svo {
namespace common {
namespace container_helpers {

class ContainerHelpersTest : public ::testing::Test {
 protected:
  const std::vector<int> original_ = std::vector<int>({0, 1, 2, 3, 4});

  bool checkEraseIndicesFromVector(const std::vector<size_t>& to_remove,
             const std::vector<int>& expected) {
    std::vector<int> copy(original_);
    eraseIndicesFromVector(to_remove, &copy);
    return std::equal(copy.begin(), copy.end(), expected.begin());
  }
};

TEST_F(ContainerHelpersTest, EraseIndicesFromVector) {
  EXPECT_TRUE(checkEraseIndicesFromVector({0u}, {1, 2, 3, 4}));
  EXPECT_TRUE(checkEraseIndicesFromVector({2u}, {0, 1, 3, 4}));
  EXPECT_TRUE(checkEraseIndicesFromVector({4u}, {0, 1, 2, 3}));
  EXPECT_TRUE(checkEraseIndicesFromVector({0u, 1u}, {2, 3, 4}));
  EXPECT_TRUE(checkEraseIndicesFromVector({1u, 2u}, {0, 3, 4}));
  EXPECT_TRUE(checkEraseIndicesFromVector({3u, 4u}, {0, 1, 2}));
  EXPECT_TRUE(checkEraseIndicesFromVector({0u, 1u, 2u, 3u, 4u}, {}));
}

}  // namespace container_helpers
}  // namespace common
}  // namespace svo

VIKIT_UNITTEST_ENTRYPOINT
