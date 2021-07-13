#include <gtest/gtest.h>

#include <functional>
#include <type_traits>

#include <random>

#include <eigen-checks/gtest.h>
#include <svo/common/feature_wrapper.h>

template<class T>
typename std::enable_if<std::is_integral<T>::value, std::function<T()> >::type
getRandomGenerator()
{
  std::default_random_engine generator;
  std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::lowest(),
                                                std::numeric_limits<T>::max());
  auto random_val = std::bind(distribution, generator);
  return random_val;
}

template<class T>
typename std::enable_if<!std::is_integral<T>::value, std::function<T()> >::type
getRandomGenerator()
{
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(std::numeric_limits<T>::lowest(),
                                                 std::numeric_limits<T>::max());
  auto random_val = std::bind(distribution, generator);
  return random_val;
}

/* TODO(mwe) Adapt to changes in FeatureWrapper.
TEST(SvoCommonTestSuite,TestFeatureWrapper)
{
  auto random_val_real = getRandomGenerator<svo::FloatType>();
  auto random_val_integral = getRandomGenerator<svo::Level>();

  svo::FeatureType ft_type = (static_cast<int>(random_val_integral()) % 2 ==0)
      ? svo::FeatureType::kCorner
      : svo::FeatureType::kEdgelet;
  svo::Keypoint px(random_val_real(), random_val_real());
  svo::BearingVector f(random_val_real(), random_val_real(), random_val_real());
  svo::GradientVector dx(random_val_real(), random_val_real());
  svo::Level level = random_val_integral();

  svo::FeatureWrapper ft(ft_type, px, f, dx, level);

  EXPECT_EQ(ft_type, ft.type);
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(px, ft.px));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(f, ft.f));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(dx, ft.grad));
  EXPECT_EQ(level, ft.level);

  EXPECT_FLOAT_EQ(px[0], ft.px[0]);
  EXPECT_FLOAT_EQ(px[1], ft.px[1]);
  EXPECT_FLOAT_EQ(px.x(), ft.px.x());
  EXPECT_FLOAT_EQ(px.y(), ft.px.y());
//  EXPECT_FLOAT_EQ(px.x, ft.px.x);
//  EXPECT_FLOAT_EQ(px.y, ft.px.y);
}
*/
