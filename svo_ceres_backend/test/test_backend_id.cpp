// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <svo/vio_common/test_utils.hpp>

#include "svo/ceres_backend/estimator_types.hpp"

TEST(BackendIdTests, NFrameIdTest)
{
  int32_t bundle_id =
      svo::test_utils::sampleUniformIntDistribution<int32_t>(true, 0);

  svo::BackendId id = svo::createNFrameId(bundle_id);

  EXPECT_EQ(id.type(), svo::IdType::NFrame);
  EXPECT_EQ(id.bundleId(), bundle_id);
}

TEST(BackendIdTests, LandmarkIdTest)
{
  int track_id = svo::test_utils::sampleUniformIntDistribution<int>(true, 0);

  svo::BackendId id = svo::createLandmarkId(track_id);

  EXPECT_EQ(id.type(), svo::IdType::Landmark);
  EXPECT_EQ(id.trackId(), track_id);
}

TEST(BackendIdTests, ImuStatesIdTest)
{
  int32_t bundle_id =
      svo::test_utils::sampleUniformIntDistribution<int32_t>(true, 0);

  svo::BackendId id = svo::createImuStateId(bundle_id);

  EXPECT_EQ(id.type(), svo::IdType::ImuStates);
  EXPECT_EQ(id.bundleId(), bundle_id);
}

TEST(BackendIdTests, ExtrinsicsIdTest)
{
  int32_t bundle_id =
      svo::test_utils::sampleUniformIntDistribution<int32_t>(true, 0);
  uint8_t camera_index =
      svo::test_utils::sampleUniformIntDistribution<uint8_t>(true);

  svo::BackendId id = svo::createExtrinsicsId(camera_index, bundle_id);

  EXPECT_EQ(id.type(), svo::IdType::Extrinsics);
  EXPECT_EQ(id.bundleId(), bundle_id);
  EXPECT_EQ(id.cameraIndex(), camera_index);
}

TEST(BackendIdTests, ChangeTypeTest)
{
  using namespace svo;
  int32_t bundle_id =
      svo::test_utils::sampleUniformIntDistribution<int32_t>(true, 0);
  uint8_t camera_index =
      svo::test_utils::sampleUniformIntDistribution<uint8_t>(true);

  const BackendId nframe_id = createNFrameId(bundle_id);
  const BackendId speed_and_bias_id = createImuStateId(bundle_id);
  const BackendId extrinsics_id = createExtrinsicsId(camera_index, bundle_id);

  BackendId speed_and_bias_id_gen, extrinsics_id_gen, nframe_id_gen;
  // Change from nframe to others.
  speed_and_bias_id_gen = changeIdType(nframe_id, IdType::ImuStates);
  extrinsics_id_gen = changeIdType(nframe_id, IdType::Extrinsics, camera_index);
  EXPECT_EQ(speed_and_bias_id, speed_and_bias_id_gen);
  EXPECT_EQ(extrinsics_id, extrinsics_id_gen);

  // Change speed and bias ID to others.
  nframe_id_gen = changeIdType(speed_and_bias_id, IdType::NFrame);
  extrinsics_id_gen =
      changeIdType(speed_and_bias_id, IdType::Extrinsics, camera_index);
  EXPECT_EQ(nframe_id, nframe_id_gen);
  EXPECT_EQ(extrinsics_id, extrinsics_id_gen);

  // Change extrinsics ID to others.
  nframe_id_gen = changeIdType(extrinsics_id, IdType::NFrame);
  speed_and_bias_id_gen = changeIdType(extrinsics_id, IdType::ImuStates);
  EXPECT_EQ(nframe_id, nframe_id_gen);
  EXPECT_EQ(speed_and_bias_id, speed_and_bias_id_gen);
}

VIKIT_UNITTEST_ENTRYPOINT
