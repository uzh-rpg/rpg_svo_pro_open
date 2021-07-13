/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2016, ETH Zurich, Wyss Zurich, Zurich Eye
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jul 6, 2016
 *      Author: Zurich Eye
 *********************************************************************************/

#pragma once

#include <map>
#include <vector>

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <Eigen/Core>
#pragma diagnostic pop

#include <svo/common/transformation.h>
#include <svo/common/point.h>
#include <svo/vio_common/logging.hpp>
#include <svo/global.h>

namespace svo {

//------------------------------------------------------------------------------
/// \brief Struct to define the behavior of the camera extrinsics.
struct ExtrinsicsEstimationParameters
{
  // set to 0 in order to turn off
  /// \brief Default Constructor -- fixed camera extrinsics.
  ExtrinsicsEstimationParameters()
      : sigma_absolute_translation(0.0),
        sigma_absolute_orientation(0.0),
        sigma_c_relative_translation(0.0),
        sigma_c_relative_orientation(0.0)
  {
  }

  /**
   * @brief Constructor.
   * @param sigma_absolute_translation Absolute translation stdev. [m]
   * @param sigma_absolute_orientation Absolute orientation stdev. [rad]
   * @param sigma_c_relative_translation Relative translation noise density. [m/sqrt(Hz)]
   * @param sigma_c_relative_orientation Relative orientation noise density. [rad/sqrt(Hz)]
   */
  ExtrinsicsEstimationParameters(double sigma_absolute_translation,
                                 double sigma_absolute_orientation,
                                 double sigma_c_relative_translation,
                                 double sigma_c_relative_orientation)
      : sigma_absolute_translation(sigma_absolute_translation),
        sigma_absolute_orientation(sigma_absolute_orientation),
        sigma_c_relative_translation(sigma_c_relative_translation),
        sigma_c_relative_orientation(sigma_c_relative_orientation)
  {
  }

  inline bool isExtrinsicsFixed() const
  {
    return absoluteTranslationVar() < 1.0e-16 &&
        absoluteRotationVar() < 1.0e-16;
  }

  inline double absoluteTranslationVar() const
  {
    return sigma_absolute_translation * sigma_absolute_translation;
  }

  inline double absoluteRotationVar() const
  {
    return sigma_absolute_orientation * sigma_absolute_orientation;
  }

  // absolute (prior) w.r.t frame S
  double sigma_absolute_translation; ///< Absolute translation stdev. [m]
  double sigma_absolute_orientation; ///< Absolute orientation stdev. [rad]

  // relative (temporal)
  double sigma_c_relative_translation;
  ///< Relative translation noise density. [m/sqrt(Hz)]
  double sigma_c_relative_orientation;
  ///< Relative orientation noise density. [rad/sqrt(Hz)]
};

typedef std::vector<ExtrinsicsEstimationParameters,
                    Eigen::aligned_allocator<ExtrinsicsEstimationParameters> >
        ExtrinsicsEstimationParametersVec;

// -----------------------------------------------------------------------------
/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double a_max = 150;  ///< Accelerometer saturation. [m/s^2]
  double g_max = 7.8;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c = 0.001;  ///< Gyroscope noise density. [rad/s*1/sqrt(Hz)]
  double sigma_bg = 0.01;
  ///< Initial gyroscope bias uncertainty. [rad/s*1/sqrt(Hz)]
  double sigma_a_c = 0.02;  ///< Accelerometer noise density. [m/s^2*1/sqrt(Hz)]
  double sigma_ba = 0.1;
  ///< Initial accelerometer bias uncertainty. [m/s^2*1/sqrt(Hz)]
  double sigma_gw_c; ///< Gyroscope drift noise density. [rad/s^2*1/sqrt(Hz)]
  double sigma_aw_c; ///< Accelerometer drift noise density. [m/s^3*1/sqrt(Hz)]
  double g;  ///< Earth acceleration. [m/s^2]
  Eigen::Vector3d a0 = Eigen::Vector3d::Zero();
  ///< Mean of the prior acceleration bias. [m/s^2*1/sqrt(Hz)]
  double rate = 200;  ///< IMU rate. [Hz].
  double delay_imu_cam = 0.0;
  ///< Camera-IMU delay: delay_imu_cam = cam_timestamp - imu_timestamp [s]
};

// -----------------------------------------------------------------------------
// IDs
enum class IdType : uint8_t
{
  NFrame = 0,
  Landmark = 1,
  ImuStates = 2,
  Extrinsics = 3
};

//! The Backend ID for multiple types.
//! Memory layout for types {Frame, IMU state}:
//! Byte 0: IdType
//! Byte 1: zero
//! Byte 2-5: BundleID
//! Byte 6-7: zero
//!
//! For Extrinsics
//! Byte 0: IdType
//! Byte 1: CameraIdx
//! Byte 2-5: BundleID
//! Byte 6-7: zero
//!
//! For Landmarks
//! Byte 0: IdType
//! Byte 1-3: zero
//! Byte 4-7: 32 bit Track ID
class BackendId
{
public:
  BackendId() = default;
  explicit BackendId(uint64_t id) : id_(id) {}

  uint64_t asInteger() const
  {
    return id_;
  }

  IdType type() const
  {
    // The first byte represents the type.
    return static_cast<IdType>(id_ >> 56);
  }

  int32_t bundleId() const
  {
    DEBUG_CHECK(type() != IdType::Landmark)
        << "Landmarks do not have a bundle ID.";
    // The bundle ID is byte 2 -> 6 in id.
    return static_cast<int32_t>((id_ >> 16) & 0xFFFFFFFF);
  }

  uint32_t trackId() const
  {
    DEBUG_CHECK(type() == IdType::Landmark);
    // In case of a landmark, the last 4 bytes are the track ID.
    return static_cast<uint32_t>(id_ & 0xFFFFFFFF);
  }

  uint16_t nFrameHandle() const
  {
    DEBUG_CHECK(type() == IdType::NFrame ||
                type() == IdType::ImuStates ||
                type() == IdType::Extrinsics);
    // In case of an NFrame, the last 2 bytes are the handle.
    return static_cast<uint16_t>(id_ & 0xFFFF);
  }

  uint8_t cameraIndex() const
  {
    DEBUG_CHECK(type() == IdType::Extrinsics);
   // The second byte is the camara index.
    return static_cast<uint8_t>((id_ >> 48) & 0x00000FF);
  }

  bool valid() const
  {
    return id_ != 0;
  }

private:
  uint64_t id_{0};
};

// Factories
inline BackendId createLandmarkId(int track_id)
{
  return BackendId(static_cast<uint64_t>(track_id) |
                   (static_cast<uint64_t>(IdType::Landmark) << 56));
}

inline BackendId createNFrameId(int32_t bundle_id)
{
  CHECK_GE(bundle_id, 0);
  return BackendId((static_cast<uint64_t>(bundle_id) << 16) |
                   (static_cast<uint64_t>(IdType::NFrame) << 56));
}

inline BackendId createExtrinsicsId(uint8_t camera_index,
                                    int32_t bundle_id){
  return BackendId((static_cast<uint64_t>(
                      static_cast<uint32_t>(bundle_id)) << 16) |
                   (static_cast<uint64_t>(camera_index) << 48) |
                   (static_cast<uint64_t>(IdType::Extrinsics) << 56));
}

inline BackendId createImuStateId(int32_t bundle_id)
{
  return BackendId((static_cast<uint64_t>(
                      static_cast<uint32_t>(bundle_id)) << 16) |
                   (static_cast<uint64_t>(IdType::ImuStates) << 56));
}

inline BackendId changeIdType(BackendId id, IdType type, size_t cam_index = 0)
{
  DEBUG_CHECK(id.type() != IdType::Landmark);
  DEBUG_CHECK(type != IdType::Landmark);
  DEBUG_CHECK(cam_index == 0 || type == IdType::Extrinsics);
  // Last 6 bytes remain the same.
  return BackendId((id.asInteger() & 0xFFFFFFFFFFFF) |
                   (static_cast<uint64_t>(cam_index) << 48) |
                   (static_cast<uint64_t>(type) << 56));
}

// Comparison operator for use in maps.
inline bool operator<(const BackendId& lhs, const BackendId& rhs)
{
  return lhs.asInteger() < rhs.asInteger();
}

inline bool operator==(const BackendId& lhs, const BackendId& rhs)
{
  return lhs.asInteger() == rhs.asInteger();
}

inline bool operator!=(const BackendId& lhs, const BackendId& rhs)
{
  return lhs.asInteger() != rhs.asInteger();
}

inline bool operator>=(const BackendId& lhs, const BackendId& rhs)
{
  return lhs.asInteger() >= rhs.asInteger();
}

inline std::ostream& operator<<(std::ostream& out, const BackendId& id)
{
  out << std::hex << id.asInteger() << std::dec;
  return out;
}

//------------------------------------------------------------------------------
/**
 * @brief A type to store information about a point in the world map.
 */
struct MapPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Default constructor. Point is nullptr.
  MapPoint()
      : point(nullptr), fixed_position(false)
  {}
  /**
   * @brief Constructor.
   * @param point     Pointer to underlying svo::Point
   */
  MapPoint(const PointPtr& point)
    : point(point), fixed_position(false)
  {
    hom_coordinates << point->pos(), 1;
  }

  Eigen::Vector4d hom_coordinates; ///< Continuosly updates position of point


  //! Pointer to the point. The position is not updated inside backend
  //! because of possible multithreading conflicts.
  PointPtr point;

  //! Observations of this point. The uint64_t's are the casted
  //! ceres::ResidualBlockId values of the reprojection error residual block.
  std::map<KeypointIdentifier, uint64_t> observations;

  //! Is the point position fixed by a loop closure?
  bool fixed_position;
};

typedef std::vector<MapPoint, Eigen::aligned_allocator<MapPoint> > MapPointVector;
typedef std::map<BackendId, MapPoint, std::less<BackendId>,
  Eigen::aligned_allocator<std::pair<const BackendId, MapPoint>> > PointMap;

//------------------------------------------------------------------------------
// [velocity, gyro biases, accel biases]
typedef Eigen::Matrix<double, 9, 1> SpeedAndBias;

} // namespace svo
