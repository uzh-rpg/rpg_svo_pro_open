#pragma once

#include <cmath>

#include <kindr/minimal/quat-transformation.h>

#include "rpg_common/aligned.h"

namespace rpg_common {

typedef kindr::minimal::QuatTransformation Pose;
typedef kindr::minimal::RotationQuaternion Rotation;
typedef Eigen::Vector3d Position;

using PoseVec = Aligned<std::vector, Pose>;
using PositionVec = Aligned<std::vector, Position>;
using RotationVec = Aligned<std::vector, Rotation>;

namespace pose {

inline Pose xRotationDeg(const double angle_deg)
{
  const double c_d2 = cos(angle_deg * M_PI / 360.);
  const double s_d2 = sin(angle_deg * M_PI / 360.);
  const double w = c_d2;
  return Pose(Pose::Rotation(w, s_d2, 0., 0.), Eigen::Vector3d::Zero());
}
inline Pose yRotationDeg(const double angle_deg)
{
  const double c_d2 = cos(angle_deg * M_PI / 360.);
  const double s_d2 = sin(angle_deg * M_PI / 360.);
  const double w = c_d2;
  return Pose(Pose::Rotation(w, 0., s_d2, 0.), Eigen::Vector3d::Zero());
}
inline Pose zRotationDeg(const double angle_deg)
{
  const double c_d2 = cos(angle_deg * M_PI / 360.);
  const double s_d2 = sin(angle_deg * M_PI / 360.);
  const double w = c_d2;
  return Pose(Pose::Rotation(w, 0., 0., s_d2), Eigen::Vector3d::Zero());
}

inline Pose yawPitchRollDeg(
    const double yaw, const double pitch, const double roll)
{
  return zRotationDeg(yaw) * yRotationDeg(pitch) * xRotationDeg(roll);
}

}  // namespace pose

}  // namespace rpg_common
namespace rpg = rpg_common;
