#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <aslam/common/macros.h>
#include <aslam/common/pose-types.h>

namespace vk {
namespace cameras {

class CameraGeometryBase;
using Camera = CameraGeometryBase;
using Transformation = aslam::Transformation;
using TransformationVector = aslam::TransformationVector;
using Quaternion = aslam::Quaternion;

class NCamera
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ASLAM_POINTER_TYPEDEFS(NCamera);

protected:
  NCamera() = default;

public:
  NCamera(
      const TransformationVector& T_C_B,
      const std::vector<std::shared_ptr<Camera>>& cameras,
      const std::string& label);

  ~NCamera() {}

  NCamera(const NCamera&) = delete;
  void operator=(const NCamera&) = delete;

  /// Load a camera rig form a yaml file. Returns a nullptr if the loading fails.
  static std::shared_ptr<NCamera> loadFromYaml(const std::string& yaml_file);

  /// Get the number of cameras.
  inline size_t getNumCameras() const { return cameras_.size(); }

  /// Get the pose of body frame with respect to the camera i.
  const Transformation& get_T_C_B(size_t camera_index) const;

  /// Get all transformations.
  const TransformationVector& getTransformationVector() const;

  /// Get the geometry object for camera i.
  const Camera& getCamera(size_t camera_index) const;

  /// Get the geometry object for camera i.
  std::shared_ptr<Camera> getCameraShared(size_t camera_index);

  /// Get the geometry object for camera i.
  std::shared_ptr<const Camera> getCameraShared(size_t camera_index) const;

  /// How many cameras does this system have?
  inline size_t numCameras() const { return cameras_.size(); }

  /// Print camera infos
  void printParameters(std::ostream& out, const std::string& s = "Camera: ") const;

  /// Get all cameras.
  const std::vector<std::shared_ptr<Camera>>& getCameraVector() const;

  /// Get a label for the camera.
  inline const std::string& getLabel() const {return label_;}

  /// Set a label for the camera.
  inline void setLabel(const std::string& label) {label_ = label;}

  /// keep first N cameras
  inline void keepFirstNCams(const int N)
  {
    CHECK_LT(N, static_cast<int>(cameras_.size()));
    CHECK_GT(N, 0);
    T_C_B_.erase(T_C_B_.begin() + N, T_C_B_.end());
    cameras_.erase(cameras_.begin() + N, cameras_.end());
  }

private:
  /// Internal consistency checks and initialization.
  void initInternal();

  /// The mounting transformations.
  TransformationVector T_C_B_;

  /// The camera geometries.
  std::vector<std::shared_ptr<Camera>> cameras_;

  /// A label for this camera rig, a name.
  std::string label_;
};

} // namespace cameras
} // namespace vikit

