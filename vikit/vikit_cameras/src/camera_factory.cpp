#include "vikit/cameras/camera_factory.h"

#include <Eigen/Geometry>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop
#include <vikit/cameras.h>

#include "vikit/cameras/atan_distortion.h"
#include "vikit/cameras/camera_geometry.h"
#include "vikit/cameras/equidistant_distortion.h"
#include "vikit/cameras/no_distortion.h"
#include "vikit/cameras/pinhole_projection.h"
#include "vikit/cameras/radial_tangential_distortion.h"

namespace vk {
namespace cameras {
namespace factory {

CameraGeometryBase::Ptr makePinholeCamera(
    const Eigen::VectorXd& intrinsics, uint32_t width, uint32_t height)
{
  CameraGeometryBase::Ptr cam = std::make_shared<PinholeGeometry>(
        width, height,
        PinholeProjection<NoDistortion>(
          intrinsics, NoDistortion()));
  return cam;
}

CameraGeometryBase::Ptr loadFromYAML(
    const std::string& filename,
    const std::string& cam_name)
{
  YAML::Node data = YAML::LoadFile(filename);
  if(!data)
  {
    std::cerr << "Could not load camera from file: " << filename << std::endl;
    return nullptr;
  }

  // The YAML File may contain multiple cameras, specify which one to load
  data = data[cam_name];
  if(!data)
  {
    std::cerr << "Camera with name '"<< cam_name << "' does not exist in "
              << "file: " << filename << std::endl;
    return nullptr;
  }

  // load imu camera transformation
  Transformation T_body_cam; // identity by default
  if(data["T_body_cam"].IsDefined())
  {
    T_body_cam = Transformation(
        Eigen::Quaterniond(
            data["T_body_cam"]["qw"].as<double>(),
            data["T_body_cam"]["qx"].as<double>(),
            data["T_body_cam"]["qy"].as<double>(),
            data["T_body_cam"]["qz"].as<double>()),
        Eigen::Vector3d(
            data["T_body_cam"]["tx"].as<double>(),
            data["T_body_cam"]["ty"].as<double>(),
            data["T_body_cam"]["tz"].as<double>()));
  }
  else if(data["T_cam_body"].IsDefined())
  {
    T_body_cam = Transformation(
        Eigen::Quaterniond(
            data["T_cam_body"]["qw"].as<double>(),
            data["T_cam_body"]["qx"].as<double>(),
            data["T_cam_body"]["qy"].as<double>(),
            data["T_cam_body"]["qz"].as<double>()),
        Eigen::Vector3d(
            data["T_cam_body"]["tx"].as<double>(),
            data["T_cam_body"]["ty"].as<double>(),
            data["T_cam_body"]["tz"].as<double>())).inverse();
  }

  // load camera
  CameraGeometryBase::Ptr cam;
  std::string cam_model = data["cam_model"].as<std::string>();
  if(cam_model == "PinholeRadialTangential")
  {
    double d0 = data["cam_d0"].IsDefined() ? data["cam_d0"].as<double>() : 0.0;
    double d1 = data["cam_d1"].IsDefined() ? data["cam_d1"].as<double>() : 0.0;
    double d2 = data["cam_d2"].IsDefined() ? data["cam_d2"].as<double>() : 0.0;
    double d3 = data["cam_d3"].IsDefined() ? data["cam_d3"].as<double>() : 0.0;
    if(d0 == 0 && d1 == 0 && d2 == 0 && d3 == 0)
    {
      cam_model = "PinholeNoDistortion";
    }
    else
    {
      cam.reset(new PinholeRadTanGeometry(
                  data["cam_width"].as<int>(),
                  data["cam_height"].as<int>(),
                  PinholeProjection<RadialTangentialDistortion>(
                    data["cam_fx"].as<double>(),
                    data["cam_fy"].as<double>(),
                    data["cam_cx"].as<double>(),
                    data["cam_cy"].as<double>(),
                    RadialTangentialDistortion(d0, d1, d2, d3))));
    }
  }
  if(cam_model == "PinholeNoDistortion")
  {
    cam.reset(new PinholeGeometry(
                data["cam_width"].as<int>(),
                data["cam_height"].as<int>(),
                PinholeProjection<NoDistortion>(
                  data["cam_fx"].as<double>(),
                  data["cam_fy"].as<double>(),
                  data["cam_cx"].as<double>(),
                  data["cam_cy"].as<double>(),
                  NoDistortion())));
  }
  else if(cam_model == "PinholeEquidistant")
  {
    cam.reset(new PinholeEquidistantGeometry(
                data["cam_width"].as<int>(),
                data["cam_height"].as<int>(),
                PinholeProjection<EquidistantDistortion>(
                  data["cam_fx"].as<double>(),
                  data["cam_fy"].as<double>(),
                  data["cam_cx"].as<double>(),
                  data["cam_cy"].as<double>(),
                  EquidistantDistortion(
                    data["cam_d0"].as<double>(),
                    data["cam_d1"].as<double>(),
                    data["cam_d2"].as<double>(),
                    data["cam_d3"].as<double>()))));
  }
  else if(cam_model == "PinholeAtan")
  {
    cam.reset(new PinholeAtanGeometry(
                data["cam_width"].as<int>(),
                data["cam_height"].as<int>(),
                PinholeProjection<AtanDistortion>(
                  data["cam_fx"].as<double>(),
                  data["cam_fy"].as<double>(),
                  data["cam_cx"].as<double>(),
                  data["cam_cy"].as<double>(),
                  AtanDistortion(
                    data["cam_d0"].as<double>()))));
  }
  else if(cam_model == "Pinhole")
  {
    std::cerr << "ERROR: Camera Factory: Select between 'PinholeNoDistortion' "
              << "and 'PinholeRadialTangential'." << std::endl;
  }
  else if(cam_model == "OCam")
  {
    std::cerr << "ERROR: Camera Factory: OCam model not yet implemented."
              << std::endl;
  }
  else if(cam_model != "PinholeRadialTangential")
  {
    std::cerr << "ERROR: Camera Factory: Camera model '"<< cam_model
              << "' doesn't exist." << std::endl;
  }

  cam->setLabel(cam_name);

  return cam;
}

} // namespace factory
} // namespace cameras
} // namespace vk
