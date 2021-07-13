#include "svo/test_utils/simple_raytracer.h"

#include <opencv2/highgui/highgui.hpp>
#include <svo/common/camera.h>
#include <svo/common/conversions.h>
#include <vikit/cameras/camera_factory.h>
#include <vikit/cameras/equidistant_fisheye_geometry.h>

#include "svo/test_utils/trajectory_generator.h"

namespace svo {
namespace raytracer {

IntersectionResult Plane::intersect(const Ray& ray_obj, Eigen::Vector3d *intersection)
{

  const double epsilon = 1e-10;

  const Eigen::Vector3d dir = ray_obj.dir_;
  const Eigen::Vector3d origin = ray_obj.origin_;

  // Degenerate cases: ray lies within plane or is parallel to plane without touching it
  if(std::abs(dir[2]) < epsilon)
  {
    if(std::abs(origin[2]))
      return IntersectionResult(IntersectionResult::Status::kLineIntersection);
    else
      return IntersectionResult(IntersectionResult::Status::kNoIntersection);
  }

  // General case
  const double d = -origin[2]/dir[2];
  *intersection = origin + d * dir;
  assert(std::abs((*intersection)[2]) < epsilon);

  if(d >= 0 && std::abs((*intersection)[0]) <= width_/2.0 && std::abs((*intersection)[1]) <= height_/2.0)
  {
    return IntersectionResult(IntersectionResult::Status::kUniqueIntersection);
  }
  else
  {
    return IntersectionResult(IntersectionResult::Status::kNoIntersection);
  }
}

std::vector<ObjectWithPose> Scene::createBox(double sizeX, double sizeY, double sizeZ)
{
  // Box centered on (0, 0, 0)
  ObjectWithPose front(std::make_shared<Plane>(sizeX, sizeZ));
  front.rotateX(-90.0);
  front.translate(Eigen::Vector3d(0.0, 0.0, sizeY/2.0));
  front.object()->property_.color_ = cv::Vec3f(0.f, 0.f, 1.f);

  ObjectWithPose behind(std::make_shared<Plane>(sizeX, sizeZ));
  behind.rotateX(90.0);
  behind.translate(Eigen::Vector3d(0.0, 0.0, sizeY/2.0));
  behind.object()->property_.color_ = cv::Vec3f(0.f, 1.f, 1.f);

  ObjectWithPose right(std::make_shared<Plane>(sizeZ, sizeY));
  right.rotateY(90.0);
  right.translate(Eigen::Vector3d(0.0, 0.0, sizeX/2.0));
  right.object()->property_.color_ = cv::Vec3f(0.f, 1.f, 0.f);

  ObjectWithPose left(std::make_shared<Plane>(sizeZ, sizeY));
  left.rotateY(-90.0);
  left.translate(Eigen::Vector3d(0.0, 0.0, sizeX/2.0));
  left.object()->property_.color_ = cv::Vec3f(1.f, 0.f, 0.f);

  ObjectWithPose top(std::make_shared<Plane>(sizeX, sizeY));
  top.translate(Eigen::Vector3d(0.0, 0.0, sizeZ/2.0));
  top.object()->property_.color_ = cv::Vec3f(0.f, 0.5f, 1.f);

  ObjectWithPose down(std::make_shared<Plane>(sizeX, sizeY));
  down.translate(Eigen::Vector3d(0.0, 0.0, -sizeZ/2.0));
  down.object()->property_.color_ = cv::Vec3f(1.f, 1.f, 0.f);

  return {top, down, front, behind, left, right};
}

std::vector<ObjectWithPose> Scene::createCanyon(double width, double length, double height)
{
  // Canyon = road (Z=-height/2) + left wall + right wall
  ObjectWithPose down(std::make_shared<Plane>(width, length));
  down.translate(Eigen::Vector3d(0.0, 0.0, -height/2.0));
  down.object()->property_.color_ = cv::Vec3f(0.f, 1.f, 0.f);

  ObjectWithPose left(std::make_shared<Plane>(height, length));
  left.rotateY(-90.0);
  left.translate(Eigen::Vector3d(0.0, 0.0, width/2.0));
  left.object()->property_.color_ = cv::Vec3f(0.f, 0.f, 1.f);

  ObjectWithPose right(std::make_shared<Plane>(height, length));
  right.rotateY(90.0);
  right.translate(Eigen::Vector3d(0.0, 0.0, width/2.0));
  right.object()->property_.color_ = cv::Vec3f(1.f, 0.f, 0.f);

  return {down, left, right};
}

IntersectionResult SimpleRayTracer::intersectPixel(const CameraWithPose &cam,
                                                   const Transformation& T_world_cam,
                                                   const Eigen::Vector2d& px,
                                                   Eigen::Vector3d *intersection_cam,
                                                   Object::Ptr& intersected_object) const
{
  double closest_distance_cam = kInfinity;
  Eigen::Vector3d intersection_obj;

  Eigen::Vector3d dir_cam;
  cam.cam()->backProject3(px, &dir_cam);
  for(size_t i=0;i<scene_.num_objects();++i) {
    const Transformation T_obj_cam = scene_.T_obj_w(i) * T_world_cam;
    const Ray ray_obj(T_obj_cam.getPosition(),
                      (T_obj_cam.getRotation().rotate(dir_cam)).normalized());
    IntersectionResult res = scene_.object(i)->intersect(ray_obj, &intersection_obj);
    if(res.status_ == IntersectionResult::Status::kUniqueIntersection) {
      Eigen::Vector3d current_intersection_cam = T_world_cam.inverse() * scene_.T_w_obj(i) * intersection_obj;
      double distance_cam = current_intersection_cam.norm();
      if(distance_cam < closest_distance_cam) {
        closest_distance_cam = distance_cam;
        *intersection_cam = current_intersection_cam;
        intersected_object = scene_.object(i);
      }
    }
  }
  if(closest_distance_cam == kInfinity)
    return IntersectionResult::Status::kNoIntersection;
  else
    return IntersectionResult::Status::kUniqueIntersection;
}

cv::Mat SimpleRayTracer::render(const CameraWithPose& cam,
                                const Transformation& T_world_cam,
                                bool depth_shading) const
{
  cv::Mat canvas = cv::Mat::zeros(cam.cam()->imageHeight(), cam.cam()->imageWidth(), CV_32FC3);
  for(int i=0;i<canvas.rows;++i) {
    for(int j=0;j<canvas.cols;++j) {
      const Eigen::Vector2d px((double)j, (double)i);

      if(cam.cam()->hasMask() && cam.cam()->isMasked(px))
        continue;

      Eigen::Vector3d intersection;
      Object::Ptr intersected_object;
      IntersectionResult res = intersectPixel(cam, T_world_cam, px, &intersection, intersected_object);
      if(res.status_ == IntersectionResult::Status::kUniqueIntersection) {
        if(depth_shading)
          canvas.at<cv::Vec3f>(i, j) = 1.f / intersection.norm() * intersected_object->property_.color_;
        else
          canvas.at<cv::Vec3f>(i, j) = intersected_object->property_.color_;
      }
    }
  }

  if(depth_shading)
    cv::normalize(canvas, canvas, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC3);

  return canvas;
}

template <>
CameraPtr CameraWithPose::createCamera<vk::cameras::PinholeGeometry>(
    double fov_deg, uint32_t image_width, uint32_t image_height)
{
  Eigen::VectorXd intrinsics(4);
  const double f =
      (double)image_width / (2.0 * std::tan(
          0.5 * fov_deg * common::conversions::kDegToRad));
  intrinsics << f, f, (double)image_width/2.0, (double)image_height/2.0;
  return vk::cameras::factory::makePinholeCamera(
      intrinsics, image_width, image_height);
}

template <>
CameraPtr CameraWithPose::createCamera<vk::cameras::EquidistantFisheyeGeometry>(
    double fov_deg, uint32_t image_width, uint32_t image_height)
{
  const double focal_length =
      image_width / (fov_deg * common::conversions::kDegToRad);
  const Eigen::Vector2d principal_point(image_width / 2., image_height / 2.);
  const double radius = image_width / 2.;
  return std::make_shared<vk::cameras::EquidistantFisheyeGeometry>(
      image_width, image_height, focal_length, principal_point, radius);
}

}
}
