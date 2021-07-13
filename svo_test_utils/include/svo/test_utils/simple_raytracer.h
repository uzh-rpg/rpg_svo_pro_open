#pragma once

#include <memory>

#include <Eigen/Geometry>
#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <svo/common/camera_fwd.h>
#include <vikit/cameras.h>

namespace svo {
namespace raytracer {

struct Ray {
  Ray(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir)
    : origin_(origin), dir_(dir) {}
  Eigen::Vector3d origin_;
  Eigen::Vector3d dir_;
};

struct IntersectionResult
{
  /// Possible intersection result.
  enum class Status
  {
    /// One intersection between ray and object
    kUniqueIntersection,
    /// Intersection is a line
    kLineIntersection,
    /// No intersection
    kNoIntersection,
  };

  IntersectionResult(Status status) : status_(status) {}
  Status status_;
};

class Object
{
public:
  typedef std::shared_ptr<Object> Ptr;

  struct Property
  {
    cv::Vec3f color_;
  };
  Property property_;

  Object()
  {
    property_.color_ = cv::Vec3f((float)rand()/(float)RAND_MAX,
                                 (float)rand()/(float)RAND_MAX,
                                 (float)rand()/(float)RAND_MAX);
  }

  virtual IntersectionResult intersect(const Ray& ray_o,
                                       Eigen::Vector3d *intersection) = 0;
};

class Plane : public Object
{
public:
  Plane(double width, double height)
    : Object()
    , width_(width)
    , height_(height) {}

  virtual ~Plane() = default;

  virtual IntersectionResult intersect(const Ray& ray_obj, Eigen::Vector3d *intersection);

private:
  double width_;
  double height_;
};

class CameraWithPose
{
public:
  CameraWithPose() {}
  CameraWithPose(const CameraPtr& cam)
    : cam_(cam)
    , T_cam_body_(CameraWithPose::DownLooking()) {}

  CameraWithPose(const CameraPtr& cam, const Transformation& T_cam_body)
    : cam_(cam)
    , T_cam_body_(T_cam_body) {}

  Transformation T_cam_body() const { return T_cam_body_; }
  Transformation T_body_cam() const { return T_cam_body_.inverse(); }

  CameraPtr cam() const { return cam_; }

  static Transformation DownLooking()
  {
    Transformation::TransformationMatrix M;
    M << 1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 1.0; // <- down looking camera (negative z-axis)
    return Transformation(M);
  }

  static Transformation FrontLooking()
  {
    Transformation::TransformationMatrix M;
    M << 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0; // <- front looking camera (positive y-axis)
    return Transformation(M);
  }

  template <typename CameraGeometry>
  static CameraPtr createCamera(double fov_deg,
                                uint32_t image_width,
                                uint32_t image_height);

  template <typename CameraGeometry>
  static CameraWithPose createCameraWithPose(double fov_deg,
                                     uint32_t image_width,
                                     uint32_t image_height,
                                     const std::string& camera_orientation)
  {
    CameraPtr cam = createCamera<CameraGeometry>(
        fov_deg, image_width, image_height);

    if(camera_orientation == "downward")
      return CameraWithPose(cam, CameraWithPose::DownLooking());
    else
      return CameraWithPose(cam, CameraWithPose::FrontLooking());
  }

private:
  CameraPtr cam_;
  Transformation T_cam_body_;
};

class ObjectWithPose
{
public:
  ObjectWithPose(const Object::Ptr& object)
    : object_(object)
    , T_w_obj_(Transformation()) {}

  ObjectWithPose(const Object::Ptr& object, const Transformation& T_w_obj)
    : object_(object)
    , T_w_obj_(T_w_obj) {}

  Object::Ptr object() const { return object_; }
  Transformation T_obj_w() const { return T_w_obj_.inverse(); }
  Transformation T_w_obj() const { return T_w_obj_; }

  void translate(const Eigen::Vector3d& t)
  {
    Transformation::TransformationMatrix translation_matrix;
    translation_matrix << 1.0, 0.0, 0.0, t[0],
        0.0, 1.0, 0.0, t[1],
        0.0, 0.0, 1.0, t[2],
        0.0, 0.0, 0.0, 1.0;
    T_w_obj_ = Transformation((Transformation::TransformationMatrix)
                              (T_w_obj_.getTransformationMatrix() *
                               translation_matrix));
  }

  void rotateX(double theta_deg)
  {
    T_w_obj_ = Transformation((Transformation::TransformationMatrix)
                              (T_w_obj_.getTransformationMatrix() *
                               rot4x4fromAngleAxis(theta_deg,
                                                   Eigen::Vector3d::UnitX())));
  }

  void rotateY(double theta_deg)
  {
    T_w_obj_ = Transformation((Transformation::TransformationMatrix)
                              (T_w_obj_.getTransformationMatrix() *
                               rot4x4fromAngleAxis(theta_deg,
                                                   Eigen::Vector3d::UnitY())));
  }

  void rotateZ(double theta_deg)
  {
    T_w_obj_ = Transformation((Transformation::TransformationMatrix)
                              (T_w_obj_.getTransformationMatrix() *
                               rot4x4fromAngleAxis(theta_deg,
                                                   Eigen::Vector3d::UnitZ())));
  }

private:

  static Transformation::TransformationMatrix rot4x4fromAngleAxis(
      double theta_deg,
      const Eigen::Vector3d& axis)
  {
    Transformation::TransformationMatrix rot4x4;
    rot4x4.setIdentity(4,4);
    rot4x4.block<3,3>(0,0) =
        (Eigen::Matrix3d) Eigen::AngleAxisd(theta_deg * M_PI / 180.0, axis);
    return rot4x4;
  }

  Object::Ptr object_;
  Transformation T_w_obj_;
};

class Scene
{
public:
  Scene() {}
  Scene(std::vector<ObjectWithPose>& objects)
    : objects_(objects) {}

  size_t num_objects() const
  {
    return objects_.size();
  }

  Object::Ptr object(size_t idx) const
  {
    return objects_[idx].object();
  }

  Transformation T_obj_w(size_t idx) const
  {
    return objects_[idx].T_obj_w();
  }

  Transformation T_w_obj(size_t idx) const
  {
    return objects_[idx].T_w_obj();
  }

  void addObjects(const ObjectWithPose& new_object)
  {
    objects_.push_back(new_object);
  }

  void addObjects(const std::vector<ObjectWithPose>& new_objects)
  {
    objects_.insert(objects_.end(), new_objects.begin(), new_objects.end());
  }

  static std::vector<ObjectWithPose> createBox(double sizeX,
                                               double sizeY,
                                               double sizeZ);

  static std::vector<ObjectWithPose> createCanyon(double width,
                                                  double length,
                                                  double height);

private:
  std::vector<ObjectWithPose> objects_;
};

class SimpleRayTracer
{
public:
  SimpleRayTracer(const Scene& scene)
    : scene_(scene) {}

  IntersectionResult intersectPixel(const CameraWithPose &cam,
                                    const Transformation &T_world_cam,
                                    const Eigen::Vector2d& px,
                                    Eigen::Vector3d *intersection_cam,
                                    Object::Ptr& intersected_object) const;

  cv::Mat render(const CameraWithPose &cam,
                 const Transformation &T_world_cam,
                 bool depth_shading=false) const;

  static constexpr double kInfinity = 1e12;

private:
  Scene scene_;
};

}
}
