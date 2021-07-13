#include <svo/common/types.h>
#include <svo/common/transformation.h>

namespace svo
{
class OutlierRejection
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<OutlierRejection> Ptr;

  OutlierRejection(const double reproj_err_threshold):
    reproj_err_threshold_(reproj_err_threshold){}
  ~OutlierRejection(){}

  /**
   * @brief Remove outliers based on reprojection error for a frame with given
   *        pixel 2D positions and landmark 3D positions
   * @param frame The frame to be treated, outliers will be removed
   * @param[out] n_deleted_edges number of deleted edgelet features
   * @param[out] n_deleted_corners number of deleted corner features
   * @param[out] deleted_points list of deleted points
   */
  void removeOutliers(Frame &frame,
                      size_t &n_deleted_edges,
                      size_t &n_deleted_corners,
                      std::vector<int> &deleted_points,
                      const bool ignore_fixed_lm=false) const;

  void setPixelThreshold(const double reproj_err_threshold)
  {
    reproj_err_threshold_ = reproj_err_threshold;
  }

private:
  double reproj_err_threshold_; ///< The threshold in pixels

  /**
   * @brief Calculate the reprojection error of a landmark on the unit plane
   *        along the normal direction of an edgelet.
   * @param[in] f bearing vector of the observation
   * @param[in] xyz_in_world Position of the observed landmark
   * @param[in] grad Gradient direction of the edgelet
   * @param[in] T_cam_world current camera position
   * @param[out] unwhitened_error residual of the observation
   */
  void calculateEdgeletResidualUnitPlane(
      const Eigen::Ref<const BearingVector>& f,
      const Position& xyz_in_world,
      const Eigen::Ref<const GradientVector>& grad,
      const Transformation& T_cam_world,
      double& unwhitened_error) const;

  /**
   * @brief Calculate the reprojection error of a landmark on the unit plane
   *        for corner features
   * @param[in] f bearing vector of the observation
   * @param[in] xyz_in_world Position of the observed landmark
   * @param[in] T_cam_world current camera position
   * @param[out] unwhitened_error residual of the observation
   */
  void calculateFeatureResidualUnitPlane(
      const Eigen::Ref<const BearingVector>& f,
      const Position& xyz_in_world,
      const Transformation& T_cam_world,
      double& unwhitened_error) const;
};
} //namespace svo
