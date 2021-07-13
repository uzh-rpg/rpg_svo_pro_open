#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <boost/optional.hpp>

namespace gtsam
{
template <class POSE, class POINT>
class CameraBearingExtrinsicsFactor
    : public NoiseModelFactor3<POSE, POSE, POINT>
{
protected:
  POINT measured_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef NoiseModelFactor3<POSE, POSE, POINT> Base;
  typedef CameraBearingExtrinsicsFactor<POSE, POINT> This;
  typedef boost::shared_ptr<This> shared_ptr;

  CameraBearingExtrinsicsFactor() = delete;

  CameraBearingExtrinsicsFactor(const POINT& measured,
                                const SharedNoiseModel& noise_model,
                                Key pose_key, Key extrinsic_key, Key point_key)
    : Base(noise_model, pose_key, extrinsic_key, point_key)
    , measured_(measured.normalized())
  {
  }

  virtual ~CameraBearingExtrinsicsFactor()
  {
  }

  virtual gtsam::NonlinearFactor::shared_ptr clone() const
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  virtual bool equals(const NonlinearFactor& p, double tol = 1e-9) const
  {
    const This* e = dynamic_cast<const This*>(&p);
    return e && Base::equals(p, tol) &&
           traits<POINT>::Equals(this->measured_, e->measured_, tol);
  }

  Vector evaluateError(const POSE& T_w_b, const POSE& T_b_c, const POINT& p_w,
                       boost::optional<Matrix&> H1 = boost::none,
                       boost::optional<Matrix&> H2 = boost::none,
                       boost::optional<Matrix&> H3 = boost::none) const
  {
    throw std::runtime_error("CameraBearingFactor: only specializaed evaluate "
                             "error should be used.");
    return Vector3(0.0);
  }
};
template <class POSE, class POINT>
struct traits<CameraBearingExtrinsicsFactor<POSE, POINT>>
    : public Testable<CameraBearingExtrinsicsFactor<POSE, POINT>>
{
};

template <>
inline Vector CameraBearingExtrinsicsFactor<Pose3, Point3>::evaluateError(
    const Pose3& T_w_b, const Pose3& T_b_c, const Point3& p_w,
    boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
    boost::optional<Matrix&> H3) const
{
  if (H1)
  {
    gtsam::Matrix dTwc_dTwb;
    gtsam::Matrix dTwc_dTbc;
    Pose3 T_w_c = T_w_b.compose(T_b_c, dTwc_dTwb, dTwc_dTbc);
    gtsam::Matrix dpc_dpw;
    gtsam::Matrix dpc_dTwc;
    Point3 p_c = T_w_c.transform_to(p_w, dpc_dTwc, dpc_dpw);

    gtsam::Matrix df_dpc;
    Point3 f = p_c.normalized(df_dpc);
    Point3 e_f = f - measured_;

    gtsam::Matrix df_dTwc = df_dpc * dpc_dTwc;
    *H1 = df_dTwc * dTwc_dTwb;
    *H2 = df_dTwc * dTwc_dTbc;
    *H3 = df_dpc * dpc_dpw;

    return Vector3(e_f);
  }
  else
  {
    Pose3 T_w_cam = T_w_b.compose(T_b_c);
    Point3 p_c = T_w_cam.transform_to(p_w);
    Point3 f = p_c.normalized();
    return Vector3(f - measured_);
  }
}
}
