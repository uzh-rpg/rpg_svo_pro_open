#pragma once

#include <map>

#include <Eigen/Core>

#include <svo/common/transformation.h>
#include <rpg_common/aligned.h>

namespace svo
{
// for container indices
using IdCorrespondence = std::pair<size_t, size_t>;
using CorrespondIds = std::vector<IdCorrespondence>;

// point ids
using CorrespondPointIds = std::map<int, int>;

// visualization
using LoopVizInfo = Eigen::Matrix<float, 1, 6>;
using LoopVizInfoVec = rpg::Aligned<std::vector, LoopVizInfo>;

// pose map
using BundleIdToTwb = std::map<int, Transformation>;

struct ClosedLoop
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int lc_id_ = 0;
  int cf_id_ = 0;
  double lc_t_sec_ = 0;
  double cf_t_sec_ = 0;
  Transformation T_lc_cf_corrected_;
  ClosedLoop(const int lc_id, const int cf_id,
             const double lc_t_sec, const double cf_t_sec,
             const Transformation& T_lc_cf_corrected)
    : lc_id_(lc_id), cf_id_(cf_id),
      lc_t_sec_(lc_t_sec), cf_t_sec_(cf_t_sec),
      T_lc_cf_corrected_(T_lc_cf_corrected) {}
};

struct LoopCorrectionInfo
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int lc_kf_bundle_id_ = -1;
  int cur_kf_bundle_id_ = -1;
  Transformation w_T_new_old_;
  LoopCorrectionInfo(const int lc_bundle_id,
                     const int cur_bundle_id,
                     const Transformation& w_T_correction)
    :lc_kf_bundle_id_(lc_bundle_id), cur_kf_bundle_id_(cur_bundle_id),
      w_T_new_old_(w_T_correction)
  { }
};
}
