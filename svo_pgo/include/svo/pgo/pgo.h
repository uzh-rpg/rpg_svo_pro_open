/*
 * pgo.h
 * Ceres Based pose Graph optimisation Class
 *
 *  Created on: Nov 16, 2018
 *      Author: kunal71091
 */
#pragma once

#include <thread>
#include <mutex>
#include <memory>
#include <deque>
#include <fstream>

#include <ceres/ceres.h>
#include <svo/common/types.h>
#include <svo/common/transformation.h>

#include "svo/pgo/ceres/types.h"
#include "svo/pgo/ceres/pose_graph_3d_error_term.h"

namespace svo
{
class Pgo
{
public:
  Pgo()
  {
    options_.max_num_iterations = 200;
    options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    has_updated_result_ = false;
    poses_ = new ceres::MapOfPoses;
  };
  ~Pgo(){};

  void addPoseToPgoProblem(const Transformation t, const int frame_id);
  void addSequentialConstraintToPgoProblem(const Transformation t_be,
                                           const int frame_id_b,
                                           const int frame_id_e);
  void addLoopConstraintToPgoProblem(const Transformation t_be,
                                     const int frame_id_b,
                                     const int frame_id_e);
  void updateKeyframeDatabase();
  void purgeProblem();
  /* The functions below are only used for testing */
  void addConstraint(const Transformation& t_be, const int& frame_id_b,
                     const int& frame_id_e, Eigen::Matrix<double, 6, 6>& info);
  void solve();
  bool searchKfIdInQueue(int id);
  bool traceTimingData(std::string path);
  ceres::MapOfPoses* poses_;
  bool has_updated_result_;
  std::vector<ceres::ResidualBlockId> seq_constraint_ids_;
  std::deque<int> ignore_seq_constraint_kfs_;

  /* Vectors to record timings */
  std::vector<double> opt_timing_;
  std::vector<int> num_nodes_;

private:
  bool problem_lock_ = false;
  ceres::VectorOfConstraints constraints_;
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
  ceres::LossFunction* loss_function_ = NULL;
  ceres::LocalParameterization* quaternion_local_parameterization_ =
      new ceres::EigenQuaternionParameterization;
};
}
