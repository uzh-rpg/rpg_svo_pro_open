/*
 * pgo.cpp
 *
 *  Created on: Nov 16, 2018
 *      Author: kunal71091
 */

#include "svo/pgo/pgo.h"

namespace svo
{
void Pgo::addPoseToPgoProblem(const Transformation t, const int frame_id)
{
  if (!problem_lock_)
  {
    ceres::Pose3d pose;
    pose.q = Eigen::Quaterniond(t.getEigenQuaternion());
    pose.p = Eigen::Vector3d(t.getPosition());
    (*poses_)[frame_id] = pose;
  }
  else
  {
    VLOG(40) << "###### Warning: Problem is locked, not adding any poses "
                "#########";
  }
}

void Pgo::addSequentialConstraintToPgoProblem(const Transformation t_be,
                                              const int frame_id_b,
                                              const int frame_id_e)
{
  ceres::Constraint3d constraint;
  constraint.id_begin = frame_id_b;
  constraint.id_end = frame_id_e;
  ceres::Pose3d pose;
  pose.q = Eigen::Quaterniond(t_be.getEigenQuaternion());
  pose.p = Eigen::Vector3d(t_be.getPosition());
  constraint.t_be = pose;
  constraint.information = Eigen::Matrix<double, 6, 6>::Identity();

  constraints_.push_back(constraint);

  /* add constraint to problem */

  ceres::MapOfPoses::iterator pose_begin_iter =
      poses_->find(constraint.id_begin);
  if (pose_begin_iter == poses_->end())
  {
    return;
  }
  ceres::MapOfPoses::iterator pose_end_iter = poses_->find(constraint.id_end);
  if (pose_end_iter == poses_->end())
  {
    return;
  }
  Eigen::Matrix<double, 6, 6> sqrt_information;
  if (searchKfIdInQueue(frame_id_e))
  {
    VLOG(40) << "Frame Id " << frame_id_e << " added with low weight ";
    sqrt_information = constraint.information.llt().matrixL();
    sqrt_information *= 0.0000001;
  }
  else
  {
    sqrt_information = constraint.information.llt().matrixL();
  }
  // Ceres will take ownership of the pointer.
  ceres::CostFunction* cost_function =
      ceres::PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);
  if (!problem_lock_)
  {
    ceres::ResidualBlockId res = problem_.AddResidualBlock(
        cost_function, loss_function_, pose_begin_iter->second.p.data(),
        pose_begin_iter->second.q.coeffs().data(),
        pose_end_iter->second.p.data(),
        pose_end_iter->second.q.coeffs().data());
    seq_constraint_ids_.push_back(res);
    VLOG(40) << "Sequential constraint added for frame " << constraint.id_begin
             << " and " << constraint.id_end;
    problem_.SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization_);
    problem_.SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization_);
  }
  else
  {
    VLOG(20) << "###### Warning: Problem is locked, not adding any sequential "
                "constraints #########";
  }
}

void Pgo::addLoopConstraintToPgoProblem(const Transformation t_be,
                                        const int frame_id_b,
                                        const int frame_id_e)
{
  ceres::Constraint3d constraint;
  constraint.id_begin = frame_id_b;
  constraint.id_end = frame_id_e;
  ceres::Pose3d pose;
  pose.q = Eigen::Quaterniond(t_be.getEigenQuaternion());
  pose.p = Eigen::Vector3d(t_be.getPosition());
  constraint.t_be = pose;
  constraint.information = Eigen::Matrix<double, 6, 6>::Identity();

  constraints_.push_back(constraint);

  /* add constraint to problem */

  ceres::MapOfPoses::iterator pose_begin_iter =
      poses_->find(constraint.id_begin);
  if (pose_begin_iter == poses_->end())
  {
    return;
  }
  ceres::MapOfPoses::iterator pose_end_iter = poses_->find(constraint.id_end);
  if (pose_end_iter == poses_->end())
  {
    return;
  }
  const Eigen::Matrix<double, 6, 6> sqrt_information =
      constraint.information.llt().matrixL();
  // Ceres will take ownership of the pointer.
  ceres::CostFunction* cost_function =
      ceres::PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);
  if (!problem_lock_)
  {
    ceres::ResidualBlockId res_block_id = problem_.AddResidualBlock(
        cost_function, loss_function_, pose_begin_iter->second.p.data(),
        pose_begin_iter->second.q.coeffs().data(),
        pose_end_iter->second.p.data(),
        pose_end_iter->second.q.coeffs().data());

    VLOG(40) << "Loop constraint added for frame " << constraint.id_begin
             << " and " << constraint.id_end;
    problem_.SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization_);
    problem_.SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization_);

    /* Set the query frame constant and solve the problem */
    problem_.SetParameterBlockConstant(pose_begin_iter->second.p.data());
    problem_.SetParameterBlockConstant(
        pose_begin_iter->second.q.coeffs().data());
    VLOG(40) << "Beginning to Re-optimize Pose Graph";
    ceres::Solve(options_, &problem_, &summary_);
    VLOG(10) << summary_.FullReport();
    opt_timing_.push_back(summary_.total_time_in_seconds);
    num_nodes_.push_back(summary_.num_parameter_blocks);
    VLOG(40) << "Re-Optimization Complete";
    problem_.SetParameterBlockVariable(pose_begin_iter->second.p.data());
    problem_.SetParameterBlockVariable(
        pose_begin_iter->second.q.coeffs().data());
    has_updated_result_ = true;
    problem_lock_ = true;
    problem_.RemoveResidualBlock(res_block_id);
  }
  else
  {
    VLOG(20) << "###### Warning: Problem is locked, not adding any loop "
                "constraints #########";
  }
}

void Pgo::purgeProblem()
{
  for (size_t i = 0; i < seq_constraint_ids_.size(); i++)
  {
    problem_.RemoveResidualBlock(seq_constraint_ids_[i]);
  }
  seq_constraint_ids_.clear();
  constraints_.clear();
  problem_lock_ = false;
}

void Pgo::addConstraint(const Transformation& t_be, const int& frame_id_b,
                        const int& frame_id_e,
                        Eigen::Matrix<double, 6, 6>& info)
{
  ceres::Constraint3d constraint;
  constraint.id_begin = frame_id_b;
  constraint.id_end = frame_id_e;
  ceres::Pose3d pose;
  pose.q = Eigen::Quaterniond(t_be.getEigenQuaternion());
  pose.p = Eigen::Vector3d(t_be.getPosition());
  constraint.t_be = pose;
  constraint.information = info;

  constraints_.push_back(constraint);

  /* add constraint to problem */

  ceres::MapOfPoses::iterator pose_begin_iter =
      poses_->find(constraint.id_begin);
  CHECK(pose_begin_iter != poses_->end())
      << "Pose with ID: " << constraint.id_begin << " not found.";
  ceres::MapOfPoses::iterator pose_end_iter = poses_->find(constraint.id_end);
  CHECK(pose_end_iter != poses_->end())
      << "Pose with ID: " << constraint.id_end << " not found.";
  const Eigen::Matrix<double, 6, 6> sqrt_information =
      constraint.information.llt().matrixL();
  // Ceres will take ownership of the pointer.
  ceres::CostFunction* cost_function =
      ceres::PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);
  problem_.AddResidualBlock(
      cost_function, loss_function_, pose_begin_iter->second.p.data(),
      pose_begin_iter->second.q.coeffs().data(), pose_end_iter->second.p.data(),
      pose_end_iter->second.q.coeffs().data());
  VLOG(40) << "Sequential constraint added for frame " << constraint.id_begin
           << " and " << constraint.id_end;
  problem_.SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                               quaternion_local_parameterization_);
  problem_.SetParameterization(pose_end_iter->second.q.coeffs().data(),
                               quaternion_local_parameterization_);
}

void Pgo::solve()
{
  ceres::MapOfPoses::iterator pose_start_iter = poses_->begin();
  CHECK(pose_start_iter != poses_->end()) << "There are no poses.";
  problem_.SetParameterBlockConstant(pose_start_iter->second.p.data());
  problem_.SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
  ceres::Solve(options_, &problem_, &summary_);
  VLOG(10) << summary_.FullReport();
}

bool Pgo::searchKfIdInQueue(int id)
{
  if (ignore_seq_constraint_kfs_.size() == 0)
  {
    return false;
  }
  for (size_t i = 0; i < ignore_seq_constraint_kfs_.size(); i++)
  {
    if (id == ignore_seq_constraint_kfs_[i])
    {
      ignore_seq_constraint_kfs_.pop_front();
      return true;
    }
  }
  return false;
}

bool Pgo::traceTimingData(std::string path)
{
  std::ofstream trace_file;
  trace_file.open(path);
  trace_file.precision(10);
  if (!trace_file)
  {
    return false;
  }
  else
  {
    for (size_t i = 0; i < opt_timing_.size(); i++)
    {
      trace_file << opt_timing_[i] << " " << num_nodes_[i] << std::endl;
    }
  }
  trace_file.close();
  return true;
}
}
