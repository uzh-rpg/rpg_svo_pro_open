/*
 *  This file can be used to test the ceres based pose graph optimisation given
 * a g20 file of noisy vertices and constraints.
 *  This code is partly adapted from the examples in ceres github repository.
 *
 *  Created on: Nov 22, 2018
 *      Author: kunal71091
 */

#include "svo/pgo/pgo.h"

#include <fstream>
#include <string>

using namespace svo;

bool OutputPoses(const std::string& filename, const ceres::MapOfPoses& poses)
{
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile)
  {
    LOG(ERROR) << "Error opening the file: " << filename;
    return false;
  }
  for (std::map<int, ceres::Pose3d, std::less<int>,
                Eigen::aligned_allocator<
                    std::pair<const int, ceres::Pose3d> > >::const_iterator
           poses_iter = poses.begin();
       poses_iter != poses.end(); ++poses_iter)
  {
    const std::map<int, ceres::Pose3d, std::less<int>,
                   Eigen::aligned_allocator<
                       std::pair<const int, ceres::Pose3d> > >::value_type&
        pair = *poses_iter;
    outfile << pair.first << " " << pair.second.p.transpose() << " "
            << pair.second.q.x() << " " << pair.second.q.y() << " "
            << pair.second.q.z() << " " << pair.second.q.w() << '\n';
  }
  return true;
}

int main()
{
  Pgo pgo_;
  std::string path = "/media/kunal71091/common_storage/Huawei_Project/"
                     "sphere_bignoise_vertex3.g2o";

  std::ifstream infile(path.c_str());

  std::string data_type;
  while (infile.good())
  {
    // Read whether the type is a node or a constraint.
    infile >> data_type;
    if (data_type == ceres::Pose3d::name())
    {
      int id;
      infile >> id;
      kindr::minimal::Position p;
      Eigen::Quaternion<double> q_eigen;
      Transformation t;
      infile >> p.x() >> p.y() >> p.z() >> q_eigen.x() >> q_eigen.y() >>
          q_eigen.z() >> q_eigen.w();
      Quaternion q = Quaternion(q_eigen);
      q.normalize();
      t = Transformation(q, p);
      pgo_.addPoseToPgoProblem(t, id);
    }
    else if (data_type == ceres::Constraint3d::name())
    {
      int id_b, id_e;
      kindr::minimal::Position p;
      Eigen::Quaternion<double> q_eigen;
      Transformation t_be;
      Eigen::Matrix<double, 6, 6> info;
      infile >> id_b >> id_e;
      infile >> p.x() >> p.y() >> p.z() >> q_eigen.x() >> q_eigen.y() >>
          q_eigen.z() >> q_eigen.w();
      Quaternion q = Quaternion(q_eigen);
      q.normalize();
      t_be = Transformation(q, p);
      for (int i = 0; i < 6 && infile.good(); ++i)
      {
        for (int j = i; j < 6 && infile.good(); ++j)
        {
          infile >> info(i, j);
          if (i != j)
          {
            info(j, i) = info(i, j);
          }
        }
      }
      pgo_.addConstraint(t_be, id_b, id_e, info);
    }
  }

  std::cout << "Beginning to Solve" << std::endl;
  pgo_.solve();

  std::string out = "/media/kunal71091/common_storage/Huawei_Project/"
                    "sphere_bignoise_vertex3_out.txt";
  OutputPoses(out, *pgo_.poses_);

  return 0;
}
