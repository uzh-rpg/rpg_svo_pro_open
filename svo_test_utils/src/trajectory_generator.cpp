#include <svo/test_utils/trajectory_generator.h>
#include <fstream>

namespace svo {
namespace trajectory {

Trajectory Trajectory::loadfromFile(const std::string& filename)
{
  std::ifstream fs;
  fs.open(filename);
  std::vector<StampedPose> poses;
  double stamp;
  double tx, ty, tz, qx, qy, qz, qw;
  if(fs.is_open())
  {
    while(fs >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
    {
      fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      Eigen::Quaterniond q(qw, qx, qy, qz);
      if(std::abs(q.norm()-1.0) > 0.000001)
        LOG(WARNING) << "trajectory::Trajectory: Quaternion norm = " << q.norm();
      q.normalize();
      poses.push_back(StampedPose(stamp, Transformation(q, Eigen::Vector3d(tx, ty, tz))));
    }
  } else {
    LOG(FATAL) << "Could not open trajectory file " << filename;
  }
  return Trajectory(poses);
}

Trajectory Trajectory::createCircularTrajectory(size_t length, double fps, const Eigen::Vector3d& center, double radius)
{
  std::vector<StampedPose> poses;
  double stamp = 0.0;
  for(size_t i=0;i<length;++i)
  {
    const double theta = 2*(double)i*CV_PI/(double)length;
    Eigen::Vector3d pos = center + radius * Eigen::Vector3d(cos(theta), sin(theta), 0.0);
    Eigen::AngleAxisd rot(theta, Eigen::Vector3d::UnitZ());
    Transformation T(Eigen::Quaterniond(rot), pos);
    poses.push_back(StampedPose(stamp, T));
    stamp += 1.0/fps;
  }
  return Trajectory(poses);
}

Trajectory Trajectory::createStraightTrajectory(size_t length, double fps, const Eigen::Vector3d& start, const Eigen::Vector3d& end)
{
  std::vector<StampedPose> poses;
  double stamp = 0.0;
  for(size_t i=0;i<length;++i)
  {
    Eigen::Vector3d current_pos = start + (double)i/(double)length * (end - start);
    poses.push_back(StampedPose(stamp, Transformation(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), current_pos)));
    stamp += 1.0/fps;
  }
  return Trajectory(poses);
}

}
}
