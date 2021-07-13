#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop

// std
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>

// svo
#include <svo/io.h>
#include <svo/global.h>
#include <svo/map.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>

namespace svo {
namespace io {

bool saveMap(
    const MapPtr& map,
    const std::string& save_dir,
    const std::string& map_name)
{
  std::string filename = save_dir+"/"+map_name;
  std::cout << "save map to file " << filename << std::endl;
  YAML::Emitter out;
  out << YAML::BeginMap
      << YAML::Key << "frames" << YAML::Value
      << YAML::BeginSeq;

  // safe frames
  for(const auto& keyval : map->keyframes_)
  {
    const FramePtr& frame = keyval.second;
    const Vector3d& t = frame->T_world_cam().getPosition();
    const Eigen::Quaterniond& q = frame->T_world_cam().getRotation().toImplementation();
    out << YAML::BeginMap
        << YAML::Key << "frame_id" << YAML::Value << frame->id()
        << YAML::Key << "cam_name" << YAML::Value << frame->cam()->getLabel()
        << YAML::Key << "timestamp" << YAML::Value << frame->getTimestampNSec()
        << YAML::Key << "T_world_cam" << YAML::Value << YAML::Flow
        << YAML::BeginSeq
          << t.x() << t.y() << t.z()
          << q.x() << q.y() << q.z() << q.w()
        << YAML::EndSeq
        //<< YAML::Comment("tx ty tz qx qy qz qw")
        << YAML::EndMap;
  }
  out << YAML::EndSeq
      << YAML::Key << "features" << YAML::Value
      << YAML::BeginSeq;

  // safe features
  CHECK(false) << "fix implementation.";
  std::unordered_set<PointPtr> points;
  /*
  for(const auto& keyval : map->keyframes_)
  {
    for(const FeaturePtr& ftr : keyval.second->fts_)
    {
      if(ftr->point->type() == Point::TYPE_CORNER_SEED
         || ftr->point->type() == Point::TYPE_EDGELET_SEED )
      {
        continue;
      }

      out << YAML::BeginMap
          << YAML::Key << "frame_id" << YAML::Value << keyval.second->id()
          << YAML::Key << "point_id" << YAML::Value << ftr->point->id()
          << YAML::Key << "px" << YAML::Value << YAML::Flow
          << YAML::BeginSeq
            << ftr->px[0]
            << ftr->px[1]
          << YAML::EndSeq
          << YAML::Key << "level" << YAML::Value << ftr->level
          << YAML::EndMap;
      points.insert(ftr->point);
    }
  }
  */

  out << YAML::EndSeq
      << YAML::Key << "points" << YAML::Value
      << YAML::BeginSeq;

  // safe points
  for(const PointPtr point : points)
  {
    out << YAML::BeginMap
        << YAML::Key << "point_id" << YAML::Value << point->id()
        << YAML::Key << "pos" << YAML::Value << YAML::Flow
        << YAML::BeginSeq
          << point->pos_[0]
          << point->pos_[1]
          << point->pos_[2]
        << YAML::EndSeq
        << YAML::EndMap;
  }

  out << YAML::EndSeq;

  // write to file
  std::ofstream fout(filename.c_str());
  if(!fout.is_open())
    return false;
  fout << out.c_str();
  fout.close();
  return true;
}

bool loadMap(
    const std::string& load_dir,
    MapPtr& map)
{
  // TODO: assume camera has name cam0 and is in cam_calib.yaml file
  SVO_ERROR_STREAM("FIX LOAD CAMERA IMPLEMENTATION");
  CameraPtr cam;
  /*
  CameraPtr cam = vk::cameras::factory::loadFromYAML(
        load_dir+"/cam_calib.yaml", "cam0");
  */
  if(cam == nullptr)
    return false;
  cam->printParameters(std::cout, "Loaded camera:");

  map = std::make_shared<Map>();

  const std::string map_filename(load_dir+"/map.yaml");
  std::cout << "loading map from file: " << map_filename << std::endl;
  YAML::Node data = YAML::LoadFile(map_filename);

  // load frames
  for(YAML::const_iterator it=data["frames"].begin(); it!=data["frames"].end(); ++it)
  {
    YAML::Node frame = *it;
    Vector3d t_wc(frame["T_world_cam"][0].as<double>(),
                  frame["T_world_cam"][1].as<double>(),
                  frame["T_world_cam"][2].as<double>());
    Eigen::Quaterniond q_wc(frame["T_world_cam"][6].as<double>(),
                     frame["T_world_cam"][3].as<double>(),
                     frame["T_world_cam"][4].as<double>(),
                     frame["T_world_cam"][5].as<double>());

    FramePtr f = std::make_shared<Frame>(
          frame["frame_id"].as<int>(),
          frame["timestamp"].as<double>(),
          cam,
          Transformation(q_wc, t_wc));
    map->addKeyframe(f, false);
  }
  std::cout << "loaded " << map->size() << " frames." << std::endl;

  // load points
  std::unordered_map<int, PointPtr> points;
  for(YAML::const_iterator it=data["points"].begin(); it!=data["points"].end(); ++it)
  {
    YAML::Node point = *it;
    PointPtr p = std::make_shared<Point>(
          point["point_id"].as<int>(),
          Vector3d(
            point["pos"][0].as<double>(),
            point["pos"][1].as<double>(),
            point["pos"][2].as<double>()));
    points.insert(std::make_pair(point["point_id"].as<int>(), p));
  }
  std::cout << "loaded " << points.size() << " points." << std::endl;

  // load features
  size_t n=0;
  for(YAML::const_iterator it=data["features"].begin(); it!=data["features"].end(); ++it)
  {
    YAML::Node ftr = *it;
    PointPtr point = points.find(ftr["point_id"].as<int>())->second;
    FramePtr frame = map->getKeyframeById(ftr["frame_id"].as<int>());
    Eigen::Vector2d px(ftr["px"][0].as<double>(), ftr["px"][1].as<double>());
    int level = ftr["level"].as<int>();

    Vector3d bearing_vector;
    cam->backProject3(px, &bearing_vector);
    CHECK(false);
    /* TODO(cfo)
    FeaturePtr f = std::make_shared<Feature>(
          frame, px, bearing_vector.normalized(), level);
    f->point = point;
    frame->addFeature(f);
    point->addObservation(frame, f);
    ++n;
    */
  }
  std::cout << "loaded " << n << " features." << std::endl;
  return true;
}

} // namespace io
} // namespace svo
