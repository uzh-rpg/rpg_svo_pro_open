#include <svo_ros/csv_dataset_reader.h>

#include <fstream>
#include <iostream>

#include <svo/common/conversions.h>

namespace svo {

namespace utils {

void checkHeaderAndOpenStream(
    const std::string& filename,
    const std::string& header,
    std::ifstream* fs)
{
  CHECK_NOTNULL(fs);
  VLOG(1) << "Reading file " << filename;
  fs->open(filename.c_str(), std::ios::in);
  CHECK(fs);
  CHECK(fs->is_open()) << "Failed to open file " << filename;
  CHECK(!fs->eof()) << "File seems to contain no content!";

  std::string line;
  std::getline(*fs, line);
  CHECK_EQ(line, header) << "Invalid header.";
}

void readLine(
    const std::string& line,
    const size_t num_elements,
    std::vector<std::string>* elements)
{
  elements->resize(num_elements);
  std::stringstream line_stream(line);
  for (size_t i = 0u; i < num_elements; ++i)
  {
    CHECK(!line_stream.eof());
    std::getline(line_stream, elements->at(i), ',');
    CHECK(!elements->at(i).empty());
  }
}

} // namespace utils

CsvDatasetReader::CsvDatasetReader(
    const std::string& csv_directory,
    const std::vector<size_t> imu_indices,
    const std::vector<size_t> camera_indices,
    const std::vector<size_t> track_indices)
{
  VLOG(1) << "Loading csv dataset from directory \"" << csv_directory << "\".";

  CHECK_LE(imu_indices.size(), 1u) << "Using multiple IMUs not implemented";
  for(size_t i : imu_indices)
  {
    loadImuData(csv_directory + "/imu" + std::to_string(i), 0u);
  }

  for(size_t i : camera_indices)
  {
    std::string dir = csv_directory + "/cam" + std::to_string(i);
    loadCameraData(
        dir, i, 100 * common::conversions::kMilliSecondsToNanoSeconds);
  }

  for(size_t i : track_indices)
  {
    std::string dir = csv_directory + "/tracks" + std::to_string(i);
    loadFeatureTracksData(
        dir, i, 80 * common::conversions::kMilliSecondsToNanoSeconds);
  }

  VLOG(1) << "done.";
}

void CsvDatasetReader::loadImuData(const std::string data_dir, const int64_t playback_delay)
{
  const std::string kHeader = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]";
  std::ifstream fs;
  utils::checkHeaderAndOpenStream(data_dir+"/data.csv", kHeader, &fs);
  std::string line;
  size_t i = 0;
  while(std::getline(fs, line))
  {
    std::vector<std::string> elements(7u);
    utils::readLine(line, 7u, &elements);
    Eigen::Matrix<double, 6, 1> acc_gyr_measurement;
    acc_gyr_measurement << std::stod(elements[4]),  // acc_x
                           std::stod(elements[5]),  // acc_y
                           std::stod(elements[6]),  // acc_x
                           std::stod(elements[1]),  // gyr_x
                           std::stod(elements[2]),  // gyr_y
                           std::stod(elements[3]);  // gyr_z
    dataset::ImuMeasurement::Ptr imu_measurement(
        new dataset::ImuMeasurement(std::stoll(elements[0]), acc_gyr_measurement));

    buffer_.insert(std::make_pair(
                     imu_measurement->timestamp_nanoseconds + playback_delay,
                     imu_measurement));
    ++i;
  }
  VLOG(2) << "Loaded " << i << " IMU measurements.";
  fs.close();
  CHECK(!fs.is_open());
}

void CsvDatasetReader::loadCameraData(
    const std::string& data_dir,
    const size_t camera_index,
    int64_t playback_delay)
{
  const std::string kHeader = "#timestamp [ns],filename";
  std::ifstream fs;
  utils::checkHeaderAndOpenStream(data_dir+"/data.csv", kHeader, &fs);
  std::string line;
  size_t i = 0;
  while(std::getline(fs, line))
  {
    std::vector<std::string> elements(2u);
    utils::readLine(line, 2u, &elements);
    dataset::CameraMeasurement::Ptr camera_measurement(
        new dataset::CameraMeasurement(
            std::stoll(elements[0]), camera_index, data_dir + "/data/" + elements[1]));

    buffer_.insert(std::make_pair(
                     camera_measurement->timestamp_nanoseconds + playback_delay,
                     camera_measurement));
    ++i;
  }
  VLOG(2) << "Loaded " << i << " camera measurements.";
  fs.close();
  CHECK(!fs.is_open());
}

void CsvDatasetReader::loadFeatureTracksData(
    const std::string& /*data_dir*/,
    const size_t /*camera_index*/,
    int64_t /*playback_delay*/)
{
  LOG(FATAL) << "Not implemented";
}

} // namespace svo
