#pragma once

#include <svo/common/types.h>
#include <glog/logging.h>
namespace svo
{
/// We divide the image into a grid of cells and try to find maximally one
/// feature per cell. This is to ensure good distribution of features in the
/// image.
class OccupandyGrid2D
{
public:
  using Grid = std::vector<bool>;
  using FeatureGrid = std::vector<Keypoint>;

  OccupandyGrid2D(int cell_size, int n_cols, int n_rows)
    : cell_size(cell_size)
    , n_cols(n_cols)
    , n_rows(n_rows)
    , occupancy_(n_cols * n_rows, false)
    , feature_occupancy_(n_cols * n_rows, Keypoint(0, 0))
  {
  }

  OccupandyGrid2D(const OccupandyGrid2D& rhs)
    : cell_size(rhs.cell_size)
    , n_cols(rhs.n_cols)
    , n_rows(rhs.n_rows)
    , occupancy_(rhs.occupancy_)
    , feature_occupancy_(rhs.feature_occupancy_)
  {
  }

  ~OccupandyGrid2D() = default;

  inline static int getNCell(
      const int n_pixels, const int size)
  {
    return std::ceil(static_cast<double>(n_pixels)
                      / static_cast<double>(size));
  }

  const int cell_size;
  const int n_cols;
  const int n_rows;
  Grid occupancy_;
  FeatureGrid feature_occupancy_;

  inline void reset()
  {
    std::fill(occupancy_.begin(), occupancy_.end(), false);
  }

  inline size_t size()
  {
    return occupancy_.size();
  }

  inline bool empty()
  {
    return occupancy_.empty();
  }

  inline bool isOccupied(const size_t cell_index)
  {
    CHECK_LT(cell_index, occupancy_.size());
    return occupancy_[cell_index];
  }

  inline void setOccupied(const size_t cell_index)
  {
    CHECK_LT(cell_index, occupancy_.size());
    occupancy_[cell_index] = true;
  }

  inline int numOccupied() const
  {
    return std::count(occupancy_.begin(), occupancy_.end(), true);
  }

  template <typename Derived>
  size_t getCellIndex(const Eigen::MatrixBase<Derived>& px) const
  {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    return std::floor(px(1) / cell_size) * n_cols + std::floor(px(0) / cell_size);
//    return static_cast<size_t>((px(1)) / cell_size * n_cols +
//                               (px(0)) / cell_size);
  }

  inline size_t getCellIndex(int x, int y, int scale = 1) const
  {
    return getCellIndex(Eigen::Vector2d(scale * x, scale * y));
//    return static_cast<size_t>((scale * y) / cell_size * n_cols +
//                               (scale * x) / cell_size);
  }

  inline void fillWithKeypoints(const Keypoints& keypoints)
  {
    // TODO(cfo): could be implemented using block operations.
    for (int i = 0; i < keypoints.cols(); ++i)
    {
      const int int_x = static_cast<int>(keypoints(0, i));
      const int int_y = static_cast<int>(keypoints(1, i));
      const size_t idx = getCellIndex(int_x, int_y, 1);
      occupancy_.at(idx) = true;
      feature_occupancy_.at(idx) = Keypoint(keypoints(0, i), keypoints(1, i));
    }
  }

};

using OccGrid2DPtr = std::shared_ptr<OccupandyGrid2D>;

}  // namespace svo
