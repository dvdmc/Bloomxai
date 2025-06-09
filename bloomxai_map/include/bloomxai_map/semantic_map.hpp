#pragma once

#include <iostream>
#include <bonxai/bonxai.hpp>
#include <eigen3/Eigen/Geometry>

namespace Bloomxai {
using namespace Bonxai;

template <class Functor>
void RayIterator(const CoordT& key_origin, const CoordT& key_end,
                 const Functor& func);

inline void ComputeRay(const CoordT& key_origin, const CoordT& key_end,
                       std::vector<CoordT>& ray) {
  ray.clear();
  RayIterator(key_origin, key_end, [&ray](const CoordT& coord) {
    ray.push_back(coord);
    return true;
  });
}

/**
 * @brief The SemanticMap works similarly to ProbabilisticMap but includes
 * different fusion methods for semantic information. Semantics might be:
 * probability vector for classes, features, or labels.
 */
class SemanticMap {
 private:
  int _sem_dim;

 public:
  using Vector3D = Eigen::Vector3d;
  using VSemanticProb = Eigen::VectorXf;
  using VSemanticLogOds = Eigen::VectorXi;

  static constexpr float kFixedPrecision = 1e6f;
  static constexpr float kEps = 1e-9f;

  /// Compute the logds, but return the result as an integer,
  /// The real number is represented as a fixed precision
  /// integer (6 decimals after the comma)
  [[nodiscard]] static constexpr int32_t logods(float prob) {
    return int32_t(kFixedPrecision * std::log((prob + kEps) / (1.0 - prob + kEps)));
  }

  /// Expect the fixed comma value returned by logods()
  [[nodiscard]] static constexpr float prob(int32_t logods_fixed) {
    float logods = float(logods_fixed) * (1.0 / kFixedPrecision);
    return (1.0 - 1.0 / (1.0 + std::exp(logods)));
  }

  // TODO(dvdmc): Check if this causes too much artifacts.
  [[nodiscard]] VSemanticLogOds vlogods(const VSemanticProb &probs) {
    VSemanticLogOds vec(_sem_dim);
    for (int i = 0; i < _sem_dim; ++i) {
      vec[i] = logods(probs[i]);
    }
    return vec;
  }

  [[nodiscard]] VSemanticProb vprob(const VSemanticLogOds& logodds_vec) {
    VSemanticProb vec(_sem_dim);
    for (int i = 0; i < _sem_dim; ++i) {
      vec[i] = prob(logodds_vec[i]);
    }
    return vec;
  }

  static const int32_t UnknownProbability;
  const float UnknownSemProbs;
  const float UnknownSemLogOdds;

  struct SemCellT {
    // variable used to check if a cell was already updated in this loop
    int32_t update_id : 4;
    // the probability of the cell to be occupied
    int32_t occ_prob_log : 28;

    VSemanticLogOds sem_prob_log;

    SemCellT() : update_id(0), occ_prob_log(UnknownProbability) {};

    void init(int sem_dim, int value) {
      sem_prob_log.resize(sem_dim);
      sem_prob_log.setConstant(value);
    }

    [[nodiscard]] int argmax(const VSemanticLogOds& vec) {
        int min = std::numeric_limits<int>::lowest();
        int maxIndex = 0;
        for (int i = 0; i < vec.size(); i++) {
            if (vec[i] > min) {
                min = vec[i];
                maxIndex = i;
            }
        }
        return static_cast<int>(maxIndex);
    }
    
  };

  SemCellT* ensureCellInitalized(SemCellT* cell) {
    if (cell->sem_prob_log.size() == 0) {
      // Uninformed prior initialization
      cell->init(_sem_dim, UnknownSemLogOdds);
    }
    return cell;
  }

  struct Options {
    // Occupancy
    int32_t prob_miss_log = logods(0.4f);
    int32_t prob_hit_log = logods(0.7f);

    int32_t clamp_min_log = logods(0.12f);
    int32_t clamp_max_log = logods(0.97f);

    int32_t occupancy_threshold_log = logods(0.5f);

    // Semantics
    VSemanticProb prob_reg;
    float alpha_reg = 0.3f;
    int32_t clamp_min_sem_log = logods(0.12f);
    int32_t clamp_max_sem_log = logods(0.97f);

    Options(int sem_dim, float value)
        : prob_reg(VSemanticProb(sem_dim).setConstant(value)) {}
  };

  SemanticMap(double resolution, int sem_dim);

  [[nodiscard]] VoxelGrid<SemCellT>& grid();

  [[nodiscard]] const VoxelGrid<SemCellT>& grid() const;

  [[nodiscard]] const Options& options() const;

  void setOptions(const Options& options);

  /**
   * @brief insertPointCloud will update the probability map
   * with a new set of detections.
   * The template function can accept points of different types,
   * such as pcl:Point, Eigen::Vector or Bonxai::Point3d
   *
   * Both origin and points must be in world coordinates
   *
   * @param points   a vector of points with associated semantic measurements
   * @param semantics a vector of semantic measurements
   * @param origin   origin of the point cloud
   * @param max_range max range of the ray, if exceeded, we will use that to
   * compute a free space
   */
  template <typename PointT, typename AllocatorP, typename AllocatorSem>
  void insertPointCloud(
      const std::vector<PointT, AllocatorP>& points,
      const std::vector<VSemanticProb, AllocatorSem>& semantics,
      const PointT& origin, double max_range);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addHitPoint(const Vector3D& point, const VSemanticProb& semantics);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addMissPoint(const Vector3D& point);

  VSemanticProb regularizeSemantic(const VSemanticProb& input) const;

  [[nodiscard]] bool isOccupied(const Bonxai::CoordT& coord) const;

  [[nodiscard]] bool isUnknown(const Bonxai::CoordT& coord) const;

  [[nodiscard]] bool isFree(const Bonxai::CoordT& coord) const;

  void getOccupiedVoxels(std::vector<Bonxai::CoordT>& coords);

  void getOccupiedVoxelsAndClass(std::vector<Bonxai::CoordT>& coords, std::vector<int>& classes);

  void getFreeVoxels(std::vector<Bonxai::CoordT>& coords);

  template <typename PointT>
  void getOccupiedVoxels(std::vector<PointT>& points) {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getOccupiedVoxels(coords);
    for (const auto& coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }

  template <typename PointT>
  void getOccupiedVoxelsAndClass(std::vector<PointT>& points, std::vector<int>& point_labels) {
    thread_local std::vector<Bonxai::CoordT> coords;
    thread_local std::vector<int> labels;
    coords.clear();
    labels.clear();
    getOccupiedVoxelsAndClass(coords, labels);
    for (size_t i = 0; i < coords.size(); i++) {
      const auto coord = coords[i];
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
      point_labels.emplace_back(labels[i]);
    }
  }

 private:
  Options _options;
  VoxelGrid<SemCellT> _grid;
  uint8_t _update_count = 1;

  std::vector<CoordT> _miss_coords;
  std::vector<CoordT> _hit_coords;

  mutable Bonxai::VoxelGrid<SemCellT>::Accessor _accessor;

  void updateFreeCells(const Vector3D& origin);
};

//--------------------------------------------------

template <typename PointT, typename AllocatorP, typename AllocatorSem>
inline void SemanticMap::insertPointCloud(
    const std::vector<PointT, AllocatorP>& points,
    const std::vector<SemanticMap::VSemanticProb, AllocatorSem>& semantics,
    const PointT& origin, double max_range) {
  const auto from = ConvertPoint<Vector3D>(origin);
  const double max_range_sqr = max_range * max_range;

  for (size_t i = 0; i < points.size(); ++i) {
    const auto& point = points[i];
    const auto& semantic = semantics[i];
    const auto to = ConvertPoint<Vector3D>(point);
    Vector3D vect(to - from);
    const double squared_norm = vect.squaredNorm();

    // Points that exceed the max_range will create a cleaning ray
    if (squared_norm >= max_range_sqr) {
      // The new point will have distance == max_range from origin
      const Vector3D new_point =
          from + ((vect / std::sqrt(squared_norm)) * max_range);
      addMissPoint(new_point);
    } else {
      addHitPoint(to, semantic);
    }
  }
  updateFreeCells(from);
}

template <class Functor>
inline void RayIterator(const CoordT& key_origin, const CoordT& key_end,
                        const Functor& func) {
  if (key_origin == key_end) {
    return;
  }
  if (!func(key_origin)) {
    return;
  }

  CoordT error = {0, 0, 0};
  CoordT coord = key_origin;
  CoordT delta = (key_end - coord);
  const CoordT step = {delta.x < 0 ? -1 : 1, delta.y < 0 ? -1 : 1,
                       delta.z < 0 ? -1 : 1};

  delta = {delta.x < 0 ? -delta.x : delta.x, delta.y < 0 ? -delta.y : delta.y,
           delta.z < 0 ? -delta.z : delta.z};

  const int max = std::max(std::max(delta.x, delta.y), delta.z);

  // maximum change of any coordinate
  for (int i = 0; i < max - 1; ++i) {
    // update errors
    error = error + delta;
    // manual loop unrolling
    if ((error.x << 1) >= max) {
      coord.x += step.x;
      error.x -= max;
    }
    if ((error.y << 1) >= max) {
      coord.y += step.y;
      error.y -= max;
    }
    if ((error.z << 1) >= max) {
      coord.z += step.z;
      error.z -= max;
    }
    if (!func(coord)) {
      return;
    }
  }
}

}  // namespace Bloomxai
