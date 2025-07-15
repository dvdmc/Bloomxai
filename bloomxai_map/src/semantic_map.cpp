#include "bloomxai_map/semantic_map.hpp"

#include <eigen3/Eigen/Geometry>
#include <unordered_set>

namespace Bloomxai {

const int32_t SemanticMap::UnknownProbability = SemanticMap::logods(0.5f);

VoxelGrid<SemanticMap::SemCellT>& SemanticMap::grid() {
  return _grid;
}

SemanticMap::SemanticMap(double resolution, int sem_dim)
    : _sem_dim(sem_dim),
      _resolution(resolution),
      UnknownSemProbs(1.0f / sem_dim),
      UnknownSemLogOdds(SemanticMap::logods(UnknownSemProbs)),
      _options(sem_dim, UnknownSemProbs),
      _grid(resolution),
      _accessor(_grid.createAccessor()) {}

const VoxelGrid<SemanticMap::SemCellT>& SemanticMap::grid() const {
  return _grid;
}

const SemanticMap::Options& SemanticMap::options() const {
  return _options;
}

void SemanticMap::setOptions(const Options& options) {
  _options = options;
}

void SemanticMap::addHitPoint(const Vector3D& point, const SemanticMap::VSemanticProb& semantics) {
  const auto coord = _grid.posToCoord(point);
  SemCellT* cell = SemanticMap::ensureCellInitalized(_accessor.value(coord, true));

  SemanticMap::VSemanticLogOds sem_log_odds =
      SemanticMap::vlogods(SemanticMap::regularizeSemantic(semantics));
  
  // TODO(dvdmc): Check if only updating once causes artifacts
  if (cell->update_id != _update_count) {
    cell->occ_prob_log =
        std::max(cell->occ_prob_log + _options.prob_hit_log, _options.clamp_min_log);

    cell->sem_prob_log = cell->sem_prob_log + sem_log_odds;
    // TODO(dvdmc): Check if clamping causes artifacts
    cell->sem_prob_log.cwiseMax(_options.clamp_min_sem_log).cwiseMin(_options.clamp_max_sem_log);
    
    cell->update_id = _update_count;
    _hit_coords.push_back(coord);
  }
}

void SemanticMap::addMissPoint(const Vector3D& point) {
  const auto coord = _grid.posToCoord(point);
  SemCellT* cell = SemanticMap::ensureCellInitalized(_accessor.value(coord, true));

  if (cell->update_id != _update_count) {
    cell->occ_prob_log =
        std::max(cell->occ_prob_log + _options.prob_miss_log, _options.clamp_min_log);

    // Make the semantics more uncertain
    cell->sem_prob_log = SemanticMap::vlogods(
        SemanticMap::regularizeSemantic(SemanticMap::vprob(cell->sem_prob_log)));

    cell->update_id = _update_count;
    _miss_coords.push_back(coord);
  }
}

SemanticMap::VSemanticProb SemanticMap::regularizeSemantic(
    const SemanticMap::VSemanticProb& input) const {
  return (1 - _options.alpha_reg) * input + _options.alpha_reg * _options.prob_reg;
}

bool SemanticMap::isOccupied(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->occ_prob_log > _options.occupancy_threshold_log;
  }
  return false;
}

bool SemanticMap::isUnknown(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->occ_prob_log == _options.occupancy_threshold_log;
  }
  return false;
}

bool SemanticMap::isFree(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->occ_prob_log < _options.occupancy_threshold_log;
  }
  return false;
}

void SemanticMap::updateFreeCells(const Vector3D& origin) {
  auto accessor = _grid.createAccessor();

  // same as addMissPoint, but using lambda will force inlining
  auto clearPoint = [this, &accessor](const CoordT& coord) {
    SemCellT* cell = SemanticMap::ensureCellInitalized(accessor.value(coord, true));
    if (cell->update_id != _update_count) {
      cell->occ_prob_log =
          std::max(cell->occ_prob_log + _options.prob_miss_log, _options.clamp_min_log);

      // Make the semantics more uncertain
      cell->sem_prob_log = SemanticMap::vlogods(
          SemanticMap::regularizeSemantic(SemanticMap::vprob(cell->sem_prob_log)));

      cell->update_id = _update_count;
    }
    return true;
  };

  const auto coord_origin = _grid.posToCoord(origin);

  for (const auto& coord_end : _hit_coords) {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _hit_coords.clear();

  for (const auto& coord_end : _miss_coords) {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _miss_coords.clear();

  if (++_update_count == 4) {
    _update_count = 1;
  }
}

void SemanticMap::getOccupiedVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
}

void SemanticMap::getMapLimits(std::vector<float>& min, std::vector<float>& max) const {
  min.clear();
  max.clear();

  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    const auto p = _grid.coordToPos(coord);
    if (p.x < min[0])
      min[0] = p.x;
    if (p.x > max[0])
      max[0] = p.x;
    if (p.y < min[1])
      min[1] = p.y;
    if (p.y > max[1])
      max[1] = p.y;
    if (p.z < min[2])
      min[2] = p.z;
    if (p.z > max[2])
      max[2] = p.z;
  };
  _grid.forEachCell(visitor);
}

std::vector<int> SemanticMap::getMapXYSize() const {
  std::vector<float> min, max;
  getMapLimits(min, max);
  return {int(max[0] - min[0]), int(max[1] - min[1])};
}

void SemanticMap::getOccupiedVoxelsAndClass(
    std::vector<CoordT>& coords, std::vector<int>& classes) {
  coords.clear();
  classes.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > _options.occupancy_threshold_log) {
      coords.push_back(coord);
      classes.push_back(cell.argmax(cell.sem_prob_log));
    }
  };
  _grid.forEachCell(visitor);
}

void SemanticMap::getFreeVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log < _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
}

}  // namespace Bloomxai