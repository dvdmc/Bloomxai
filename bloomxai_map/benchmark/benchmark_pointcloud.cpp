#include <benchmark/benchmark.h>

#include <filesystem>

#include "bloomxai_map/pcl_utils.hpp"
#include "bloomxai_map/semantic_map.hpp"
#include "octomap/OcTree.h"

using namespace Bloomxai;

static const double voxel_res = 0.02;
static auto filepath = std::filesystem::path(DATA_PATH) / "room_scan.pcd";

static void Bonxai_ComputeRay(benchmark::State& state) {
  std::vector<Eigen::Vector3d> points;
  ReadPointsFromPCD(filepath.generic_string(), points);

  std::vector<Bonxai::CoordT> ray;

  double inv_resolution = 1.0 / voxel_res;

  for (auto _ : state) {
    for (const auto& p : points) {
      const auto coord = Bonxai::PosToCoord(p, inv_resolution);
      Bloomxai::ComputeRay({0, 0, 0}, coord, ray);
    }
  }
}

static void OctoMap_ComputeRay(benchmark::State& state) {
  std::vector<Eigen::Vector3d> points;
  ReadPointsFromPCD(filepath.generic_string(), points);

  octomap::OcTree octree(voxel_res);
  octomap::KeyRay ray;

  for (auto _ : state) {
    for (const auto& p : points) {
      ray.reset();
      octree.computeRayKeys({0, 0, 0}, {float(p.x()), float(p.y()), float(p.z())}, ray);
    }
  }
}

static void Bonxai_InsertPointCloud(benchmark::State& state) {
  std::vector<Eigen::Vector3d> points;
  ReadPointsFromPCD(filepath.generic_string(), points);

  Bloomxai::SemanticMap bloomxai_map(voxel_res, 5);

  // Create a dummy vector of probs for the semantics
  std::vector<Eigen::VectorXf> probs(points.size(), Eigen::VectorXf::Constant(5, 1.0f/5.0f));

  Eigen::Vector3d origin(0, 0, 0);
  double max_distance = 10.0;

  for (auto _ : state)
  {
    bloomxai_map.insertPointCloud(points, probs, origin, max_distance);
  }
}

static void OctoMap_InsertPointCloud(benchmark::State& state) {
  std::vector<Eigen::Vector3d> points;
  ReadPointsFromPCD(filepath.generic_string(), points);
  octomap::Pointcloud pointcloud;
  for (const auto& p : points) {
    pointcloud.push_back(octomap::point3d(p.x(), p.y(), p.z()));
  }

  octomap::OcTree octree(voxel_res);

  octomap::point3d origin(0, 0, 0);
  double max_distance = 10.0;

  for (auto _ : state)
  {
    octree.insertPointCloud(pointcloud, origin, max_distance, false, true);
  }

}

// Register the function as a benchmark
BENCHMARK(Bonxai_ComputeRay);
BENCHMARK(OctoMap_ComputeRay);

BENCHMARK(Bonxai_InsertPointCloud);
BENCHMARK(OctoMap_InsertPointCloud);

// Run the benchmark
BENCHMARK_MAIN();
