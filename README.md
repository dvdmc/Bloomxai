# Bloomxai

![Bloomxai](doc/bloomxai.png)

Bloomxai is a version of the `bonxai_map` that implements dynamic size vectors in the voxel structure to allow `semantic mapping` on top of [Bonxai](https://github.com/facontidavide/Bonxai). Quick benchmarking indicates that, while slower than Bonxai's probabilistic map, Bloomxai is still faster than Ocotmap's implementation for probabilistic occupancy mapping. The integration time of point clouds takes ~2ms. Thus, you will be mostly limited by the inference time of your semantic network.

As input, it expects a semantic point cloud in the form of the [sensors_tools](https://github.com/dvdmc/sensors_tools) package in the `jazzy` branch. Currently, this package is being reworked to support different semantic inference networks but the construction of the point cloud will remain stable (see [generate_point_cloud_semantics_msg](https://github.com/dvdmc/sensors_tools/blob/06f905d2261c00a6ef070166711c471443824c86/sensors_tools_ros/sensors_tools_ros/semantic_ros.py#L438)). The storage use is probably another issue to solve depending on the size of your semantic vectors.

## Install

Clone the repository in your ROS 2 workspace and build the package

```
git clone git@github.com:dvdmc/Bloomxai.git
colcon build --packages-select bloomxai_ros --symlink-install
```

If you would like to use [sensors_tools](https://github.com/dvdmc/sensors_tools), follow its installation instructions.

## Usage

We provide launch files to run the node. Please, check that the topics correspond to yours. Currently, the set of semantic classes has to be selected manually in the `bloomxai.yaml` config file.

## Roadmap

- Implement a static version that defines the semantic vector size at compile time.
- Include different fusion methods.
- Have a better integration with Bonxai by implementing a custom allocator.