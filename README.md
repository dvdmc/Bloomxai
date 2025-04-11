# Bloomxai

Bloomxai is a version of the `boxai_map` that implements dynamic size vectors in the voxel structure to allow `semantic mapping` on top of [Bonxai](https://github.com/facontidavide/Bonxai). Rought and quick benchmarking indicates that, while slower than Bonxai's probabilistic map, Bloomxai is still faster than Ocotmap's implementation for probabilistic occupancy mapping.

As input, it expects a semantic point cloud in the form of the [sensors_tools]() package.