# mrf

Markov-Random-Field based guided depth upsampling and reconstruction given camera images and laser observations.

For further information, please refer to our publication "Guided Depth Upsampling for Precise Mapping of Urban Environments", presented at the IEEE Intelligent Vehicles Symposium 2017, Redondo Beach, CA, USA.

## Installation

Dependencies required:
- [Ceres Solver](http://ceres-solver.org/)
- [Point Cloud Library](http://www.pointclouds.org/)
- [OpenCV](http://opencv.org/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)

You can use [CMake](https://cmake.org/) to build this package.
However, we recommend using [catkin](http://wiki.ros.org/catkin) which is part of [ROS](http://www.ros.org/).

## Usage

The 'Solver' class represents the interface for depth upsampling.
It is initialized with a camera model and an optional parameters structure.
Please refer to 'parameters.hpp' for hints on the different parameters.
To solve a depth upsampling problem a 'Data' structure must be provided that consists of a 3D point cloud, a feature image and a transform between laser and camera.

This package currently contains two standalone applications:
- `eval_planes`
- `eval_scenenet`

### `eval_planes`

This is a simple program showing our upsampling approach on an artificially generated set of planes.
Please follow the command line help, starting the program with the `--help` argument.

### `eval_scenenet`

We evaluated our approach on the [scenenet dataset](https://arxiv.org/abs/1612.05079) which is publicly available.
Please follow the command line help, starting the program with the `--help` argument.

## History

2017-04-05 Initial commit

## Credits

Sascha Wirges <wirges(at)fzi.de>,
Matthias Mayr <mayr(at)fzi.de>,
Björn Roxin <roxinbj(at)gmail.com>

Partly based on the work of James Diebel and Sebastian Thrun, Stanford University.

