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

Please refer to the 'mrf_tool' for exemplary applications.

## History

2017-04-05 Initial commit

## Credits

Sascha Wirges <wirges(at)fzi.de>, Matthias Mayr <mayr(at)fzi.de>, Björn Roxin <roxinbj(at)gmail.com>

Partly based on the work of James Diebel and Sebastian Thrun, Stanford University.

## License

Copyright (C) 2017  FZI Forschungszentrum Informatik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
