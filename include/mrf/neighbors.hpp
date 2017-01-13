#pragma once

#include <map>
#include <opencv2/core/core.hpp>

#include "neighbor_relation.hpp"
#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

std::map<NeighborRelation, Pixel> getNeighbors(const Pixel& p,
                                               const cv::Mat& img,
                                               const Parameters::Neighborhood& mode);
}
