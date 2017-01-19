#pragma once

#include <opencv2/core/core.hpp>

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p,
                                const cv::Mat& img,
                                const Parameters::Neighborhood& mode);
}
