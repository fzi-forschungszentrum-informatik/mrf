#pragma once

#include <map>
#include <opencv2/core/core.hpp>

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p,
                                const cv::Mat& img,
                                const int& row_max,
                                const int& col_max,
                                const int& row_min = 0,
                                const int& col_min = 0);
}
