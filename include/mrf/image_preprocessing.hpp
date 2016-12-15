#pragma once

#include <opencv2/core/core.hpp>

namespace mrf {

cv::Mat gradientSobel(const cv::Mat& in, const bool normalize = true);
}
