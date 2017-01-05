#pragma once

#include <opencv2/core/core.hpp>

namespace mrf {

cv::Mat edge(const cv::Mat&, const bool normalize = true);
cv::Mat blur(const cv::Mat&, const size_t& kernel_size);
}
