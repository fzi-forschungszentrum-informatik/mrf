#pragma once

#include <opencv2/core/core.hpp>

namespace mrf {

cv::Mat edge(const cv::Mat&, const bool normalize = true);
cv::Mat blur(const cv::Mat&, const size_t& kernel_size);
cv::Mat norm_color(const cv::Mat&, const bool use_instance = false);
cv::Mat get_gray_image(const cv::Mat&);
}
