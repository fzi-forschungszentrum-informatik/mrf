#pragma once

#include <opencv/cv.h>

namespace mrf {

cv::Mat gradientSobel(const cv::Mat& in, const bool normalize = true);
}
