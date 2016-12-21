#pragma once

#include <opencv2/core/core.hpp>

#include "quality.hpp"


namespace mrf {

inline Quality evaluate(const cv::Mat& est, const cv::Mat& ref) {
    Quality q;
    q.depth_error = est - ref;
    q.depth_error_avg = cv::mean(q.depth_error);
    return q;
}
}
