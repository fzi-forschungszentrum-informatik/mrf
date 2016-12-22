#pragma once

#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "quality.hpp"

namespace mrf {

inline double median(const cv::Mat& in) {
    std::vector<double> v;
    in.reshape(0, 1).copyTo(v);
    auto median = begin(v);
    std::advance(median, v.size() / 2);
    std::nth_element(begin(v), median, end(v));
    return *median;
}

inline double squaredSum(const cv::Mat& in) {
    cv::Mat sq;
    cv::pow(in, 2, sq);
    return cv::sum(sq)[0];
}

inline Quality evaluate(const cv::Mat& est, const cv::Mat& ref) {
    CHECK_EQ(est.size, ref.size) << "Images do not have the same size";
    const size_t size = est.rows * est.cols;

    Quality q;

    q.depth_error = est - ref;

    using namespace cv;
    const Mat depth_error_abs{abs(q.depth_error)};
    q.depth_error_mean = sum(q.depth_error)[0] / size;
    q.depth_error_mean_abs = sum(depth_error_abs)[0] / size;
    q.depth_error_median = median(q.depth_error);
    q.depth_error_median_abs = median(depth_error_abs);
    q.depth_error_rms = std::sqrt(squaredSum(q.depth_error)) / size;
    return q;
}

inline size_t badMatchedPixels(const cv::Mat& in, const double& max_val) {
    size_t count{0};
    for (int r = 0; r < in.rows; r++) {
        for (int c = 0; c < in.cols; c++) {
            if (in.at<float>(r, c) > max_val) {
                count++;
            }
        }
    }
    return count;
}
}
