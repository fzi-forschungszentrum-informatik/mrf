#include "image_preprocessing.hpp"

#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace mrf {

cv::Mat edge(const cv::Mat& in, const bool normalize) {
    using namespace cv;

    Mat grad_x, grad_y;
    Scharr(in, grad_x, DataType<float>::type, 1, 0);
    Scharr(in, grad_y, DataType<float>::type, 0, 1);

    Mat normalized;
    addWeighted(abs(grad_x), 0.5, abs(grad_y), 0.5, 0, normalized);
    if (normalize) {
        cv::normalize(normalized, normalized, 0, 1, NORM_MINMAX);
    }

    Mat out{Mat::zeros(normalized.rows, normalized.cols, DataType<float>::type)};
    const int channels{normalized.channels()};
    float* ptr;
    for (int r = 0; r < normalized.rows; r++) {
        ptr = normalized.ptr<float>(r);
        for (int c = 0; c < normalized.cols; c++) {
            for (int d = 0; d < channels; d++) {
                out.at<float>(r, c) += ptr[c * channels + d];
            }
        }
    }

    if (normalize) {
        cv::normalize(out, out, 0, 1, NORM_MINMAX);
    }
    return out;
}

cv::Mat blur(const cv::Mat& in, const size_t& kernel_size) {
    cv::Mat out;
    cv::GaussianBlur(in, out, cv::Size(kernel_size, kernel_size), 0);
    return out;
}
}
