#include "image_preprocessing.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace mrf {

cv::Mat edge(const cv::Mat& in, const bool normalize) {
    using namespace cv;

    Mat grad_x, grad_y;
    Scharr(in, grad_x, cv::DataType<float>::type, 1, 0);
    Scharr(in, grad_y, cv::DataType<float>::type, 0, 1);

    Mat out;
    addWeighted(abs(grad_x), 0.5, abs(grad_y), 0.5, 0, out);
    if (normalize) {
        cv::normalize(out, out, 0, 1, NORM_MINMAX);
    }

    if (out.channels() > 1) {
    	cv::cvtColor(out, out, CV_BGR2GRAY);
    }

    return out;
}

cv::Mat blur(const cv::Mat& in, const size_t& kernel_size) {
    cv::Mat out;
    cv::GaussianBlur(in, out, cv::Size(kernel_size, kernel_size), 0);
    return out;
}
}
