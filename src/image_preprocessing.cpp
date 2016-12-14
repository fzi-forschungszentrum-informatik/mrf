#include "image_preprocessing.hpp"

namespace mrf {

cv::Mat gradientSobel(const cv::Mat& in, const bool normalize) {

    cv::Mat grad_x, grad_y;
    cv::Sobel(in, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(in, grad_y, CV_32F, 0, 1, 3);
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);

    cv::Mat out;
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, out);
    if (normalize) {
        cv::normalize(out, out, 0, 1, cv::NORM_MINMAX);
    }
    return out;
}
}
