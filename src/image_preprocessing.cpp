#include "image_preprocessing.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace mrf {

cv::Mat gradientSobel(const cv::Mat& in, const bool normalize) {
    using namespace cv;

    Mat grad_x, grad_y;
    Sobel(in, grad_x, CV_32FC1, 1, 0, 3);
    Sobel(in, grad_y, CV_32FC1, 0, 1, 3);
    convertScaleAbs(grad_x, grad_x);
    convertScaleAbs(grad_y, grad_y);

    Mat out;
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, out);
    if (normalize) {
        cv::normalize(out, out, 0, 1, NORM_MINMAX);
    }
    return out;
}
}
