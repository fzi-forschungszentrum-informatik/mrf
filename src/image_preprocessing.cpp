#include "image_preprocessing.hpp"

#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "cv_helper.hpp"

namespace mrf {

cv::Mat edge(const cv::Mat& in, const bool normalize) {
    using namespace cv;

    Mat grad_x, grad_y;
    Scharr(in, grad_x, DataType<float>::type, 1, 0);
    Scharr(in, grad_y, DataType<float>::type, 0, 1);

    Mat normalized;
    addWeighted(abs(grad_x), 0.5, abs(grad_y), 0.5, 0, normalized);
    if (normalize)
        cv::normalize(normalized, normalized, 0, 1, NORM_MINMAX);

    Mat out{Mat::zeros(normalized.rows, normalized.cols, DataType<float>::type)};
    for (int r = 0; r < normalized.rows; r++)
        for (int c = 0; c < normalized.cols; c++)
            out.at<float>(r, c) = getVector<float>(normalized, r, c).sum();

    if (normalize)
        cv::normalize(out, out, 0, 1, NORM_MINMAX);
    return out;
}

cv::Mat blur(const cv::Mat& in, const size_t& kernel_size) {
    cv::Mat out;
    cv::GaussianBlur(in, out, cv::Size(kernel_size, kernel_size), 0);
    return out;
}

cv::Mat norm_color(const cv::Mat& in, const bool use_instance) {
    using namespace cv;
    const int channels{in.channels()};
    std::vector<Mat> in_split(in.channels());
    std::vector<Mat> out_split;
    Mat out;
    split(in, in_split);
    for (size_t i = 0; i < in.channels(); i++) {
        double minVal;
        double maxVal;
        Point minLoc;
        Point maxLoc;

        minMaxLoc(in_split[i], &minVal, &maxVal, &minLoc, &maxLoc);
        LOG(INFO) << "Channel " << i << " min + max values: " << minVal << " + " << maxVal;
        normalize(in_split[i], in_split[i], 0, 1, cv::NORM_MINMAX);
    }
    for (size_t i = 0; i < in.channels() - 1; i++) { //> -1 only color channels
        out_split.emplace_back(in_split[i]);
    }
    LOG(INFO) << "Use Instance: " << use_instance;
    if (use_instance) {
        out_split.emplace_back(in_split.back()); //> because instance is always the last channel
    }
    merge(out_split, out);
    return out;
}
}
