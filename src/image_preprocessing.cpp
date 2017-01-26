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
        if (i != in.channels()) {
            normalize(in_split[i], in_split[i], 0, 1, cv::NORM_MINMAX);
        }
    }
    merge(in_split, out);
    return out;
}

cv::Mat get_gray_image(const cv::Mat& in) {
    using namespace cv;
    const int channels{in.channels()};
    cv::Mat out;
    if (channels == 3) {
        cvtColor(in, out, CV_BGR2GRAY);
        return out;
    }
    if (channels == 1) {
        return in;
    }
    if (channels == 2) {
        std::vector<Mat> in_split(channels);
        split(in, in_split);
        return in_split[0];
    }
    if (channels == 4) {
        std::vector<Mat> in_split(channels);
        split(in, in_split);
        std::vector<Mat> out_split;
        for (size_t c = 0; c < 3; c++) {
            out_split.emplace_back(in_split[c]);
        }
        merge(out_split, out);
        cvtColor(out, out, CV_BGR2GRAY);
        return out;
    }

    return Mat::zeros(in.rows, in.cols, 5);
}
}
