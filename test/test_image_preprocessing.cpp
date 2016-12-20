#include <glog/logging.h>
#include <gtest/gtest.h>

#include "image_preprocessing.hpp"

TEST(ImagePreprocessing, Instantiation) {

	google::InitGoogleLogging("ImagePreprocessing");
	google::InstallFailureSignalHandler();

    using namespace mrf;
    constexpr size_t width = 20;
    constexpr size_t height = 10;
    cv::Mat in{cv::Mat::eye(height, width, CV_32FC1)};
    cv::Mat out{gradientSobel(in)};
}
