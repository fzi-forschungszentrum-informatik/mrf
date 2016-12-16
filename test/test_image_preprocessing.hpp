#pragma once

#include <gtest/gtest.h>

#include "image_preprocessing.hpp"

TEST(ImagePreprocessing, Instantiation) {
    using namespace mrf;

    constexpr size_t width = 100;
	constexpr size_t height = 100;
    cv::Mat in(height, width, CV_32FC1, double(0));
    cv::Mat out = gradientSobel(in);
}
