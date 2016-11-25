
#include "gtest/gtest.h"

#include <data.hpp>

TEST(MrfData, DataImport) {
    // TODO: Add your test code here
    cv::Mat cv_image = (cv::Mat_<float>(3, 4) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    Eigen::VectorXf depths(12);
    depths << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    Eigen::VectorXf certainty(12);
    certainty << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    const int width{4};
    ASSERT_EQ(width, 4);

    const int cv_size{cv_image.rows * cv_image.cols};
    const int depth_size{depths.size()};
    const int certain_size{certainty.size()};
    ASSERT_EQ(cv_size, 12);
    ASSERT_EQ(depth_size, 12);
    ASSERT_EQ(certain_size, 12);



    mrf::Data test_data(cv_image, depths, certainty, width);

    const int data_image_size{test_data.image.size()};
    const int data_depth_size{test_data.depth.size()};
    const int data_certainty_size{test_data.certainty.size()};

    ASSERT_EQ(data_image_size, 12);
    ASSERT_EQ(data_depth_size, 12);
    ASSERT_EQ(data_certainty_size, 12);

    for (int i = 1; i < 13; i++) {
        const float data_val_image{test_data.image(i-1)};
        const float data_val_depth{test_data.depth(i-1)};
        const float data_val_cer{test_data.certainty(i-1)};

        ASSERT_EQ(data_val_image, i);
        ASSERT_EQ(data_val_depth, i);
        ASSERT_EQ(data_val_cer, i);
    }
}
