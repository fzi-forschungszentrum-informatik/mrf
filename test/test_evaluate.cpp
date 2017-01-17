#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>

#include "camera_model_ortho.h"
#include "evaluate.hpp"

mrf::Data<pcl::PointXYZINormal> create(const size_t& rows, const size_t& cols) {
    using namespace mrf;

    using T = pcl::PointXYZINormal;
    const typename Data<T>::Cloud::Ptr cl{new typename Data<T>::Cloud};

    cl->height = rows;
    cl->width = cols;
    cl->resize(cl->width * cl->height);

    for (size_t r = 0; r < cl->height; r++) {
        for (size_t c = 0; c < cl->width; c++) {
            cl->at(c, r) = T();
        }
    }
    cv::Mat img{cv::Mat::zeros(rows, cols, cv::DataType<float>::type)};
    return Data<T>(cl, img);
}

TEST(Evaluate, Evaluate) {
    using namespace mrf;

    google::InitGoogleLogging("Evaluate");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 15;
    constexpr size_t cols = 20;
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using Point = pcl::PointXYZINormal;
    const Data<Point> in{create(rows, cols)};
    const Data<Point> out{in};

    const Quality q{evaluate(in, out, cam)};

    ASSERT_NEAR(q.depth_error_mean, 0, 1e-12);
    ASSERT_NEAR(q.depth_error_mean_abs, 0, 1e-10);
}
