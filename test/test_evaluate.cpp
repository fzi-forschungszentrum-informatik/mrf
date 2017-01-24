#include <Eigen/Eigen>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera_model_ortho.h"
#include "evaluate.hpp"

mrf::Data<pcl::PointXYZINormal> create(const size_t& rows,
                                       const size_t& cols,
                                       const double& delta) {
    using namespace mrf;

    using T = pcl::PointXYZINormal;
    const typename Data<T>::Cloud::Ptr cl{new typename Data<T>::Cloud};

    cl->height = rows;
    cl->width = cols;
    cl->resize(cl->width * cl->height);
    Eigen::MatrixXd depth{Eigen::MatrixXd::Zero(rows, cols)};

    for (size_t r = 0; r < cl->height; r++) {
        for (size_t c = 0; c < cl->width; c++) {
            T p;
            p.x = c;
            p.y = r;
            p.z = delta;
            cl->at(c, r) = p;
            depth(r, c) = delta;
        }
    }
    cv::Mat depth_image;
    cv::eigen2cv(depth, depth_image);
    return Data<T>(cl, depth_image);
}


TEST(Evaluate_new, Evaluate_new) {
    using namespace mrf;
    using namespace cv;
    google::InitGoogleLogging("Evaluate_new");
    google::InstallFailureSignalHandler();

    constexpr size_t rows = 20;
    constexpr size_t cols = 10;
    const int diff{-2};
    std::shared_ptr<CameraModelOrtho> cam{new CameraModelOrtho(cols, rows)};

    using Point = pcl::PointXYZINormal;
    const Data<Point> in{create(rows, cols, 0)};
    const Data<Point> out{create(rows, cols, diff)};
    const Quality q{mrf::evaluate(in, out, cam)};

    ASSERT_NEAR(q.depth_error_mean, diff, 1e-12);
    ASSERT_NEAR(q.depth_error_mean_abs, std::abs(diff), 1e-12);
    ASSERT_NEAR(q.depth_error_rms, std::abs(diff), 1e-12);
    ASSERT_NEAR(q.depth_error_median_abs, std::abs(diff), 1e-12);
    ASSERT_NEAR(q.depth_error_median, diff, 1e-12);
}
