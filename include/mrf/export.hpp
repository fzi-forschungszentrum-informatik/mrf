#pragma once

#include <camera_models/camera_model.h>
#include <glog/logging.h>
#include <Eigen/src/Geometry/Hyperplane.h>
#include <Eigen/src/Geometry/ParametrizedLine.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>

#include "data.hpp"
#include "image_preprocessing.hpp"

namespace mrf {

template <typename T>
void exportData(const Data<T>& d, const std::string& p, const bool normalize = true) {

    std::string file_name;

    /**
     * Raw image
     */
    file_name = p + "image.png";
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::Mat out;
    if (normalize) {
        cv::normalize(d.image, out, 0, 255, cv::NORM_MINMAX);
    } else {
        out = d.image;
    }
    cv::imwrite(file_name, out);

    /**
     * Cloud
     */
    file_name = p + "cloud.pcd";
    LOG(INFO) << "Writing cloud to '" << file_name << "'.";
    pcl::io::savePCDFile<T>(file_name, *(d.cloud));
}

template <typename T>
void exportDepthImage(const Data<T>& d, const std::shared_ptr<CameraModel>& cam,
                      const std::string& p, const bool normalize = true) {

    int rows, cols;
    cam->getImageSize(cols, rows);
    cv::Mat img_depth{cv::Mat::zeros(rows, cols, CV_32FC1)};
    const Eigen::Matrix3Xd pts_3d{
        d.transform * d.cloud->getMatrixXfMap().template topRows<3>().template cast<double>()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    const std::vector<bool> in_img{cam->getImagePoints(pts_3d, img_pts_raw)};
    for (size_t c = 0; c < in_img.size(); c++) {
        const int row = img_pts_raw(1, c);
        const int col = img_pts_raw(0, c);
        Eigen::Vector3d support, direction;
        cam->getViewingRay(Eigen::Vector2d(img_pts_raw(0, c), img_pts_raw(1, c)), support,
                           direction);
        const Eigen::Hyperplane<double, 3> plane(direction, pts_3d.col(c));
        img_depth.at<float>(row, col) =
            (Eigen::ParametrizedLine<double, 3>(support, direction).intersectionPoint(plane) -
             support)
                .norm();
    }
    cv::Mat out;
    if (normalize) {
        cv::normalize(img_depth, out, 0, 255, cv::NORM_MINMAX);
    } else {
        out = img_depth;
    }
    const std::string file_name{p + "depth_est.png"};
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, out);
}

inline void exportGradientImage(const cv::Mat& img, const std::string& p,
                                const bool normalize = true) {
    cv::Mat out;
    if (normalize) {
        cv::normalize(gradientSobel(img), out, 0, 255, cv::NORM_MINMAX);
    } else {
        out = gradientSobel(img, false);
    }
    const std::string file_name{p + "gradient.png"};
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, out);
}
}
