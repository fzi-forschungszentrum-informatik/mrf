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

inline cv::Mat createOutput(const cv::Mat& in, const bool normalize = true) {
    cv::Mat out;
    if (normalize) {
        cv::normalize(in, out, 0, std::numeric_limits<uint16_t>::max(), cv::NORM_MINMAX);
        out.convertTo(out, cv::DataType<uint16_t>::type);
    } else {
        in.convertTo(out, cv::DataType<uint16_t>::type);
    }
    return out;
}

template <typename T>
void exportData(const Data<T>& d, const std::string& p, const bool normalize = true) {

    std::string file_name;

    /**
     * Raw image
     */
    file_name = p + "image.png";
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, createOutput(d.image, normalize));

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
    cv::Mat img_depth{cv::Mat::zeros(rows, cols, cv::DataType<double>::type)};
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
        img_depth.at<double>(row, col) =
            (Eigen::ParametrizedLine<double, 3>(support, direction).intersectionPoint(plane) -
             support)
                .norm();
    }

    const std::string file_name{p + "depth.png"};
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, createOutput(img_depth, normalize));
}

inline void exportImage(const cv::Mat& img, const std::string& p, const bool normalize = true) {
    const std::string file_name{p + "image.png"};
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, createOutput(img, normalize));
}

inline void exportGradientImage(const cv::Mat& img, const std::string& p,
                                const bool normalize = true) {
    const std::string file_name{p + "gradient.png"};
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::imwrite(file_name, createOutput(edge(img), normalize));
}
}
