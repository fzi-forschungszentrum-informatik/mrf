#pragma once

#include <Eigen/Geometry>
#include <camera_models/camera_model.h>
#include <glog/logging.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>

#include "data.hpp"
#include "image_preprocessing.hpp"
#include "pixel.hpp"
#include "result_info.hpp"

namespace mrf {

inline cv::Mat createOutput(const cv::Mat& in, const bool normalize = true) {
    using TargetT = uint8_t;
    cv::Mat out;
    if (normalize) {
        cv::normalize(in, out, 0, std::numeric_limits<TargetT>::max(), cv::NORM_MINMAX);
        out.convertTo(out, cv::DataType<TargetT>::type);
    } else
        in.convertTo(out, cv::DataType<TargetT>::type);
    return out;
}

inline void exportImage(const cv::Mat& img,
                        const std::string& p,
                        const bool normalize = true,
                        const bool invert = false,
                        const bool apply_color_map = false) {
    const std::string file_name{p + "image.png"};
    LOG(INFO) << "Writing image to '" << file_name << "'.";
    cv::Mat out{createOutput(img, normalize)};
    if (invert)
        cv::bitwise_not(out, out);

    if (apply_color_map)
        cv::applyColorMap(out, out, cv::COLORMAP_HOT);

    cv::imwrite(file_name, out);
}

template <typename T>
inline void exportCloud(const typename pcl::PointCloud<T>::ConstPtr& cl, const std::string& p) {
    if (!cl->empty()) {
        const std::string file_name = p + "cloud.pcd";
        LOG(INFO) << "Writing cloud to '" << file_name << "'.";
        pcl::io::savePCDFile<T>(file_name, *cl, true);
    }
}

template <typename T>
void exportData(const Data<T>& d, const std::string& p, const bool normalize = true) {
    exportImage(createOutput(d.image, normalize), p);
    exportCloud<T>(d.cloud, p);
}

template <typename T>
void exportDepthImage(const Data<T>& d,
                      const std::shared_ptr<CameraModel>& cam,
                      const std::string& p,
                      const bool normalize = true,
                      const bool invert = false,
                      const bool apply_color_map = false) {
    int rows, cols;
    cam->getImageSize(cols, rows);
    cv::Mat img_depth{cv::Mat::zeros(rows, cols, cv::DataType<double>::type)};
    const Eigen::Matrix3Xd pts_3d{
        d.transform * d.cloud->getMatrixXfMap().template topRows<3>().template cast<double>()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    const std::vector<bool> in_front{cam->getImagePoints(pts_3d, img_pts_raw)};
    for (size_t c = 0; c < in_front.size(); c++) {
        const Pixel p{img_pts_raw(0, c), img_pts_raw(1, c)};
        if (!p.inImage(rows, cols) || !in_front[c])
            continue;
        Eigen::Vector3d support, direction;
        cam->getViewingRay(Eigen::Vector2d(p.col, p.row), support, direction);
        const Eigen::Hyperplane<double, 3> plane(direction, pts_3d.col(c));
        img_depth.at<double>(p.row, p.col) =
            (Eigen::ParametrizedLine<double, 3>(support, direction).intersectionPoint(plane) -
             support)
                .norm();
    }
    exportImage(
        createOutput(img_depth, normalize), p + "depth_", normalize, invert, apply_color_map);
}

template <typename T>
void exportOverlay(const Data<T>& d,
                   const std::shared_ptr<CameraModel>& cam,
                   const std::string& p,
                   const bool normalize = true) {
    int rows, cols;
    cam->getImageSize(cols, rows);
    cv::Mat img{d.image};
    const int channels{img.channels()};
    const Eigen::Matrix3Xd pts_3d{
        d.transform * d.cloud->getMatrixXfMap().template topRows<3>().template cast<double>()};
    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    const std::vector<bool> in_front{cam->getImagePoints(pts_3d, img_pts_raw)};
    for (size_t c = 0; c < in_front.size(); c++) {
        const Pixel p{img_pts_raw(0, c), img_pts_raw(1, c)};
        if (p.inImage(rows, cols) && in_front[c])
            for (int d = 0; d < channels; d++)
                img.ptr<float>(p.row)[p.col * channels + d] = 0;
    }
    exportImage(createOutput(img, normalize), p + "depth_", normalize);
}


void exportResultInfo(const ResultInfo& info, const std::string& p) {
    if (info.has_covariance_depth) {
        cv::Mat out;
        cv::eigen2cv(info.covariance_depth, out);
        exportImage(createOutput(out), p + "covariance_depth_");
    }
    if (info.has_weights) {
        cv::Mat out;
        cv::eigen2cv(info.weights, out);
        exportImage(createOutput(out), p + "weights_", true, false, true);
    }
}
}
