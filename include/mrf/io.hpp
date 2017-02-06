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
        cv::applyColorMap(out, out, cv::COLORMAP_BONE);

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

inline double interpolate(double val, double y0, double x0, double y1, double x1) {
    return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}
double base(double val) {
    if (val <= -0.75)
        return 0;
    else if (val <= -0.25)
        return interpolate(val, 0.0, -0.75, 1.0, -0.25);
    else if (val <= 0.25)
        return 1.0;
    else if (val <= 0.75)
        return interpolate(val, 1.0, 0.25, 0.0, 0.75);
    else
        return 0.0;
}
inline double red(double gray) {
    return base(gray - 0.5);
}
inline double green(double gray) {
    return base(gray);
}
inline double blue(double gray) {
    return base(gray + 0.5);
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

    const double d_max{35};
    const double d_min{5};
    for (size_t c = 0; c < in_front.size(); c++) {
        const Pixel p{img_pts_raw(0, c), img_pts_raw(1, c)};
        if (p.inImage(rows, cols) && in_front[c]) {
            double d{pts_3d.col(c).norm()};
            d = std::min(d, d_max);
            d = std::max(d, d_min);
            const double d_rel{2 * ((d - d_min) / (d_max - d_min) - 0.5)};
            cv::circle(
                img,
                cv::Point(p.col, p.row),
                2,
                cv::Scalar(255 * (1 - blue(d_rel)), 255 * green(d_rel), 255 * (1 - red(d_rel))),
                -1);
        }
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
