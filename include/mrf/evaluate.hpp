#pragma once

#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "data.hpp"
#include "pixel.hpp"
#include "quality.hpp"

namespace mrf {

inline double median(std::vector<double>& in) {
    auto median = begin(in);
    std::advance(median, in.size() / 2);
    std::nth_element(begin(in), median, end(in));
    return *median;
}

template <typename T, typename U>
Quality evaluate(const Data<T>& ref, const Data<U>& est, const std::shared_ptr<CameraModel> cam) {

    Quality q;

    using namespace Eigen;
    int rows, cols;
    cam->getImageSize(cols, rows);
    q.depth_error = cv::Mat::zeros(rows, cols, cv::DataType<double>::type);

    const Matrix3Xd refs_3d{est.transform *
                      ref.cloud->getMatrixXfMap().template topRows<3>().template cast<double>()};
    Matrix2Xd refs_img(3, refs_3d.cols());
    const std::vector<bool> in_front{cam->getImagePoints(refs_3d, refs_img)};
    std::vector<double> depth_errors, depth_errors_abs;
    depth_errors.reserve(in_front.size());
    depth_errors_abs.reserve(in_front.size());
    for (size_t c = 0; c < in_front.size(); c++) {
        const T& p_ref{ref.cloud->points[c]};
        const Vector3d& ref_3d{refs_3d.col(c)};
        const Vector2d& ref_img{refs_img.col(c)};

        if (!std::isfinite(ref_3d.x()) || !std::isfinite(ref_3d.y()) ||
            !std::isfinite(ref_3d.z()) || !std::isfinite(ref_img.x()) ||
            !std::isfinite(ref_img.y()))
            LOG(INFO) << "NAN point: " << ref_3d.transpose() << ", " << ref_img.transpose();

        const Pixel p(ref_img.x(), ref_img.y());
        if (ref_3d.z() < 0 || !in_front[c] || (p.col < 0) || (p.col >= cols) || (p.row < 0) ||
            (p.row >= rows)) {
            //            LOG(INFO) << "Not in image: " << ref_3d.transpose() << ", " <<
            //            ref_img.transpose();
            continue;
        }
        q.ref_distances_evaluated++;

        /**
         * Determine depth
         */
        Vector3d support, direction;
        cam->getViewingRay(ref_img, support, direction);
        const double distance_ref{
            (ParametrizedLine<double, 3>(support, direction)
                 .intersectionPoint(Hyperplane<double, 3>(direction, ref_3d)) -
             support)
                .norm()};

        /**
         * Depth errors
         */
        const U& p_est{est.cloud->at(p.col, p.row)};
        const double distance_error{est.image.template at<double>(p.row, p.col) - distance_ref};
        if (!std::isfinite(distance_error) || !std::isfinite(p_est.x) || !std::isfinite(p_est.y) ||
            !std::isfinite(p_est.z)) {
            q.ref_distances_evaluated--;
            continue;
        }
        const double distance_error_abs{std::abs(distance_error)};
        q.depth_error.at<double>(p.row, p.col) = distance_error;
        q.depth_error_mean += distance_error;
        q.depth_error_mean_abs += distance_error_abs;
        depth_errors.push_back(distance_error);
        depth_errors_abs.push_back(distance_error_abs);
        q.depth_error_rms += std::pow(distance_error, 2);

        /**
         * Normal errors
         */
        if (!std::isfinite(p_ref.normal_x) || !std::isfinite(p_ref.normal_y) ||
            !std::isfinite(p_ref.normal_z))
            continue;
        q.ref_normals_evaluated++;
        const Eigen::Vector3d n_ref_eigen{p_ref.getNormalVector3fMap().template cast<double>()};

        const Eigen::Vector3d n_est{p_est.getNormalVector3fMap().template cast<double>()};
        const Eigen::Vector3d delta{n_est - n_ref_eigen};
        const double dot_product{n_est.dot(n_ref_eigen)};
        q.normal_error_mean = q.normal_error_mean + delta;
        q.normal_error_mean_abs += delta.array().abs().matrix();
        q.normal_dot_product_mean += dot_product;
        q.normal_dot_product_mean_abs += std::abs(dot_product);
    }
    q.depth_error_mean /= q.ref_distances_evaluated;
    q.depth_error_mean_abs /= q.ref_distances_evaluated;
    q.depth_error_median = median(depth_errors);
    q.depth_error_median_abs = median(depth_errors_abs);
    q.depth_error_rms = std::sqrt(q.depth_error_rms / q.ref_distances_evaluated);
    q.normal_error_mean /= q.ref_normals_evaluated;
    q.normal_error_mean_abs /= q.ref_normals_evaluated;
    q.normal_dot_product_mean /= q.ref_normals_evaluated;
    q.normal_dot_product_mean_abs /= q.ref_normals_evaluated;

    return q;
}
}
