#pragma once

#include <pcl/point_types.h>

#include "eigen.hpp"
#include "point_cloud.hpp"
#include "point.hpp"

namespace pcl_ceres {

template <typename PointT, typename U = double>
const typename PointCloud<PointT>::Ptr transform(const typename PointCloud<PointT>::ConstPtr in,
                                            const Eigen::Affine3<U>& tf) {
    const typename PointCloud<PointT>::Ptr out{PointCloud<PointT>::create(*in)};
    transform<PointT, U>(tf, out);
    return out;
}
template <typename PointT, typename U = double>
void transform(const Eigen::Affine3<U>& tf, const typename PointCloud<PointT>::Ptr cl) {
    using T = typename PointT::ScalarT;
    const Eigen::Matrix<T, 3, 3> rotation{tf.rotation().template cast<T>()};
    for (auto& p : cl->points) {
        p.position = tf.template cast<T>() * p.position;
        p.normal = rotation * p.normal;
    }
}
}
