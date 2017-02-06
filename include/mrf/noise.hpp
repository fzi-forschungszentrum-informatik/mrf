#pragma once

#include <pcl/common/transforms.h>
#include <random>
#include <chrono>

namespace mrf {

template <typename T>
const typename pcl::PointCloud<T>::Ptr addDepthNoise(
    const typename pcl::PointCloud<T>::ConstPtr& in, const float& sigma) {
    const typename pcl::PointCloud<T>::Ptr out{new typename pcl::PointCloud<T>};
    pcl::copyPointCloud(*in, *out);

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0, sigma);
    for (auto& p : out->points)
        p.getVector3fMap() = distribution(generator) * p.getVector3fMap();
    return out;
}

template <typename T>
const typename pcl::PointCloud<T>::Ptr addCalibrationNoise(
    const typename pcl::PointCloud<T>::ConstPtr& in,
    const float& sigma_trans,
    const float& sigma_rot) {
    using namespace Eigen;
    std::default_random_engine generator;
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::normal_distribution<float> distribution_rot(0, sigma_rot);
    std::normal_distribution<float> distribution_trans(0, sigma_trans);
    const float err_rotation{distribution_rot(generator)};
    const float err_translation{distribution_trans(generator)};
    Matrix3f m;
    m = AngleAxisf(err_rotation * M_PI / 180, Vector3f::UnitX()) *
        AngleAxisf(err_rotation * M_PI, Vector3f::UnitY()) *
        AngleAxisf(err_rotation * M_PI, Vector3f::UnitZ());
    Affine3f transform(m);
    transform.translation() = err_translation * Vector3f::Ones();

    const typename pcl::PointCloud<T>::Ptr out{new typename pcl::PointCloud<T>};
    pcl::transformPointCloud(*in, *out, transform);
    return out;
}
}
