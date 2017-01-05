#pragma once

#include <random>

namespace mrf {

template <typename T>
const typename pcl::PointCloud<T>::Ptr addNoise(const typename pcl::PointCloud<T>::ConstPtr& in,
                                                const float& sigma_x, const float& sigma_y,
                                                const float& sigma_z) {
    const typename pcl::PointCloud<T>::Ptr out{new typename pcl::PointCloud<T>};
    pcl::copyPointCloud(*in, *out);

    std::default_random_engine generator;
    std::normal_distribution<float> distribution_x(0, sigma_x);
    std::normal_distribution<float> distribution_y(0, sigma_y);
    std::normal_distribution<float> distribution_z(0, sigma_z);
    for (auto& p : out->points) {
        p.x += distribution_x(generator);
        p.y += distribution_y(generator);
        p.z += distribution_z(generator);
    }
    return out;
}
}
