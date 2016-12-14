#pragma once

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>

namespace mrf {

template <typename T, typename U>
const pcl::PointCloud<T>::Ptr estimateNormals(const pcl::PointCloud<T>::ConstPtr& in,
                                              const double& radius) {
    using namespace pcl;
    const PointCloud<U>::Ptr out{new PointCloud<U>};
    NormalEstimationOMP<T, U> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(in);
    ne.compute(out);
    return out;
}

template <typename T>
const pcl::PointCloud<T>::Ptr transform(const pcl::PointCloud<T>::ConstPtr& in,
                                        const Eigen::Affine3d& tf) {
    using namespace pcl;
    const PointCloud<T>::Ptr out{new PointCloud<T>};
    transformPointCloud(*in, *out, tf);
    return out;
}
}
