#pragma once

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>

namespace mrf {

template <typename T, typename U>
const typename pcl::PointCloud<U>::Ptr estimateNormals(
    const typename pcl::PointCloud<T>::ConstPtr& in, const double& radius) {
    using namespace pcl;
    const typename PointCloud<U>::Ptr out{new PointCloud<U>};
    NormalEstimationOMP<T, U> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(in);
    ne.compute(*out);
    concatenateFields(*out, *in, *out);
    return out;
}

template <typename T>
const typename pcl::PointCloud<T>::Ptr transform(const typename pcl::PointCloud<T>::ConstPtr& in,
                                                 const Eigen::Affine3d& tf) {
    using namespace pcl;
    const typename PointCloud<T>::Ptr out{new PointCloud<T>};
//    if (!pcl::traits::has_normal<T>::value) {
//        transformPointCloudWithNormals(*in, *out, tf);
//    } else {
//        transformPointCloud(*in, *out, tf);
//    }
    transformPointCloud(*in, *out, tf);
    return out;
}

const typename pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformWithNormals(const typename pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr& in,
                                                 const Eigen::Affine3d& tf) {
    using namespace pcl;
    const typename PointCloud<pcl::PointXYZINormal>::Ptr out{new PointCloud<pcl::PointXYZINormal>};
    transformPointCloudWithNormals(*in, *out, tf);
    return out;
}

}
