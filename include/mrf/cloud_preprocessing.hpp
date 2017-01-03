#pragma once

#include <pcl/features/normal_3d_omp.h>

namespace mrf {

template <typename T, typename U>
const typename pcl::PointCloud<U>::Ptr estimateNormals(
    const typename pcl::PointCloud<T>::ConstPtr& in, const double& radius) {
    using namespace pcl;
    const typename PointCloud<U>::Ptr out{new PointCloud<U>};
    NormalEstimationOMP<T, pcl::Normal> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(in);
    pcl::PointCloud<pcl::Normal> cl_normals;
    ne.compute(cl_normals);
    concatenateFields(*in, cl_normals, *out);

    for (auto const& p : out->points) {
    	LOG(INFO) << "Point: " << p.getVector3fMap().transpose() << ", normal: " << p.getNormalVector3fMap().transpose();
    }

    return out;
}
}
