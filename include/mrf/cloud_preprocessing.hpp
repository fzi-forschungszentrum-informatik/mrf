#pragma once

#include <glog/logging.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>

namespace mrf {

template <typename T, typename U>
const typename pcl::PointCloud<U>::Ptr estimateNormals(
    const typename pcl::PointCloud<T>::ConstPtr& in, const double& radius,
    const bool remove_invalid = true) {
    using namespace pcl;
    const typename PointCloud<U>::Ptr out{new PointCloud<U>};
    NormalEstimationOMP<T, pcl::Normal> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(in);
    pcl::PointCloud<pcl::Normal> cl_normals;
    ne.compute(cl_normals);
    concatenateFields(*in, cl_normals, *out);

    if (remove_invalid) {
        std::vector<int> indices;
        pcl::removeNaNNormalsFromPointCloud(*out, *out, indices);
        const size_t points_removed{in->size() - out->size()};
        LOG(INFO) << "Removed " << points_removed << " invalid normal points";
        if (!points_removed) {
            out->height = in->height;
            out->width = in->width;
        }
    }
    return out;
}
}
