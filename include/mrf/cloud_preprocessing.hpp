#pragma once

#include <glog/logging.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>

namespace mrf {

template <typename T, typename U>
const typename pcl::PointCloud<U>::Ptr estimateNormals(
    const typename pcl::PointCloud<T>::ConstPtr& in,
    const double& radius,
    const bool remove_invalid = true) {
    using namespace pcl;
    const typename PointCloud<U>::Ptr out{new PointCloud<U>};
    NormalEstimationOMP<T, pcl::Normal> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(in);
    pcl::PointCloud<pcl::Normal> cl_normals;
    ne.compute(cl_normals);
    concatenateFields(*in, cl_normals, *out);

    PointCloud<U> tmp;
    size_t invalid_points{0};
    if (remove_invalid) {
        for (auto& p : out->points) {
            if (!pcl_isfinite(p.normal_x) || !pcl_isfinite(p.normal_y) ||
                !pcl_isfinite(p.normal_z)) {
                p.getNormalVector3fMap() = -p.getVector3fMap().normalized();
                invalid_points++;
            }
        }
        LOG(INFO) << "Detected " << invalid_points << " invalid normal points";
    }
    if (!invalid_points)
        out->height = in->size() / in->width;
    else
        out->width = in->size();

    return out;
}
}
