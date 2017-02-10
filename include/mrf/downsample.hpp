#pragma once

#include <ctime>
#include <pcl/point_cloud.h>

namespace mrf {

template <typename T>
const typename pcl::PointCloud<T>::Ptr downsampleEquidistant(
    const typename pcl::PointCloud<T>::ConstPtr& in,
    const size_t& skip_cols,
    const size_t& skip_rows = 1) {
    const typename pcl::PointCloud<T>::Ptr out{new typename pcl::PointCloud<T>};
    out->reserve(in->size());
    if (in->height > 1) { ///< Organized
        for (size_t r = 0; r < in->height; r += skip_rows) {
            for (size_t c = 0; c < in->width; c += skip_cols) {
                out->push_back(in->at(c, r));
            }
        }
        out->height = in->height / skip_rows;
        out->width = in->width / skip_cols;
    } else { ///< Unorganized
        for (size_t c = 0; c < in->size(); c += skip_cols) {
            out->push_back(in->points[c]);
        }
    }
    return out;
}

template <typename T>
const typename pcl::PointCloud<T>::Ptr downsampleRandom(
    const typename pcl::PointCloud<T>::ConstPtr& in, const double& random_rate) {
    const typename pcl::PointCloud<T>::Ptr out{new typename pcl::PointCloud<T>};
    const size_t num_points{static_cast<size_t>(random_rate * in->size())};
    out->reserve(num_points);

    std::srand(std::time(0));
    for (size_t c = 0; c < num_points; c++) {
        out->push_back(in->points[(std::rand() % in->size())]);
    }
    return out;
}
}
