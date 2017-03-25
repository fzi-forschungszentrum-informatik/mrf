#pragma once

#include <memory>
#include "eigen.hpp"

namespace pcl_ceres {

template <typename PointT>
class PointCloud {

    using T = typename PointT::ScalarT;

public:
    using Ptr = std::shared_ptr<PointCloud<PointT>>;
    using ConstPtr = std::shared_ptr<const PointCloud<PointT>>;

    inline PointCloud(){};
    inline PointCloud(const size_t& rows, const size_t& cols) : width{cols}, height{rows} {
        resize(width * height);
    }
    inline PointCloud(const PointCloud<PointT>& cl) {
        *this = cl;
    }

    inline void push_back(const PointT& p) {
        points.push_back(p);
        width = points.size();
        height = 1;
    }
    inline void emplace_back(PointT&& p) {
        points.emplace_back(p);
        width = points.size();
        height = 1;
    }

    inline size_t size() const {
        return points.size();
    }

    inline void resize(const size_t& n) {
        points.resize(n);
        if (width * height != n) {
            width = n;
            height = 1;
        }
    }

    inline PointT& at(const size_t& col, const size_t& row) {
        return (points.at(row * width + col));
    }
    inline const PointT& at(const size_t& col, const size_t& row) const {
        return (points.at(row * width + col));
    }

    inline Eigen::Matrix3X<T> getMatrixPoints() const {
        Eigen::Matrix<T, 3, Eigen::Dynamic> m(3, points.size());
        for (size_t c = 0; c < points.size(); c++) {
            m.col(c) = points[c].position;
        }
        return m;
    }

    static inline Ptr create() {
        return std::make_shared<PointCloud<PointT>>();
    }
    static inline Ptr create(const size_t& rows, const size_t& cols) {
        return std::make_shared<PointCloud<PointT>>(rows, cols);
    }
    static inline Ptr create(const PointCloud<PointT>& cl) {
        return std::make_shared<PointCloud<PointT>>(cl);
    }

    std::vector<PointT> points;
    size_t width{0};
    size_t height{0};
};
}
