#pragma once

#include "point.hpp"

namespace mrf {

template <typename PointT>
class Cloud {

    using T = typename PointT::ScalarT;

public:
    using Ptr = std::shared_ptr<Cloud<PointT>>;
    using ConstPtr = std::shared_ptr<const Cloud<PointT>>;

    inline Cloud(){};
    inline Cloud(const size_t& rows, const size_t& cols) : width{cols}, height{rows} {
        resize(width * height);
    }
    inline Cloud(const Cloud<PointT>& cl) {
        *this = cl;
    }

    inline void push_back(const Point<PointT>& p) {
        points.push_back(p);
        width = points.size();
        height = 1;
    }
    inline void emplace_back(Point<PointT>&& p) {
        points.emplace_back(p);
        width = points.size();
        height = 1;
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
        const size_t dim{width * height};
        Eigen::Matrix<T, 3, Eigen::Dynamic> m(3, dim);
        for (size_t c = 0; c < dim; c++) {
            m.col(c) = points[c].position;
        }
        return m;
    }
    inline Eigen::Matrix3X<T> getMatrixNormals() const {
        const size_t dim{width * height};
        Eigen::Matrix<T, 3, Eigen::Dynamic> m(3, dim);
        for (size_t c = 0; c < dim; c++) {
            m.col(c) = points[c].normal;
        }
        return m;
    }

    static inline Ptr create() {
        return std::make_shared<Cloud<PointT>>();
    }
    static inline Ptr create(const size_t& rows, const size_t& cols) {
        return std::make_shared<Cloud<PointT>>(rows, cols);
    }
    static inline Ptr create(const Cloud<PointT>& cl) {
        return std::make_shared<Cloud<PointT>>(cl);
    }

    std::vector<PointT> points;
    size_t width{0};
    size_t height{0};
};

template <typename PointT, typename U>
const typename Cloud<PointT>::Ptr transform(const typename Cloud<PointT>::ConstPtr in,
                                            const Eigen::Affine3<U>& tf) {
    const typename Cloud<PointT>::Ptr out{Cloud<PointT>::create(*in)};
    transform<PointT, U>(tf, out);
    return out;
}
template <typename PointT, typename U>
void transform(const Eigen::Affine3<U>& tf, const typename Cloud<PointT>::Ptr cl) {
    using T = typename PointT::ScalarT;
    const Eigen::Matrix<T, 3, 3> rotation{tf.rotation().template cast<T>()};
    for (auto& p : cl->points) {
        p.position = tf.template cast<T>() * p.position;
        p.normal = rotation * p.normal;
    }
}
}
