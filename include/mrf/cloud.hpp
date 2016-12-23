#pragma once

#include "point.hpp"

namespace mrf {

template <typename T>
class Cloud {

    inline Cloud(){};
    inline Cloud(const size_t& rows, const size_t& cols) : width{cols}, height{rows} {
        resize(width * height);
    }

    inline void push_back(const Point<T>& p) {
        points.push_back(p);
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

    inline Point<T>& at(const size_t& col, const size_t& row) {
        return (points.at(row * width + col));
    }
    inline const Point<T>& at(const size_t& col, const size_t& row) const {
        return (points.at(row * width + col));
    }

    inline const Eigen::Map<const Eigen::MatrixX<T>, Eigen::Aligned, Eigen::OuterStride<>>
    getMatrixXfMap(const size_t& dim, const size_t& stride, const size_t& offset) const {
        using namespace Eigen;

        if (MatrixX<T>::Flags & RowMajorBit)
            return (Map<const MatrixX<T>, Aligned, OuterStride<>>(
                reinterpret_cast<float*>(const_cast<Point<T>*>(&points[0])) + offset, points.size(),
                dim, OuterStride<>(stride)));
        else
            return (Map<const MatrixX<T>, Aligned, OuterStride<>>(
                reinterpret_cast<float*>(const_cast<Point<T>*>(&points[0])) + offset, dim,
                points.size(), OuterStride<>(stride)));
    }

    inline Eigen::Map<Eigen::MatrixX<T>, Eigen::Aligned, Eigen::OuterStride<>> getMatrixXfMap() {
        return (getMatrixXfMap(sizeof(Point<T>) / sizeof(T), sizeof(Point<T>) / sizeof(T), 0));
    }

private:
    std::vector<Point<T>> points;

    size_t width{0};
    size_t height{0};
};
}
