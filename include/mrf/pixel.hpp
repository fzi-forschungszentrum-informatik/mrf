#pragma once

#include <ostream>
#include <cmath>
#include <Eigen/Eigen>

namespace mrf {
struct Pixel {
    inline Pixel(const double& x_,
                 const double& y_,
                 const Eigen::VectorXf& val_ = Eigen::VectorXf::Zero(1))
            : x{x_}, y{y_}, val{val_}, row(std::round(y_)), col(std::round(x_)){};

    inline friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "x: " << p.x << ", y: " << p.y << ", val: " << p.val.transpose() << ", row: " << p.row
           << ", col:" << p.col;
        return os;
    }

    inline bool inImage(const int& rows, const int& cols) const {
        return row > 0 && row < rows && col > 0 && col < cols;
    }

    const int row, col;
    const double x, y;
    Eigen::VectorXf val;
};

struct PixelLess {
    inline bool operator()(const Pixel& lhs, const Pixel& rhs) const {
        if (lhs.col < rhs.col)
            return true;
        else if (lhs.col > rhs.col)
            return false;
        else if (lhs.row < rhs.row)
            return true;
        return false;
    }
};
}
