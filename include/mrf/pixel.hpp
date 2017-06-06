#pragma once

#include <cmath>
#include <ostream>
#include <Eigen/Eigen>

namespace mrf {
/** @brief The Pixel struct that stores image color information for one pixel. */
struct Pixel {
    inline Pixel(const double& x_, const double& y_, const Eigen::VectorXf& val_ = Eigen::VectorXf::Zero(1))
            : x{x_}, y{y_}, val{val_}, row(std::round(y_)), col(std::round(x_)){};

    inline friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "x: " << p.x << ", y: " << p.y << ", val: " << p.val.transpose() << ", row: " << p.row
           << ", col:" << p.col;
        return os;
    }

    inline bool operator==(const Pixel& other) const {
        return row == other.row && col == other.col;
    }

    inline bool inImage(const int& row_max, const int& col_max, const int& row_min = 0, const int& col_min = 0) const {
        return row_min <= row && row < row_max && col_min <= col && col < col_max;
    }

    double x, y;         //!< Position of the pixel
    Eigen::VectorXf val; //!< Color information
    int row, col;        //!< Row and column derived from x,y
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
