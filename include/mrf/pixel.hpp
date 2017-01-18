#pragma once

#include <ostream>
#include <stddef.h>

namespace mrf {
struct Pixel {
    inline Pixel(const double& x_, const double& y_, const double& val_ = 0)
            : x{x_}, y{y_}, val{val_}, row(y_ + 0.5), col(x_ + 0.5){};

    inline friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "x: " << p.x << ", y: " << p.y << ", val: " << p.val << ", row: " << p.row
           << ", col:" << p.col;
        return os;
    }

    inline bool inImage(const int& rows, const int& cols) const {
        return row > 0 && row < rows && col > 0 && col < cols;
    }

    const int row, col;
    const double x, y;
    double val;
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
