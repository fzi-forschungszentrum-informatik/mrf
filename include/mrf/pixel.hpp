#pragma once

#include <cmath>
#include <ostream>
#include <stddef.h>

namespace mrf {
struct Pixel {
    inline Pixel(const double& x_, const double& y_, const double& val_ = 0)
            : x{x_}, y{y_}, val{val_}, row(std::round(y_)), col(std::round(x_)){};

    inline friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "x: " << p.x << ", y: " << p.y << ", val: " << p.val << ", row: " << p.row
           << ", col:" << p.col;
        return os;
    }

    inline bool operator==(const Pixel& other) const {
        return row == other.row && col == other.col;
    }

    int row, col;
    double x, y;
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
