#pragma once

#include <ostream>
#include <stddef.h>

namespace mrf {
struct Pixel {
    inline Pixel(const double& x_, const double& y_, const double& val_ = 0)
            : x{x_}, y{y_}, val{val_}, row{static_cast<int>(y_)}, col{static_cast<int>(x_)} {};

    inline friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "x: " << p.x << ", y: " << p.y << ", val: " << p.val << ", row: " << p.row
           << ", col:" << p.col;
        return os;
    }

    const int row, col;
    const double x, y;
    double val;
};

struct PixelLess {
    inline bool operator()(const Pixel& lhs, const Pixel& rhs) {
        if (lhs.x < rhs.x)
            return true;
        else if (lhs.x > rhs.x)
            return false;
        else if (lhs.y < rhs.y)
            return true;
        return false;
    }
};
}
