#pragma once

#include <ostream>
#include <stddef.h>

namespace mrf {
struct Pixel {
    inline Pixel(const size_t& r, const size_t& c) : row{r}, col{c} {};
    size_t row;
    size_t col;

    inline friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "row: " << p.row << ", col:" << p.col;
    }
};

struct PixelLess {
    inline bool operator()(const Pixel& lhs, const Pixel& rhs) {
        if (lhs.row < rhs.row)
            return true;
        else if (lhs.row > rhs.row)
            return false;
        else if (lhs.col < rhs.col)
            return true;
        return false;
    }
};
}
