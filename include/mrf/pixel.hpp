#pragma once

#include <stddef.h>

namespace mrf {
struct Pixel {
    inline Pixel(const size_t& r, const size_t& c) : row{r}, col{c} {};
    size_t row;
    size_t col;
};
}
