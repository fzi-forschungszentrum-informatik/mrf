#pragma once

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p, const size_t& rows, const size_t& cols,
                                const Parameters::Neighborhood& mode);
}
