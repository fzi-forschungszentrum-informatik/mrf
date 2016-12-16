#include "neighbors.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p, const size_t& rows, const size_t& cols,
                                const Parameters::Neighborhood& mode) {

    std::vector<Pixel> neighbors;
    neighbors.reserve(static_cast<size_t>(mode));

    if (p.row > 0) {
        neighbors.emplace_back(p.row - 1, p.col);
    }
    if (p.row < rows - 1) {
        neighbors.emplace_back(p.row + 1, p.col);
    }
    if (p.col > 0) {
        neighbors.emplace_back(p.row, p.col - 1);
    }
    if (p.col < cols - 1) {
        neighbors.emplace_back(p.row, p.col + 1);
    }

    if (mode == Parameters::Neighborhood::eight) {
        if (p.row > 0 && p.row < rows - 1) {
            neighbors.emplace_back(p.row + 1, p.col - 1);
        }
        if (p.row < rows - 1 && p.row < rows - 1) {
            neighbors.emplace_back(p.row + 1, p.col + 1);
        }
        if (p.col > 0 && p.row > 0) {
            neighbors.emplace_back(p.row - 1, p.col - 1);
        }
        if (p.col < cols - 1 && p.row > 0) {
            neighbors.emplace_back(p.row - 1, p.col + 1);
        }
    }

    return neighbors;
}
}
