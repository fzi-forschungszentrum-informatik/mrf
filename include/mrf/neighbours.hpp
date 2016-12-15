#pragma once


#include "parameters.hpp"

namespace mrf {

enum class NeighbourCase { top_bottom, left_right, top_left_right, bottom_left_right };


int neighPos(const int p, const int pnext, const NeighbourCase& nc, const int width,
             const int dim) {
    if (((abs((p % width) - (pnext % width)) > 1) || pnext < 0) &&
        nc != NeighbourCase::top_bottom) {
        /*
         * Criteria for left right border pass
         */
        return -1;
    }
    if (((floor(p / width) == 0) && (pnext < 0)) &&
        (nc == NeighbourCase::top_bottom || nc == NeighbourCase::top_left_right)) {
        /*
         * Criteria for top pass
         */
        return -1;
    }
    if ((pnext >= static_cast<int>(dim)) &&
        (nc == NeighbourCase::top_bottom || nc == NeighbourCase::bottom_left_right)) {
        /*
         * Criteria for bottom pass
         */
        return -1;
    }
    return pnext;
}

std::vector<int> getNeighbours(const int p, const Parameters& params,
                         const int width, const int dim) {
	std::vector neigh_pos;
    /**
     * Resize vectors
     */
    if (params.neighborhood == 8) {
        neigh_pos = {-1, -1, -1, -1, -1, -1, -1, -1};

    } else {
        neigh_pos = {-1, -1, -1, -1};

    }

    /**
     * fill neigh positions
     */
    int i{0};
    for (size_t j = 1; j < 3; j++) {
        neigh_pos(i) = neighPos(p, p + pow(-1, j), NeighbourCase::left_right, width, dim);
        neigh_pos(i + 1) =
            neighPos(p, p + pow(-1, j) * width, NeighbourCase::top_bottom, width, dim);
        if (params.neighborhood == 8) {
            neigh_pos(i + 2) =
                neighPos(p, p + pow(-1, j) + width, NeighbourCase::top_left_right, width, dim);
            neigh_pos(i + 3) =
                neighPos(p, p + pow(-1, j) - width, NeighbourCase::bottom_left_right, width, dim);
            i += 2;
        }
        i += 2;
    }
return neigh_pos;

}
}
