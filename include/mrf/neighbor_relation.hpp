#pragma once

namespace mrf {

//! @brief Definition of an eight neighbor neighboorhood
enum class NeighborRelation {
    left,
    top,
    right,
    bottom,
    bottom_left,
    top_left,
    top_right,
    bottom_right
};
}
