#pragma once

#include <map>
#include <utility>
#include <vector>
#include <Eigen/Geometry>

#include "neighbor_relation.hpp"
#include "pixel.hpp"

namespace mrf {

struct OptimizationData {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    std::map<Pixel, Ray, PixelLess> rays;
    Pixel ref{Pixel(0, 0)};
    std::vector<std::pair<Pixel, Pixel>> mapping;

    inline friend std::ostream& operator<<(std::ostream& os, const OptimizationData& o) {
        // clang-format on
        os << "Ref: " << o.ref << std::endl;
        // clang-format on;
        return os;
    }

    static OptimizationData create(const Pixel&,
                                   const std::map<Pixel, Ray, PixelLess>&,
                                   const std::map<NeighborRelation, Pixel>&);
};
}
