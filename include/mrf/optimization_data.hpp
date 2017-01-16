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
    std::vector<Pixel> ref;
    std::map<Pixel, std::vector<std::pair<Pixel, Pixel>>, PixelLess> mapping;

    static OptimizationData create(const Pixel&,
                                   const std::map<Pixel, Ray, PixelLess>&,
                                   const std::map<NeighborRelation, Pixel>&);
    static OptimizationData create(
        const std::vector<Pixel>&,
        const std::map<Pixel, Ray, PixelLess>&,
        const std::map<Pixel, std::map<NeighborRelation, Pixel>, PixelLess>&);

    void addRef(const Pixel&,
                const std::map<Pixel, Ray, PixelLess>&,
                const std::map<NeighborRelation, Pixel>&);
};
}
