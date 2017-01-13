#include "optimization_data.hpp"

namespace mrf {

OptimizationData OptimizationData::create(const Pixel& ref,
                                          const std::map<Pixel, Ray, PixelLess>& all_rays,
                                          const std::map<NeighborRelation, Pixel>& neighbors) {

    OptimizationData d;

    d.ref = ref;
    d.rays[d.ref] = all_rays.at(d.ref);
    if (neighbors.count(NeighborRelation::bottom) && neighbors.count(NeighborRelation::right)) {
        d.rays[neighbors.at(NeighborRelation::bottom)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom));
        d.rays[neighbors.at(NeighborRelation::right)] =
            all_rays.at(neighbors.at(NeighborRelation::right));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::bottom),
                                           neighbors.at(NeighborRelation::right)));
    }
    if (neighbors.count(NeighborRelation::bottom_left) &&
        neighbors.count(NeighborRelation::bottom_right)) {
        d.rays[neighbors.at(NeighborRelation::bottom_left)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_left));
        d.rays[neighbors.at(NeighborRelation::bottom_right)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_right));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::bottom_left),
                                           neighbors.at(NeighborRelation::bottom_right)));
    }
    if (neighbors.count(NeighborRelation::left) && neighbors.count(NeighborRelation::bottom)) {
        d.rays[neighbors.at(NeighborRelation::left)] =
            all_rays.at(neighbors.at(NeighborRelation::left));
        d.rays[neighbors.at(NeighborRelation::bottom)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::left),
                                           neighbors.at(NeighborRelation::bottom)));
    }
    if (neighbors.count(NeighborRelation::top_left) &&
        neighbors.count(NeighborRelation::bottom_left)) {
        d.rays[neighbors.at(NeighborRelation::top_left)] =
            all_rays.at(neighbors.at(NeighborRelation::top_left));
        d.rays[neighbors.at(NeighborRelation::bottom_left)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_left));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::top_left),
                                           neighbors.at(NeighborRelation::bottom_left)));
    }
    if (neighbors.count(NeighborRelation::top) && neighbors.count(NeighborRelation::left)) {
        d.rays[neighbors.at(NeighborRelation::top)] =
            all_rays.at(neighbors.at(NeighborRelation::top));
        d.rays[neighbors.at(NeighborRelation::left)] =
            all_rays.at(neighbors.at(NeighborRelation::left));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::top),
                                           neighbors.at(NeighborRelation::left)));
    }
    if (neighbors.count(NeighborRelation::top_right) &&
        neighbors.count(NeighborRelation::top_left)) {
        d.rays[neighbors.at(NeighborRelation::top_right)] =
            all_rays.at(neighbors.at(NeighborRelation::top_right));
        d.rays[neighbors.at(NeighborRelation::top_left)] =
            all_rays.at(neighbors.at(NeighborRelation::top_left));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::top_right),
                                           neighbors.at(NeighborRelation::top_left)));
    }
    if (neighbors.count(NeighborRelation::right) && neighbors.count(NeighborRelation::top)) {
        d.rays[neighbors.at(NeighborRelation::right)] =
            all_rays.at(neighbors.at(NeighborRelation::right));
        d.rays[neighbors.at(NeighborRelation::top)] =
            all_rays.at(neighbors.at(NeighborRelation::top));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::right),
                                           neighbors.at(NeighborRelation::top)));
    }
    if (neighbors.count(NeighborRelation::bottom_right) &&
        neighbors.count(NeighborRelation::top_right)) {
        d.rays[neighbors.at(NeighborRelation::bottom_right)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_right));
        d.rays[neighbors.at(NeighborRelation::top_right)] =
            all_rays.at(neighbors.at(NeighborRelation::top_right));
        d.mapping.push_back(std::make_pair(neighbors.at(NeighborRelation::bottom_right),
                                           neighbors.at(NeighborRelation::top_right)));
    }

    return d;
}
}
