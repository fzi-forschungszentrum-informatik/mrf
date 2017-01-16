#include "optimization_data.hpp"

namespace mrf {

void OptimizationData::addRef(const Pixel& ref_,
                              const std::map<Pixel, Ray, PixelLess>& all_rays,
                              const std::map<NeighborRelation, Pixel>& neighbors) {

    ref.emplace_back(ref_);
    rays[ref_] = all_rays.at(ref_);
    if (neighbors.count(NeighborRelation::bottom) && neighbors.count(NeighborRelation::right)) {
        rays[neighbors.at(NeighborRelation::bottom)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom));
        rays[neighbors.at(NeighborRelation::right)] =
            all_rays.at(neighbors.at(NeighborRelation::right));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::bottom),
                                               neighbors.at(NeighborRelation::right)));
    }
    if (neighbors.count(NeighborRelation::bottom_left) &&
        neighbors.count(NeighborRelation::bottom_right)) {
        rays[neighbors.at(NeighborRelation::bottom_left)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_left));
        rays[neighbors.at(NeighborRelation::bottom_right)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_right));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::bottom_left),
                                               neighbors.at(NeighborRelation::bottom_right)));
    }
    if (neighbors.count(NeighborRelation::left) && neighbors.count(NeighborRelation::bottom)) {
        rays[neighbors.at(NeighborRelation::left)] =
            all_rays.at(neighbors.at(NeighborRelation::left));
        rays[neighbors.at(NeighborRelation::bottom)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::left),
                                               neighbors.at(NeighborRelation::bottom)));
    }
    if (neighbors.count(NeighborRelation::top_left) &&
        neighbors.count(NeighborRelation::bottom_left)) {
        rays[neighbors.at(NeighborRelation::top_left)] =
            all_rays.at(neighbors.at(NeighborRelation::top_left));
        rays[neighbors.at(NeighborRelation::bottom_left)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_left));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::top_left),
                                               neighbors.at(NeighborRelation::bottom_left)));
    }
    if (neighbors.count(NeighborRelation::top) && neighbors.count(NeighborRelation::left)) {
        rays[neighbors.at(NeighborRelation::top)] =
            all_rays.at(neighbors.at(NeighborRelation::top));
        rays[neighbors.at(NeighborRelation::left)] =
            all_rays.at(neighbors.at(NeighborRelation::left));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::top),
                                               neighbors.at(NeighborRelation::left)));
    }
    if (neighbors.count(NeighborRelation::top_right) &&
        neighbors.count(NeighborRelation::top_left)) {
        rays[neighbors.at(NeighborRelation::top_right)] =
            all_rays.at(neighbors.at(NeighborRelation::top_right));
        rays[neighbors.at(NeighborRelation::top_left)] =
            all_rays.at(neighbors.at(NeighborRelation::top_left));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::top_right),
                                               neighbors.at(NeighborRelation::top_left)));
    }
    if (neighbors.count(NeighborRelation::right) && neighbors.count(NeighborRelation::top)) {
        rays[neighbors.at(NeighborRelation::right)] =
            all_rays.at(neighbors.at(NeighborRelation::right));
        rays[neighbors.at(NeighborRelation::top)] =
            all_rays.at(neighbors.at(NeighborRelation::top));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::right),
                                               neighbors.at(NeighborRelation::top)));
    }
    if (neighbors.count(NeighborRelation::bottom_right) &&
        neighbors.count(NeighborRelation::top_right)) {
        rays[neighbors.at(NeighborRelation::bottom_right)] =
            all_rays.at(neighbors.at(NeighborRelation::bottom_right));
        rays[neighbors.at(NeighborRelation::top_right)] =
            all_rays.at(neighbors.at(NeighborRelation::top_right));
        mapping[ref_].push_back(std::make_pair(neighbors.at(NeighborRelation::bottom_right),
                                               neighbors.at(NeighborRelation::top_right)));
    }
}


OptimizationData OptimizationData::create(const Pixel& ref,
                                          const std::map<Pixel, Ray, PixelLess>& all_rays,
                                          const std::map<NeighborRelation, Pixel>& neighbors) {
    OptimizationData d;
    d.addRef(ref, all_rays, neighbors);
    return d;
}

OptimizationData OptimizationData::create(
    const std::vector<Pixel>& refs,
    const std::map<Pixel, Ray, PixelLess>& all_rays,
    const std::map<Pixel, std::map<NeighborRelation, Pixel>, PixelLess>& mappings) {
    OptimizationData d;
    for (auto const& ref : refs)
        d.addRef(ref, all_rays, mappings.at(ref));
    return d;
}
}
