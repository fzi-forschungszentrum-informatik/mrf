#include "neighbors.hpp"

#include "cv_helper.hpp"
#include "neighbor_relation.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p,
                                const cv::Mat& img,
                                const Parameters::Neighborhood& mode,
                                const int& row_max,
                                const int& col_max,
                                const int& row_min,
                                const int& col_min) {

    std::map<NeighborRelation, Pixel> neighbors;
    if (p.row > row_min)
        neighbors.emplace(NeighborRelation::top,
                          Pixel(p.col, p.row - 1, getVector<float>(img, p.row - 1, p.col)));
    if (p.col > col_min)
        neighbors.emplace(NeighborRelation::left,
                          Pixel(p.col - 1, p.row, getVector<float>(img, p.row, p.col - 1)));
    if (p.row < row_max - 1)
        neighbors.emplace(NeighborRelation::bottom,
                          Pixel(p.col, p.row + 1, getVector<float>(img, p.row + 1, p.col)));
    if (p.col < col_max - 1)
        neighbors.emplace(NeighborRelation::right,
                          Pixel(p.col + 1, p.row, getVector<float>(img, p.row, p.col + 1)));

    std::vector<Pixel> out;
    if (neighbors.size() < 3) {
        if (!neighbors.count(NeighborRelation::top) && !neighbors.count(NeighborRelation::left)) {
            out.emplace_back(neighbors.at(NeighborRelation::bottom));
            out.emplace_back(neighbors.at(NeighborRelation::right));
        } else if (!neighbors.count(NeighborRelation::top) &&
                   !neighbors.count(NeighborRelation::right)) {
            out.emplace_back(neighbors.at(NeighborRelation::left));
            out.emplace_back(neighbors.at(NeighborRelation::bottom));
        } else if (!neighbors.count(NeighborRelation::bottom) &&
                   !neighbors.count(NeighborRelation::left)) {
            out.emplace_back(neighbors.at(NeighborRelation::right));
            out.emplace_back(neighbors.at(NeighborRelation::top));
        } else {
            out.emplace_back(neighbors.at(NeighborRelation::top));
            out.emplace_back(neighbors.at(NeighborRelation::left));
        }
    } else if (neighbors.size() < 4) {
        if (!neighbors.count(NeighborRelation::top)) {
            out.emplace_back(neighbors.at(NeighborRelation::left));
            out.emplace_back(neighbors.at(NeighborRelation::bottom));
            out.emplace_back(neighbors.at(NeighborRelation::right));
        } else if (!neighbors.count(NeighborRelation::left)) {
            out.emplace_back(neighbors.at(NeighborRelation::bottom));
            out.emplace_back(neighbors.at(NeighborRelation::right));
            out.emplace_back(neighbors.at(NeighborRelation::top));
        } else if (!neighbors.count(NeighborRelation::bottom)) {
            out.emplace_back(neighbors.at(NeighborRelation::right));
            out.emplace_back(neighbors.at(NeighborRelation::top));
            out.emplace_back(neighbors.at(NeighborRelation::left));
        } else {
            out.emplace_back(neighbors.at(NeighborRelation::top));
            out.emplace_back(neighbors.at(NeighborRelation::left));
            out.emplace_back(neighbors.at(NeighborRelation::bottom));
        }
    } else {
        out.emplace_back(neighbors.at(NeighborRelation::top));
        out.emplace_back(neighbors.at(NeighborRelation::left));
        out.emplace_back(neighbors.at(NeighborRelation::bottom));
        out.emplace_back(neighbors.at(NeighborRelation::right));
    }
    return out;
}
}
