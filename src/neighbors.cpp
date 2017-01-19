#include "neighbors.hpp"

#include "cv_helper.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p,
                                const cv::Mat& img,
                                const Parameters::Neighborhood& mode) {

    std::vector<Pixel> neighbors;
    neighbors.reserve(static_cast<int>(mode));

    if (p.row > 0)
        neighbors.emplace_back(p.col, p.row - 1, getVector<float>(img, p.row - 1, p.col));
    if (p.col > 0)
        neighbors.emplace_back(p.col - 1, p.row, getVector<float>(img, p.row, p.col - 1));

    if (static_cast<int>(mode) > 2) {
        if (p.row < img.rows - 1)
            neighbors.emplace_back(p.col, p.row + 1, getVector<float>(img, p.row + 1, p.col));
        if (p.col < img.cols - 1)
            neighbors.emplace_back(p.col + 1, p.row, getVector<float>(img, p.row, p.col + 1));
    }

    if (static_cast<int>(mode) > 4) {
        if (p.row > 0 && p.col < img.cols - 1)
            neighbors.emplace_back(
                p.col + 1, p.row - 1, getVector<float>(img, p.row - 1, p.col + 1));
        if (p.row < img.rows - 1 && p.col < img.cols - 1)
            neighbors.emplace_back(
                p.col + 1, p.row + 1, getVector<float>(img, p.row + 1, p.col + 1));
        if (p.row > 0 && p.col > 0)
            neighbors.emplace_back(
                p.col - 1, p.row - 1, getVector<float>(img, p.row - 1, p.col - 1));
        if (p.row < img.rows - 1 && p.col > 0)
            neighbors.emplace_back(
                p.col - 1, p.row + 1, getVector<float>(img, p.row + 1, p.col - 1));
    }

    return neighbors;
}
}
