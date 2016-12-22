#include "neighbors.hpp"

namespace mrf {

std::vector<Pixel> getNeighbors(const Pixel& p, const cv::Mat& img,
                                const Parameters::Neighborhood& mode) {

    std::vector<Pixel> neighbors;
    neighbors.reserve(static_cast<size_t>(mode));

    if (p.row > 0) {
        neighbors.emplace_back(p.col, p.row - 1, img.at<double>(p.row - 1, p.col));
    }
    if (p.row < img.rows - 1) {
        neighbors.emplace_back(p.col, p.row + 1, img.at<double>(p.row + 1, p.col));
    }
    if (p.col > 0) {
        neighbors.emplace_back(p.col - 1, p.row, img.at<double>(p.row, p.col - 1));
    }
    if (p.col < img.cols - 1) {
        neighbors.emplace_back(p.col + 1, p.row, img.at<double>(p.row, p.col + 1));
    }

    if (mode == Parameters::Neighborhood::eight) {
        if (p.row > 0 && p.col < img.cols - 1) {
            neighbors.emplace_back(p.col + 1, p.row - 1, img.at<double>(p.row - 1, p.col + 1));
        }
        if (p.row < img.rows - 1 && p.col < img.cols - 1) {
            neighbors.emplace_back(p.col + 1, p.row + 1, img.at<double>(p.row + 1, p.col + 1));
        }
        if (p.row > 0 && p.col > 0) {
            neighbors.emplace_back(p.col - 1, p.row - 1, img.at<double>(p.row - 1, p.col - 1));
        }
        if (p.row < img.rows - 1 && p.col > 0) {
            neighbors.emplace_back(p.col - 1, p.row + 1, img.at<double>(p.row + 1, p.col - 1));
        }
    }

    return neighbors;
}
}
