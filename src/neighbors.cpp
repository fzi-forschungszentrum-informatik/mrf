#include "neighbors.hpp"

namespace mrf {

std::map<NeighborRelation, Pixel> getNeighbors(const Pixel& p,
                                               const cv::Mat& img,
                                               const Parameters::Neighborhood& mode) {

    std::map<NeighborRelation, Pixel> neighbors;

    if (p.row > 0) {
        neighbors.insert(std::make_pair(NeighborRelation::top,
                                        Pixel(p.col, p.row - 1, img.at<double>(p.row - 1, p.col))));
    }
    if (p.col > 0) {
        neighbors.insert(std::make_pair(NeighborRelation::left,
                                        Pixel(p.col - 1, p.row, img.at<double>(p.row, p.col - 1))));
    }

    if (static_cast<int>(mode) > 2) {
        if (p.row < img.rows - 1) {
            neighbors.insert(
                std::make_pair(NeighborRelation::bottom,
                               Pixel(p.col, p.row + 1, img.at<double>(p.row + 1, p.col))));
        }
        if (p.col < img.cols - 1) {
            neighbors.insert(
                std::make_pair(NeighborRelation::right,
                               Pixel(p.col + 1, p.row, img.at<double>(p.row, p.col + 1))));
        }
    }

    if (static_cast<int>(mode) > 4) {
        if (p.row > 0 && p.col < img.cols - 1) {
            neighbors.insert(
                std::make_pair(NeighborRelation::top_right,
                               Pixel(p.col + 1, p.row - 1, img.at<double>(p.row - 1, p.col + 1))));
        }
        if (p.row < img.rows - 1 && p.col < img.cols - 1) {
            neighbors.insert(
                std::make_pair(NeighborRelation::bottom_right,
                               Pixel(p.col + 1, p.row + 1, img.at<double>(p.row + 1, p.col + 1))));
        }
        if (p.row > 0 && p.col > 0) {
            neighbors.insert(
                std::make_pair(NeighborRelation::top_left,
                               Pixel(p.col - 1, p.row - 1, img.at<double>(p.row - 1, p.col - 1))));
        }
        if (p.row < img.rows - 1 && p.col > 0) {
            neighbors.insert(
                std::make_pair(NeighborRelation::bottom_left,
                               Pixel(p.col - 1, p.row + 1, img.at<double>(p.row + 1, p.col - 1))));
        }
    }

    return neighbors;
}
}
