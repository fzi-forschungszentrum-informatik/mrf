#include "solver.hpp"

#include <cmath>

namespace mrf {

std::vector<double> smoothnessWeights(const int p,
                                      const std::vector<int>& neighbors,
                                      const cv::Mat& img) {

    const int width{img.rows};
    std::vector<double> w(neighbors.size(), 0);
    const int p_row{p % width};
    const int p_col = std::floor(p / width);

    for (size_t c = 0; c < neighbors.size(); c++) {
        if (neighbors[c] == -1) {
            w[c] = 0;
        } else {
            const int row = neighbors[c] % width;
            const int col = std::floor(neighbors[c] / width);
            w[c] = abs(img.at<uchar>(col, row) - img.at<uchar>(p_col, p_row));
        }
    }
    return w;
}
}
