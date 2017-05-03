#pragma once

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

namespace mrf {

/** @brief Get vector with image values of a given position.
 *  @param img Input image
 *  @param row Requested row
 *  @param col Requested column
 *  @return Vector with image values of all channels. */
template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> getVector(const cv::Mat& img,
                                                     const size_t& row,
                                                     const size_t& col) {
    const int channels{img.channels()};
    Eigen::Matrix<T, Eigen::Dynamic, 1> out(channels);
    for (int d = 0; d < channels; d++)
        out[d] = img.ptr<T>(row)[col * channels + d];
    return out;
}
}
