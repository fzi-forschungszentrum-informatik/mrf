#pragma once

#include <algorithm>
#include <Eigen/src/Core/Matrix.h>

namespace mrf {

struct EigenLess {

    inline bool operator()(const Eigen::VectorXi& lhs, const Eigen::VectorXi& rhs) {
        return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(), rhs.data(),
                                            rhs.data() + rhs.size());
    }

    inline bool operator()(const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) {
        if (lhs[0] < rhs[0])
            return true;
        else if (lhs[0] > rhs[0])
            return false;
        else if (lhs[1] < rhs[1])
            return true;
        return false;
    }
};
}
