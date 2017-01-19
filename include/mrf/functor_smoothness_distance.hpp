#pragma once

#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorSmoothnessDistance {

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;

    template <typename T>
    inline bool operator()(const T* const xi, const T* const xj, T* res) const {
        res[0] = xi[0] - xj[0];
        return true;
    }

    inline static ceres::CostFunction* create() {
        return new ceres::
            AutoDiffCostFunction<FunctorSmoothnessDistance, DimResidual, DimDepth, DimDepth>(
                new FunctorSmoothnessDistance());
    }
};
}
