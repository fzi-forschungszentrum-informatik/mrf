#pragma once

#include <ostream>
#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorSmoothness {

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;

    inline FunctorSmoothness(const double& w) : w_{w} {};

    template <typename T>
    inline bool operator()(const T* const xi, const T* const xj, T* res) const {
        res[0] = T(w_) * (xi[0] - xj[0]);
        return true;
    }

    inline static ceres::CostFunction* create(const double& e) {
        return new ceres::AutoDiffCostFunction<FunctorSmoothness, DimResidual, DimDepth, DimDepth>(
            new FunctorSmoothness(e));
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorSmoothness& f) {
        os << "Weight: " << f.w_ << std::endl;
        return os;
    }

private:
    const double w_;
};
}
