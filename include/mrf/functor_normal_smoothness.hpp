#pragma once

#include <ostream>
#include <ceres/autodiff_cost_function.h>

namespace mrf {

struct FunctorNormalSmoothness {

    static constexpr size_t DimDepth = 3;
    static constexpr size_t DimResidual = 3;

    inline FunctorNormalSmoothness(const double& w) : w_{w} {};

    template <typename T>
    inline bool operator()(const T* const ni, const T* const nj, T* res) const {
        res[0] = T(w_) * (ni[0] - nj[0]);
        res[1] = T(w_) * (ni[1] - nj[2]);
        res[2] = T(w_) * (ni[1] - nj[2]);
        return true;
    }

    inline static ceres::CostFunction* create(const double& e) {
        return new ceres::AutoDiffCostFunction<FunctorNormalSmoothness, DimResidual, DimDepth, DimDepth>(
            new FunctorNormalSmoothness(e));
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormalSmoothness& f) {
        os << "Weight: " << f.w_ << std::endl;
        return os;
    }

private:
    const double w_;
};
}
