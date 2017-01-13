#pragma once

#include <ceres/dynamic_autodiff_cost_function.h>
#include <util_ceres/eigen.h>

#include "normals.hpp"
#include "optimization_data.hpp"

namespace mrf {

struct FunctorNormal {

    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n, const OptimizationData& d) : n_{n}, d_(d){};

    template <typename T>
    inline bool operator()(T const* const* parameters, T* res_ceres) const {
        size_t it{0};
        const Eigen::Affine3<T> tf{util_ceres::fromQuaternion(parameters[it++])};
        std::map<Pixel, T, PixelLess> depths;
        for (auto const& nb : d_.rays)
            depths[nb.first] = parameters[it++][0];
        res_ceres[0] =
            T(1) - estimateNormal(d_.ref, d_.rays, depths, d_.mapping).dot(tf * n_.cast<T>());
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& n, const OptimizationData& d) {
        using CostFunction = ceres::DynamicAutoDiffCostFunction<FunctorNormal>;
        CostFunction* cf{new CostFunction(new FunctorNormal(n, d))};
        cf->AddParameterBlock(DimRotation);
        for (auto const& el : d.rays)
            cf->AddParameterBlock(1);
        cf->SetNumResiduals(DimResidual);
        return cf;
    }


private:
    Eigen::Vector3d n_; ///< 3D world point normal associated to this pixel
    OptimizationData d_;
};
}
