#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

#include "neighbors.hpp"
#include "normals.hpp"
#include "optimization_data.hpp"
#include "pixel.hpp"

namespace mrf {

struct FunctorSmoothnessNormal {

    static constexpr size_t DimResidual = 1;

    inline FunctorSmoothnessNormal(const OptimizationData& d) : d_(d){};

    template <typename T>
    inline bool operator()(T const* const* parameters, T* res_ceres) const {
        using namespace Eigen;
        size_t it{0};
        std::map<Pixel, T, PixelLess> depths;
        for (auto const& el : d_.rays)
            depths[el.first] = parameters[it++][0];
        const Vector3<T> n0{
            estimateNormal(d_.ref[0], d_.rays, depths, d_.mapping.at(d_.ref[0]), T(10))};
        const Vector3<T> n1{
            estimateNormal(d_.ref[1], d_.rays, depths, d_.mapping.at(d_.ref[1]), T(10))};
        res_ceres[0] = n0.dot(n1) - static_cast<T>(1);
        return true;
    }

    inline static ceres::CostFunction* create(const OptimizationData& d) {
        using CostFunction = ceres::DynamicAutoDiffCostFunction<FunctorSmoothnessNormal>;
        CostFunction* cf{new CostFunction(new FunctorSmoothnessNormal(d))};
        for (auto const& el : d.rays)
            cf->AddParameterBlock(1);
        cf->SetNumResiduals(DimResidual);
        return cf;
    }

private:
    const OptimizationData d_;
};
}
