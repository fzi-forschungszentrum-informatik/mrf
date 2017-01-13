#pragma once

#include <Eigen/Geometry>
#include <ceres/dynamic_autodiff_cost_function.h>

#include "neighbors.hpp"
#include "normals.hpp"
#include "optimization_data.hpp"
#include "pixel.hpp"

namespace mrf {


struct FunctorNormalDistance {

    inline FunctorNormalDistance(const OptimizationData& d) : d_(d){};

    template <typename T>
    inline bool operator()(T const* const* parameters, T* res) const {
        using namespace Eigen;
        size_t it{0};
        std::map<Pixel, T, PixelLess> depths;
        for (auto const& el : d_.rays)
            depths[el.first] = parameters[it++][0];
        const Hyperplane<T, 3> plane_this(estimateNormal(d_.ref, d_.rays, depths, d_.mapping, T(10)),
                                          d_.rays.at(d_.ref).cast<T>().pointAt(depths.at(d_.ref)));
        it = 0;
        for (auto const& el : d_.rays) {
            res[it++] = plane_this.signedDistance(el.second.cast<T>().pointAt(depths.at(el.first)));
        }
        return true;
    }

    inline static ceres::CostFunction* create(const OptimizationData& d) {
        using CostFunction = ceres::DynamicAutoDiffCostFunction<FunctorNormalDistance, 10>;
        CostFunction* cf{new CostFunction(new FunctorNormalDistance(d))};
        for (auto const& el : d.rays)
            cf->AddParameterBlock(1);
        cf->SetNumResiduals(d.rays.size());
        return cf;
    }

private:
    const OptimizationData d_;
};
}
