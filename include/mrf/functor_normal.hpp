#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

#include "normals.hpp"

namespace mrf {

struct FunctorNormal {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n, const Ray& ray_0, const Ray& ray_1)
            : n_{n}, ray_0_{ray_0}, ray_1_{ray_1} {};

    template <typename T>
    inline bool operator()(const T* const rot,
                           const T* const d_0,
                           const T* const d_1,
                           T* res) const {
        using namespace Eigen;
        const Vector3<T> p_0{ray_0_.cast<T>().pointAt(d_0[0])};
        const Vector3<T> p_1{ray_1_.cast<T>().pointAt(d_1[0])};
        const Hyperplane<T, 3> plane{util_ceres::fromQuaternion(rot) * n_.cast<T>(), p_0};

        res[0] = plane.signedDistance(p_1);
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& n,
                                              const Ray& ray_0,
                                              const Ray& ray_1) {
        return new ceres::
            AutoDiffCostFunction<FunctorNormal, DimResidual, DimRotation, DimDepth, DimDepth>(
                new FunctorNormal(n, ray_0, ray_1));
    }

private:
    const Eigen::Vector3d n_;
    const Ray ray_0_;
    const Ray ray_1_;
};
}
