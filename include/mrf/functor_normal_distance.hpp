#pragma once

#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>

#include "normals.hpp"

namespace mrf {

struct FunctorNormalDistanceCorner {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 2;

    inline FunctorNormalDistanceCorner(
        const Ray& ray_0, const Ray& ray_1, const double& w_1, const Ray& ray_2, const double& w_2)
            : ray_0_{ray_0}, ray_1_{ray_1}, w_1_{w_1}, ray_2_{ray_2}, w_2_{w_2} {};

    template <typename T>
    inline bool operator()(const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           T* res) const {
        const Eigen::Hyperplane<T, 3> plane(estimateNormal1(d_0[0],
                                                            ray_0_.cast<T>(),
                                                            d_1[0],
                                                            ray_1_.cast<T>(),
                                                            d_2[0],
                                                            ray_2_.cast<T>(),
                                                            static_cast<T>(0.1)),
                                            ray_0_.cast<T>().pointAt(d_0[0]));
        res[0] = static_cast<T>(w_1_) * plane.signedDistance(ray_1_.cast<T>().pointAt(d_1[0]));
        res[1] = static_cast<T>(w_1_) * plane.signedDistance(ray_2_.cast<T>().pointAt(d_2[0]));
        return true;
    }

    inline static ceres::CostFunction* create(const Ray& ray_0,
                                              const Ray& ray_1,
                                              const double& w_1,
                                              const Ray& ray_2,
                                              const double& w_2) {
        return new ceres::AutoDiffCostFunction<FunctorNormalDistanceCorner,
                                               DimResidual,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth>(
            new FunctorNormalDistanceCorner(ray_0, ray_1, w_1, ray_2, w_2));
    }

private:
    const Ray ray_0_;
    const Ray ray_1_;
    const double w_1_;
    const Ray ray_2_;
    const double w_2_;
};

struct FunctorNormalDistanceSide {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 3;

    inline FunctorNormalDistanceSide(const Ray& ray_0,
                                     const Ray& ray_1,
                                     const double& w_1,
                                     const Ray& ray_2,
                                     const double& w_2,
                                     const Ray& ray_3,
                                     const double& w_3)
            : ray_0_{ray_0}, ray_1_{ray_1}, w_1_{w_1}, ray_2_{ray_2}, w_2_{w_2}, ray_3_{ray_3},
              w_3_{w_3} {};

    template <typename T>
    inline bool operator()(const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           const T* const d_3,
                           T* res) const {
        const Eigen::Hyperplane<T, 3> plane(estimateNormal2(d_0[0],
                                                            ray_0_.cast<T>(),
                                                            d_1[0],
                                                            ray_1_.cast<T>(),
                                                            d_2[0],
                                                            ray_2_.cast<T>(),
                                                            d_3[0],
                                                            ray_3_.cast<T>(),
                                                            static_cast<T>(0.1)),
                                            ray_0_.cast<T>().pointAt(d_0[0]));
        res[0] = static_cast<T>(w_1_) * plane.signedDistance(ray_1_.cast<T>().pointAt(d_1[0]));
        res[1] = static_cast<T>(w_2_) * plane.signedDistance(ray_2_.cast<T>().pointAt(d_2[0]));
        res[2] = static_cast<T>(w_3_) * plane.signedDistance(ray_3_.cast<T>().pointAt(d_3[0]));
        return true;
    }

    inline static ceres::CostFunction* create(const Ray& ray_0,
                                              const Ray& ray_1,
                                              const double& w_1,
                                              const Ray& ray_2,
                                              const double& w_2,
                                              const Ray& ray_3,
                                              const double& w_3) {
        return new ceres::AutoDiffCostFunction<FunctorNormalDistanceSide,
                                               DimResidual,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth>(
            new FunctorNormalDistanceSide(ray_0, ray_1, w_1, ray_2, w_2, ray_3, w_3));
    }

private:
    const Ray ray_0_;
    const Ray ray_1_;
    const double w_1_;
    const Ray ray_2_;
    const double w_2_;
    const Ray ray_3_;
    const double w_3_;
};

struct FunctorNormalDistanceFull {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 4;

    inline FunctorNormalDistanceFull(const Ray& ray_0,
                                     const Ray& ray_1,
                                     const double& w_1,
                                     const Ray& ray_2,
                                     const double& w_2,
                                     const Ray& ray_3,
                                     const double& w_3,
                                     const Ray& ray_4,
                                     const double& w_4)
            : ray_0_{ray_0}, ray_1_{ray_1}, w_1_{w_1}, ray_2_{ray_2}, w_2_{w_2}, ray_3_{ray_3},
              w_3_{w_3}, ray_4_{ray_4}, w_4_{w_4} {};

    template <typename T>
    inline bool operator()(const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           const T* const d_3,
                           const T* const d_4,
                           T* res) const {
        const Eigen::Hyperplane<T, 3> plane(estimateNormal4(d_0[0],
                                                            ray_0_.cast<T>(),
                                                            d_1[0],
                                                            ray_1_.cast<T>(),
                                                            d_2[0],
                                                            ray_2_.cast<T>(),
                                                            d_3[0],
                                                            ray_3_.cast<T>(),
                                                            d_4[0],
                                                            ray_4_.cast<T>(),
                                                            static_cast<T>(0.1)),
                                            ray_0_.cast<T>().pointAt(d_0[0]));
        res[0] = static_cast<T>(w_1_) * plane.signedDistance(ray_1_.cast<T>().pointAt(d_1[0]));
        res[1] = static_cast<T>(w_2_) * plane.signedDistance(ray_2_.cast<T>().pointAt(d_2[0]));
        res[2] = static_cast<T>(w_3_) * plane.signedDistance(ray_3_.cast<T>().pointAt(d_3[0]));
        res[3] = static_cast<T>(w_4_) * plane.signedDistance(ray_4_.cast<T>().pointAt(d_4[0]));
        return true;
    }

    inline static ceres::CostFunction* create(const Ray& ray_0,
                                              const Ray& ray_1,
                                              const double& w_1,
                                              const Ray& ray_2,
                                              const double& w_2,
                                              const Ray& ray_3,
                                              const double& w_3,
                                              const Ray& ray_4,
                                              const double& w_4) {
        return new ceres::AutoDiffCostFunction<FunctorNormalDistanceFull,
                                               DimResidual,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth>(
            new FunctorNormalDistanceFull(ray_0, ray_1, w_1, ray_2, w_2, ray_3, w_3, ray_4, w_4));
    }

private:
    const Ray ray_0_;
    const Ray ray_1_;
    const double w_1_;
    const Ray ray_2_;
    const double w_2_;
    const Ray ray_3_;
    const double w_3_;
    const Ray ray_4_;
    const double w_4_;
};
}
