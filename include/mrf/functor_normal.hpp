#pragma once

#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

#include "normals.hpp"

namespace mrf {

struct FunctorNormalCorner {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormalCorner(const Eigen::Vector3d& n,
                               const Ray& ray_0,
                               const Ray& ray_1,
                               const Ray& ray_2)
            : n_{n}, ray_0_{ray_0}, ray_1_{ray_1}, ray_2_{ray_2} {};

    template <typename T>
    inline bool operator()(const T* const rot,
                           const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           T* res) const {

        const Eigen::Vector3<T> n{estimateNormal1(d_0[0],
                                                  ray_0_.cast<T>(),
                                                  d_1[0],
                                                  ray_1_.cast<T>(),
                                                  d_2[0],
                                                  ray_2_.cast<T>(),
                                                  static_cast<T>(0.1))};
        res[0] = n.dot(util_ceres::fromQuaternion(rot) * n_.cast<T>()) - static_cast<T>(1);
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& n,
                                              const Ray& ray_0,
                                              const Ray& ray_1,
                                              const Ray& ray_2) {
        return new ceres::AutoDiffCostFunction<FunctorNormalCorner,
                                               DimResidual,
                                               DimRotation,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth>(
            new FunctorNormalCorner(n, ray_0, ray_1, ray_2));
    }

private:
    const Eigen::Vector3d n_;
    const Ray ray_0_;
    const Ray ray_1_;
    const Ray ray_2_;
};

struct FunctorNormalSide {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormalSide(const Eigen::Vector3d& n,
                             const Ray& ray_0,
                             const Ray& ray_1,
                             const Ray& ray_2,
                             const Ray& ray_3)
            : n_{n}, ray_0_{ray_0}, ray_1_{ray_1}, ray_2_{ray_2}, ray_3_{ray_3} {};

    template <typename T>
    inline bool operator()(const T* const rot,
                           const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           const T* const d_3,
                           T* res) const {

        const Eigen::Vector3<T> n{estimateNormal2(d_0[0],
                                                  ray_0_.cast<T>(),
                                                  d_1[0],
                                                  ray_1_.cast<T>(),
                                                  d_2[0],
                                                  ray_2_.cast<T>(),
                                                  d_3[0],
                                                  ray_3_.cast<T>(),
                                                  static_cast<T>(0.1))};
        res[0] = n.dot(util_ceres::fromQuaternion(rot) * n_.cast<T>()) - static_cast<T>(1);
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& n,
                                              const Ray& ray_0,
                                              const Ray& ray_1,
                                              const Ray& ray_2,
                                              const Ray& ray_3) {
        return new ceres::AutoDiffCostFunction<FunctorNormalSide,
                                               DimResidual,
                                               DimRotation,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth>(
            new FunctorNormalSide(n, ray_0, ray_1, ray_2, ray_3));
    }

private:
    const Eigen::Vector3d n_;
    const Ray ray_0_;
    const Ray ray_1_;
    const Ray ray_2_;
    const Ray ray_3_;
};

struct FunctorNormalFull {

    using Ray = Eigen::ParametrizedLine<double, 3>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormalFull(const Eigen::Vector3d& n,
                             const Ray& ray_0,
                             const Ray& ray_1,
                             const Ray& ray_2,
                             const Ray& ray_3,
                             const Ray& ray_4)
            : n_{n}, ray_0_{ray_0}, ray_1_{ray_1}, ray_2_{ray_2}, ray_3_{ray_3}, ray_4_{ray_4} {};

    template <typename T>
    inline bool operator()(const T* const rot,
                           const T* const d_0,
                           const T* const d_1,
                           const T* const d_2,
                           const T* const d_3,
                           const T* const d_4,
                           T* res) const {

        const Eigen::Vector3<T> n{estimateNormal4(d_0[0],
                                                  ray_0_.cast<T>(),
                                                  d_1[0],
                                                  ray_1_.cast<T>(),
                                                  d_2[0],
                                                  ray_2_.cast<T>(),
                                                  d_3[0],
                                                  ray_3_.cast<T>(),
                                                  d_4[0],
                                                  ray_4_.cast<T>(),
                                                  static_cast<T>(0.1))};
        res[0] = n.dot(util_ceres::fromQuaternion(rot) * n_.cast<T>()) - static_cast<T>(1);
        return true;
    }

    inline static ceres::CostFunction* create(const Eigen::Vector3d& n,
                                              const Ray& ray_0,
                                              const Ray& ray_1,
                                              const Ray& ray_2,
                                              const Ray& ray_3,
                                              const Ray& ray_4) {
        return new ceres::AutoDiffCostFunction<FunctorNormalFull,
                                               DimResidual,
                                               DimRotation,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth,
                                               DimDepth>(
            new FunctorNormalFull(n, ray_0, ray_1, ray_2, ray_3, ray_4));
    }

private:
    const Eigen::Vector3d n_;
    const Ray ray_0_;
    const Ray ray_1_;
    const Ray ray_2_;
    const Ray ray_3_;
    const Ray ray_4_;
};
}
