#pragma once

#include <memory>
#include <ostream>
#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorDistance {

    using Ptr = std::shared_ptr<FunctorDistance>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimRotation = 4;
    static constexpr size_t DimTranslation = 3;

    inline FunctorDistance(const Eigen::Vector3d& p, const Eigen::ParametrizedLine<double, 3>& ray)
            : p_{p}, ray_{ray} {};

    template <typename T>
    inline bool operator()(const T* const depth, const T* const rotation,
                           const T* const translation, T* res) const {
        using namespace Eigen;
        const Affine3<T> tf{util_ceres::fromQuaternionTranslation(rotation, translation)};
        Map<Vector3<T>>(res, DimResidual) = ray_.cast<T>().pointAt(depth[0]) - tf * p_.cast<T>();
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& p,
                             const Eigen::ParametrizedLine<double, 3>& ray) {
        return std::make_shared<FunctorDistance>(p, ray);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<FunctorDistance, DimResidual, DimDepth, DimRotation,
                                               DimTranslation>(this);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorDistance& f) {
        return os << "Vector: " << f.p_ << std::endl;
    }

    inline void setPoint(const Eigen::Vector3d& p) {
        p_ = p;
    }

private:
    Eigen::Vector3d p_; ///< 3D world point associated to this pixel
    Eigen::ParametrizedLine<double, 3> ray_;
};
}
