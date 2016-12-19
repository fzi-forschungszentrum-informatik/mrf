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

    inline FunctorDistance(const Eigen::Vector3d& p, const double& w,
                           const Eigen::Vector3d& support, const Eigen::Vector3d& direction)
            : p_{p}, w_{w}, support_{support}, direction_{direction} {};

    template <typename T>
    inline bool operator()(const T* const depth, const T* const rotation,
                           const T* const translation, T* res) const {
        const Eigen::Affine3<T> tf{util_ceres::fromQuaternionTranslation(rotation, translation)};
        Eigen::Map<Eigen::Vector3<T>>(res, DimResidual) =
            T(w_) * (support_.cast<T>() + direction_.cast<T>() * depth[0] - tf * p_.cast<T>());
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& p, const double& w,
                             const Eigen::Vector3d& support, const Eigen::Vector3d& direction) {
        return std::make_shared<FunctorDistance>(p, w, support, direction);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<FunctorDistance, DimResidual, DimDepth, DimRotation,
                                               DimTranslation>(this);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorDistance& f) {
        os << "Vector: " << f.p_ << std::endl << "Weight: " << f.w_ << std::endl;
        return os;
    }

    inline void setPoint(const Eigen::Vector3d& p) {
        p_ = p;
    }
    inline void setWeight(const double w) {
        w_ = w;
    }

private:
    Eigen::Vector3d p_; ///< 3D world point associated to this pixel
    double w_;          ///< Weight

    Eigen::Vector3d support_;
    Eigen::Vector3d direction_;
};
}
