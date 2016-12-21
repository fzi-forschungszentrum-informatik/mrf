#pragma once

#include <memory>
#include <ostream>
#include <ceres/autodiff_cost_function.h>
#include <util_ceres/eigen.h>

namespace mrf {

struct FunctorNormal {

    using Ptr = std::shared_ptr<FunctorNormal>;

    static constexpr size_t DimNormal = 3;
    static constexpr size_t DimResidual = 3;
    static constexpr size_t DimRotation = 4;

    inline FunctorNormal(const Eigen::Vector3d& n, const double& w) : n_{n}, w_{w} {};

    template <typename T>
    inline bool operator()(const T* const normal, const T* const rotation, T* res) const {
        const Eigen::Affine3<T> tf{
            util_ceres::fromQuaternionTranslation(rotation, Eigen::Vector3<T>::Zero(3))};
        Eigen::Map<Eigen::Vector3<T>>(res, DimResidual) = T(w_) * (normal - tf * n_.cast<T>());
        return true;
    }

    inline static Ptr create(const Eigen::Vector3d& n, const double& w) {
        return std::make_shared<FunctorNormal>(n, w);
    }

    inline ceres::CostFunction* toCeres() {
        return new ceres::AutoDiffCostFunction<
            FunctorNormal, DimResidual, DimNormal, DimRotation>(this);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormal& f) {
        os << "Vector: " << f.n_ << std::endl << "Weight: " << f.w_ << std::endl;
        return os;
    }

    inline void setPoint(const Eigen::Vector3d& p) {
        n_ = p;
    }
    inline void setWeight(const double w) {
        w_ = w;
    }

private:
    Eigen::Vector3d n_; ///< 3D world point normal associated to this pixel
    double w_;          ///< Weight
};
}
