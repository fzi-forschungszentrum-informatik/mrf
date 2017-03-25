#pragma once
#include <Eigen/Geometry>

namespace Eigen {
template <typename T>
using Affine3 = Transform<T, 3, Affine>;

template <typename T>
using Vector3 = Matrix<T, 3, 1>;

template <typename T>
using MatrixX = Matrix<T, Dynamic, Dynamic>;

template <typename T>
using Matrix3X = Matrix<T, 3, Dynamic>;
}


namespace util_ceres {

template <typename T, typename U>
inline void toQuaternion(const Eigen::Affine3<T>& transform, U* q) {
    const Eigen::Quaternion<U> q2{transform.rotation().template cast<U>()};
    q[0] = q2.x();
    q[1] = q2.y();
    q[2] = q2.z();
    q[3] = q2.w();
}
template <typename T, typename U = T>
inline Eigen::Quaternion<U> toQuaternion(const Eigen::Affine3<T>& transform) {
    return Eigen::Quaternion<U>{transform.rotation().template cast<U>()};
}

template <typename T, typename U>
inline void toTranslation(const Eigen::Affine3<T>& transform, U* t) {
    Eigen::Map<Eigen::Vector3<U>>(t, 3) = transform.translation().template cast<U>();
}
template <typename T, typename U = T>
inline Eigen::Vector3<U> toTranslation(const Eigen::Affine3<T>& transform) {
    return transform.translation().template cast<U>();
}

template <typename T, typename U>
inline void toQuaternionTranslation(const Eigen::Affine3<T>& transform, U* q, U* t) {
    toQuaternion(transform, q);
    toTranslation(transform, t);
}
template <typename T, typename U>
inline void toQuaternionTranslation(const Eigen::Affine3<T>& transform,
                                    Eigen::Quaternion<U>& q,
                                    Eigen::Vector3<U>& t) {
    q = toQuaternion<T, U>(transform);
    t = toTranslation<T, U>(transform);
}

template <typename T, typename U = T>
inline Eigen::Affine3<U> fromQuaternion(const T* const q) {
    using namespace Eigen;
    Affine3<U> transform_eigen{Map<const Quaternion<T>>(q).template cast<U>()};
    transform_eigen.translation() = Vector3<U>::Zero();
    return transform_eigen;
}

template <typename T, typename U = T>
inline Eigen::Affine3<U> fromQuaternionTranslation(const T* const q, const T* const t) {
    using namespace Eigen;
    Affine3<U> transform_eigen{Eigen::Quaternion<T>(q).template cast<U>()};
    transform_eigen.translation() = Map<const Vector3<T>>(t).template cast<U>();
    return transform_eigen;
}
template <typename T, typename U = T>
inline Eigen::Affine3<U> fromQuaternionTranslation(const Eigen::Quaternion<T>& q,
                                                   const Eigen::Vector3<T>& t) {
    Eigen::Affine3<U> transform_eigen{q.template cast<U>()};
    transform_eigen.translation() = t.template cast<U>();
    return transform_eigen;
}

// template <typename T, typename U>
// inline void toAngleAxis(const Eigen::Affine3<T>& transform_eigen, U* angle_axis_ceres) {
//    Eigen::Map<Eigen::AngleAxis<U>>(angle_axis_ceres, 3) =
//        transform_eigen.rotation().template cast<U>();
//}
//
// template <typename T, typename U>
// inline void toAngleAxisTranslation(const Eigen::Affine3<T>& transform_eigen, U* angle_axis_ceres,
//                                   U* t_ceres) {
//    toAngleAxis(transform_eigen, angle_axis_ceres);
//    toTranslation(transform_eigen, t_ceres);
//}
//
// template <typename T>
// inline Eigen::Affine3<T> fromAngleAxisTranslation(const T* const angle_axis_ceres,
//                                                  const T* const t_ceres) {
//    Eigen::Affine3<T> transform_eigen{Eigen::Map<const Eigen::AngleAxis<T>>(angle_axis_ceres)};
//    transform_eigen.translation() = Eigen::Map<const Eigen::Vector3<T>>(t_ceres);
//    return transform_eigen;
//}
}
