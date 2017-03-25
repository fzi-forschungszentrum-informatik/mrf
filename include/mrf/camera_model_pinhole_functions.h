#ifndef CAMERAMODELPINHOLEFUNCTIONS_H
#define CAMERAMODELPINHOLEFUNCTIONS_H

#include <boost/geometry.hpp>

template <typename T>
inline T sinc(const T x) {
    const auto taylor_0_bound = std::numeric_limits<double>::epsilon();
    const auto taylor_2_bound = std::sqrt(taylor_0_bound);
    const auto taylor_n_bound = std::sqrt(taylor_2_bound);

    auto abs_x = (x >= T(0.0)) ? x : -x;

    if (abs_x >= taylor_n_bound) {
        return sin(x) / x;
    } else {
        // approximation by taylor series in x at 0 up to order 0
        T result = T(1.0);

        if (abs_x >= T(taylor_0_bound)) {
            T x2 = x * x;

            // approximation by taylor series in x at 0 up to order 2
            result -= x2 / T(6.0);

            if (abs_x >= T(taylor_2_bound)) {
                // approximation by taylor series in x at 0 up to order 4
                result += (x2 * x2) / T(120.0);
            }
        }

        return result;
    }
}


struct CameraModelPinholeGetViewingRay {
public:
    template <typename T, typename T1, typename T2, typename T3, typename T4>
    bool operator()(T, T1&& intrinsics, T2&& imagePoint, T3&& pos, T4&& direction) const {

        auto* f = &intrinsics[0];
        auto* cc = &intrinsics[1];

        direction[0] = (imagePoint[0] - cc[0]) / f[0];
        direction[1] = (imagePoint[1] - cc[1]) / f[0];
        direction[2] = T(1.0);

        T norm = sqrt(direction[0] * direction[0] + direction[1] * direction[1] +
                      direction[2] * direction[2]);

        direction[0] /= norm;
        direction[1] /= norm;
        direction[2] /= norm;

        pos[0] = T(0.0);
        pos[1] = T(0.0);
        pos[2] = T(0.0);

        return true;
    }
};

struct CameraModelPinholeGetImagePoint {
public:
    template <typename T, typename T1, typename T2, typename T3>
    bool operator()(T, T1&& intrinsics, T2&& point3d, T3&& imagePoint) const {

        // Check if point is on the wrong side of the image plane
        if (point3d[2] <= static_cast<T>(0))
            return false;

        auto* f = &intrinsics[0];
        auto* cc = &intrinsics[1];

        imagePoint[0] = point3d[0] / point3d[2] * f[0] + cc[0];
        imagePoint[1] = point3d[1] / point3d[2] * f[0] + cc[1];
        return true;
    }
};


#endif // CAMERAMODELPINHOLEFUNCTIONS_H
