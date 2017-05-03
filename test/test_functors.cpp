#include <glog/logging.h>
#include <gtest/gtest.h>

#include "functor_distance.hpp"
#include "functor_smoothness_distance.hpp"

TEST(FunctorDistance, Instantiation) {

    google::InitGoogleLogging("FunctorDistance");
    google::InstallFailureSignalHandler();

    using namespace Eigen;
    using namespace mrf;
    FunctorDistance a(Vector3d::Ones(),
                      ParametrizedLine<double, 3>(Vector3d::Ones(), Vector3d::Ones()));
    FunctorSmoothnessDistance b();
}
