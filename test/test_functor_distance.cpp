#include <glog/logging.h>
#include <gtest/gtest.h>

#include "functor_distance.hpp"

TEST(FunctorDistance, Instantiation) {

    google::InitGoogleLogging("FunctorDistance");
    google::InstallFailureSignalHandler();

    using namespace mrf;
    auto f = FunctorDistance::create(
        Eigen::Vector3d::Ones(),
        Eigen::ParametrizedLine<double, 3>(Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones()));
}
