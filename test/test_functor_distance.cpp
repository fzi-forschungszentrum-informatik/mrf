#include <glog/logging.h>
#include <gtest/gtest.h>

#include "functor_distance.hpp"

TEST(FunctorDistance, Instantiation) {

    google::InitGoogleLogging("FunctorDistance");
    google::InstallFailureSignalHandler();

    using namespace mrf;
//    FunctorDistance::Ptr f{FunctorDistance::create(
//        Eigen::Vector3d::Ones(), 0.5,
//        Eigen::ParametrizedLine<double, 3>(Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones()))};
//    std::cout << "\nTest functor distance:\n" << *f;
}
