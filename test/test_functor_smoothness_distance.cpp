#include <functor_smoothness_distance.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(FunctorSmoothnessDistance, Instantiation) {

    google::InitGoogleLogging("FunctorSmoothness");
    google::InstallFailureSignalHandler();

    mrf::FunctorSmoothnessDistance f();
}
