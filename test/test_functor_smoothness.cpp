#include <glog/logging.h>
#include <gtest/gtest.h>

#include "functor_smoothness.hpp"

TEST(FunctorSmoothness, Instantiation) {

    google::InitGoogleLogging("FunctorSmoothness");
    google::InstallFailureSignalHandler();

    mrf::FunctorSmoothness f(0.5);
    std::cout << "\nTest functor smoothness:\n" << f;
}
