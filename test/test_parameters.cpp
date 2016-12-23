#include <glog/logging.h>
#include <gtest/gtest.h>

#include "parameters.hpp"

TEST(Parameters, Instantiation) {
    google::InitGoogleLogging("Parameters");
    google::InstallFailureSignalHandler();
    const mrf::Parameters p{"parameters.yaml"};
    LOG(INFO) << "Test parameters:\n" << p;
}
