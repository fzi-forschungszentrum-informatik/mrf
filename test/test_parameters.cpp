#include <glog/logging.h>
#include <gtest/gtest.h>

#include "parameters.hpp"

TEST(Parameters, Instantiation) {

    google::InitGoogleLogging("Parameters");
    google::InstallFailureSignalHandler();

    using namespace mrf;
    const Parameters p{"parameters.yaml"};
    std::cout << "\nTest parameters:\n" << p;
}
