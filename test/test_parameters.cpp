#include <glog/logging.h>
#include <gtest/gtest.h>

#include "parameters.hpp"

TEST(Parameters, Instantiation) {

    google::InitGoogleLogging("Parameters");
    google::InstallFailureSignalHandler();

    using namespace mrf;
    const Parameters::Ptr p{Parameters::create()};
    std::cout << "\nTest parameters:\n" << *p;
}
