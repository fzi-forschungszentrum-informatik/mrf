#pragma once

#include <gtest/gtest.h>

#include "parameters.hpp"

TEST(Parameters, Instantiation) {
    using namespace mrf;
    const Parameters::Ptr p{Parameters::create()};
    std::cout << "\nTest parameters:\n" << *p;
}
