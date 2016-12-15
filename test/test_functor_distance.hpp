#pragma once

#include <gtest/gtest.h>

#include "functor_distance.hpp"

TEST(FunctorDistance, Instantiation) {
    using namespace mrf;
    FunctorDistance::Ptr f{FunctorDistance::create(Eigen::Vector3d::Ones(), 0.5)};
    std::cout << "\nTest functor distance:\n" << *f;
}
