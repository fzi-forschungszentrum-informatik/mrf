#pragma once

#include <gtest/gtest.h>

#include "functor_smoothness.hpp"

TEST(FunctorSmoothness, Instantiation) {
    mrf::FunctorSmoothness f(0.5);
    std::cout << "\nTest functor smoothness:\n" << f;
}
