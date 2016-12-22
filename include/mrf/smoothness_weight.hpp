#pragma once

#include "parameters.hpp"
#include "pixel.hpp"

namespace mrf {

inline double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const Parameters& params) {
	const double abs_diff{std::abs(p.val - neighbor.val)};
	if (abs_diff < params.discontinuity_threshold) {
		return 1;
	} else {
		return exp(-abs(abs_diff - params.discontinuity_threshold));
	}
}
}
