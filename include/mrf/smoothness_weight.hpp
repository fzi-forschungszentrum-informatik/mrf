#pragma once

#include "pixel.hpp"

namespace mrf {

inline double smoothnessWeight(const Pixel& p, const Pixel& neighbor, const float val_p,
                               const float val_neighbor) {
//    return exp(-abs(val_p - val_neighbor));
	double delta{abs(val_p-val_neighbor)};
	LOG(INFO) << "val_p: " << val_p << " val_neigh: "<< val_neighbor << " delta: " << delta;
	if(delta < 0.3){
		LOG(INFO) << 1;
		return 1;
	}else{
		LOG(INFO) << 0;
		return 0;
	}
//    return val_p < 0.7;
}
}
