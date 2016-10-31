#pragma once

#include <memory>

namespace mrf{

enum class SolverType {
	EIGEN_CONJUGATE_GRADIENT,
	CERES_NON_LINEAR_LEAST_SQUARE };

class MrfSolver {
public:

	//MrfSolver(MrfData in){};
	//MrfSolver(MrfData&) = delete;
	//virtual SolverType getID() const = 0;


};

}
