#pragma once

#include "mrf_solver.hpp"
#include "mrf_data.hpp"

namespace mrf {

class Mrf {
public:
	std::unique_ptr<MrfSolver> mrfSolver;

    Mrf(Mrf&& m) : mrfSolver(std::move(m.mrfSolver)){};
    Mrf(){};
};


//Mrf createMrf(const SolverType& solvertype, MrfData& data);
//std::unique_ptr<MrfSolver> getMrfSolver(const SolverType& solvertype,MrfData& data);

} // end of mrf namespace
