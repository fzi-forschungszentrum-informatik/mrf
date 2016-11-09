#include <params.hpp>

namespace mrf {
Params::Params(const int ks_in, const int kd_in, const SolverType type, const int max_iterations_in,
               const int neighbours_in, const double certainty_threshhold_in,
               const int tolerance_in)
        : ks(ks_in), kd(kd_in), solver_type(type), max_iterations(max_iterations_in),neighbours(neighbours_in),certainty_threshhold(certainty_threshhold_in),tolerance(tolerance_in) {

    if (!(neighbours_in == 4 || neighbours_in == 8))
        neighbours = 4;
    if (tolerance_in <= 0) {
        tolerance = -1;
    }
    if (max_iterations_in <= 0) {
        max_iterations = -1;
    }
    if(ks <=0){
    	ks = 10;
    }

}

Params& Params::operator=(const Params& in) {
    ks = in.ks;
    kd = in.kd;
    neighbours = in.neighbours;
    max_iterations = in.max_iterations;
    tolerance = in.tolerance;
    solver_type = in.solver_type;
    certainty_threshhold = in.certainty_threshhold;
    return *this;
}

Params::Ptr Params::create(const int ks_in, const int kd_in, const SolverType type,
                           const int max_iterations_in, const int neighbours_in,
                           const double certainty_threshhold_in, const int tolerance_in) {
    return std::make_shared<Params>(ks_in, kd_in, type, neighbours_in, max_iterations_in,
                                    tolerance_in, certainty_threshhold_in);
}

void Params::setMaxIteration(const int max_iterations) {
    this->max_iterations = max_iterations;
}
//
std::ostream& operator<<(std::ostream& os, const Params& p) {
    os << "Parameters: " << std::endl;
    os << "ks: " << p.ks << std::endl;
    os << "kd: " << p.kd << std::endl;
    os << "max_iterations: " << p.max_iterations << std::endl;
    os << "tolerance: " << p.tolerance << std::endl;
}

} // end of mrf namespace
