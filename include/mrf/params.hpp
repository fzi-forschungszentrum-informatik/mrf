#pragma once
#include <memory>
#include <ostream>

namespace mrf {

enum class SolverType { EIGEN_CONJUGATE_GRADIENT = 0, CERES_ITERATIVE_SCHUR = 1 ,CERES_CGNR = 2};

class Params {
public:
    using Ptr = std::shared_ptr<Params>;

public:
    Params(const int ks = 10, const int kd = 10,
           const SolverType type = SolverType::EIGEN_CONJUGATE_GRADIENT,
           const int max_iterations = -1, const int neighbours = 4,
           const double certainty_threshhold = 0.8, const int tolerance = -1);

//    Params(const int ks = 10, const int kd = 10,
//           const SolverType type = SolverType::EIGEN_CONJUGATE_GRADIENT,
//           const int max_iterations = -1, const int neighbours = 4,
//           const double certainty_threshhold = 0.8, const int tolerance = -1):
//    ks(ks), kd(kd), solver_type(type), max_iterations(max_iterations), neighbours(neighbours),
//        certainty_threshhold(certainty_threshhold), tolerance(tolerance){};

    void setMaxIteration(const int max_iterations);
    Params& operator=(const Params& in);

    static Params::Ptr create(const int ks = 10, const int kd = 10,
                              const SolverType type = SolverType::EIGEN_CONJUGATE_GRADIENT,
                              const int max_iterations = -1, const int neighbours = 4,
                              const double certainty_threshhold = 0.8, const int tolerance = -1);

    int ks;
    int kd;
    int neighbours;
    int max_iterations;
    int tolerance = -1;
    double certainty_threshhold;
    SolverType solver_type;
private:

    friend std::ostream& operator<<(std::ostream& os, const Params& f);
};

} // end of mrf namespace
