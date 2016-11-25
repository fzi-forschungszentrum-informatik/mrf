#pragma once
#include <memory>
#include <ostream>
#include <ceres/ceres.h>

namespace mrf {

enum class SolverType { EIGEN_CONJUGATE_GRADIENT = 0, CERES_ITERATIVE_SCHUR = 1 ,CERES_CGNR = 2,CERES_SPARSE_NORMAL_CHOLESKY = 3};

class Params {
public:
    using Ptr = std::shared_ptr<Params>;

public:
    Params(const double ks = 1, const double kd = 1,const double discont_threh = 20,
           const SolverType type = SolverType::CERES_SPARSE_NORMAL_CHOLESKY,
           const int max_iterations = -1, const int neighbours = 4,
           const double certainty_threshhold = 0.8, const int tolerance = -1);

    void setCeresOptions(const ceres::Solver::Options& options_in);


    void setMaxIteration(const int max_iterations);
    Params& operator=(const Params& in);

    static Params::Ptr create(const double ks = 1, const double kd = 1,const double discont_threh = 20,
                              const SolverType type = SolverType::CERES_SPARSE_NORMAL_CHOLESKY,
                              const int max_iterations = -1, const int neighbours = 4,
                              const double certainty_threshhold = 0.8, const int tolerance = -1);

    double ks;
    double kd;
    double discont_thresh;
    int neighbours;
    int max_iterations;
    int tolerance = -1;
    double certainty_threshhold;
    ceres::Solver::Options ceres_options;

    SolverType solver_type;
private:

    friend std::ostream& operator<<(std::ostream& os, const Params& f);
};

} // end of mrf namespace
