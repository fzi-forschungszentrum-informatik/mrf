#pragma once

#include "data.hpp"
#include "params.hpp"
#include "functors.hpp"

#include <memory>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <ceres/ceres.h>
#include <eigen3/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h>

namespace mrf {

struct CostFunctor;
struct SmoothFunctor;

class Solver {

public:
    using Ptr = std::shared_ptr<Solver>;

private:
    using SpMatT = Eigen::SparseMatrix<float>;
    using TripT = Eigen::Triplet<float>;
    enum NeighbourCases { topbottom, leftright, toplr, bottomlr };

public:
    Solver(const Data&, const Params& = Params());
    //    Mrf(){};
    Solver& operator=(Solver& mrf_in);
    bool solve(Data& res_data, std::stringstream& results);

    inline static Solver::Ptr create(const Data& data_in, const Params params_in) {
        return std::make_shared<Solver>(data_in, params_in);
    }

private:
    Data data_;
    Params params_;
    int dim_;
    std::stringstream results_string_;
    Eigen::VectorXf xf_;
    Eigen::VectorXd xd_;

    SpMatT a_;
    SpMatT w_;
    SpMatT s_;
    Eigen::VectorXf b_;
    Eigen::VectorXf z_;
    float norm_factor_;
    int num_certain_points;




    bool solveEigen(Data& results);
    bool initEigen();
    bool setZ();
    bool setW();
    bool setS();
    bool setAandB();
    float neighbourDiff(const int p, int pnext, const NeighbourCases nc);
    float diff(const int i, const int j);

    int countCertainPoints();

    bool solveCeres(Data& results);


    friend std::ostream& operator<<(std::ostream& os, const Solver& s);
};



} // end of mrf namespace
