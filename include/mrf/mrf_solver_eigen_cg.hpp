#pragma once

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <eigen3/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h>
#include <mrf_data_eigen_cg.hpp>
#include "mrf_solver.hpp"

namespace mrf {

using SpMatT = Eigen::SparseMatrix<float>;
using TripT = Eigen::Triplet<float>;
enum NeighbourCases { topbottom, leftright, toplr, bottomlr };

class MrfSolverEigenCg : public MrfSolver {
public:
    static const SolverType ID = SolverType::EIGEN_CONJUGATE_GRADIENT;

    MrfSolverEigenCg(MrfDataEigenCg& in);// = delete;
    MrfSolverEigenCg(MrfDataEigenCg&& in);

    void setParameters(const int ks, const int kd,const int neighbourhood);
    void setMaxIterations(const int);
    void setTolerance(const int);

    void setPrior(Eigen::VectorXf& prior);
    void getDepth(Eigen::VectorXf&);
    //virtual SolverType getId() const override;

private:
    void init();
    void solve();
    void setCostW();
    void setSmoothnessS();
    void setAandB();
    bool neighbourTest(const int p,int pnext, const NeighbourCases nc );
    float calcGrayDiff(const int i, const int j);


    MrfDataEigenCg data_;
    bool prior_loaded_;
    int dim_;
    int num_seed_points_;
    int ks_;
    int kd_;
    int neighbourhood_ = 4;

    Eigen::ConjugateGradient<SpMatT> cg;
    int max_iterations_ = -1;
    float tolerance_ = -1;
    int num_solving_ = 1;
    SpMatT a_;
    SpMatT w_;
    SpMatT s_;
    Eigen::VectorXf b_;
    Eigen::VectorXf x_;

};
}
