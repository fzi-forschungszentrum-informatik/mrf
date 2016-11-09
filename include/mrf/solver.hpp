#pragma once

#include "data.hpp"
#include "params.hpp"

#include <memory>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <eigen3/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h>


namespace mrf{



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
    bool solve(Data& results);

    inline static Solver::Ptr create(const Data& data_in, const Params params_in) {
    	return std::make_shared<Solver>(data_in, params_in);
    }

private:
    Data data_;
    Params params_;
    int dim_;

    SpMatT a_;
    SpMatT w_;
    SpMatT s_;
    Eigen::VectorXf b_;
    Eigen::VectorXf z_;
    float norm_factor_;
    int num_certain_points;

    bool solveEigen(Data& results);
    bool initEigen();
    bool setW();
    bool setS();
    bool setAandB();
    bool neighbourTest(const int p, int pnext, const NeighbourCases nc);
    float diff(const int i, const int j);
    float addToSTriplet(std::vector<TripT>& S, const int p, const int pnext, const NeighbourCases nc);
    int countCertainPoints();
    friend std::ostream& operator<<(std::ostream& os, const Solver& s);

};

}//end of mrf namespace

