#include <solver.hpp>
#include <stdlib.h>
#include <generic_logger/generic_logger.hpp>

namespace mrf {

Solver::Solver(const Data& data_in, const Params& params_in) {
    data_ = data_in;
    params_ = params_in;
    dim_ = data_.depth.size();
    num_certain_points = 0;
    norm_factor_ = 0;
    a_.setZero();
    s_.setZero();
    w_.setZero();
    b_.setZero();
    z_.setZero();

    DEBUG_STREAM("Data in: "<< data_);
    DEBUG_STREAM("Params in: "<< params_);

    if (params_in.solver_type == SolverType::EIGEN_CONJUGATE_GRADIENT) {
        a_.resize(dim_, dim_);
        a_.setZero();
        s_.resize(dim_, dim_);
        s_.setZero();
        w_.resize(dim_, dim_);
        w_.setZero();
        b_.setZero(dim_);
        z_.setZero(dim_);
        norm_factor_ = data_.depth.maxCoeff();
        // DEBUG_STREAM("norm Factor: " << norm_factor_);
        num_certain_points = countCertainPoints();
        if (num_certain_points == 0) {
            throw std::runtime_error("Number of certain points is zero");
        }
        DEBUG_STREAM("num certain points: " << num_certain_points);
        setW();
        setS();
        setAandB();
    }
}

Solver& Solver::operator=(Solver& mrf_in) {
    this->data_ = mrf_in.data_;
    this->params_ = mrf_in.params_;
    return *this;
}

bool Solver::solve(Data& results) {

    bool success{false};

    if (params_.solver_type == SolverType::EIGEN_CONJUGATE_GRADIENT) {
        success = solveEigen(results);
    } else {
    }

    INFO_STREAM("SUCCESSFULL SOLVING:   " << success);
    return success;
}

bool Solver::setAandB() {
    a_ = s_.transpose() * s_ + w_.transpose() * w_;
    this->b_ = w_ * w_.transpose() * z_;

    if (b_ != b_) {
        ERROR_STREAM("b_ contains nan");
    }

    DEBUG_STREAM("z_ max, min: " << z_.maxCoeff() << ", " << z_.minCoeff());
    //    DEBUG_STREAM("Non Zeros in A : " << a_.nonZeros());
    //    DEBUG_STREAM("Non Zeros in b: " << b_.count());
    //    DEBUG_STREAM("Non Zeros in z: " << z_.count());

    return true;
}

bool Solver::setW() {
    std::vector<TripT> w_triplets;
    std::vector<float> ww;

    w_triplets.reserve(num_certain_points);
    ww.reserve(num_certain_points);
    int c = 0;
    int depth_zeros = 0;
    for (int i = 0; i < dim_; i++) {
        if (data_.certainty(i) >= params_.certainty_threshhold) {
            const float value{data_.depth(i)}; ///< norm_factor_
//            if (value == 0) {
//                depth_zeros++;
//            } else {
                z_(i) = value;
                const float value2{z_(i) * params_.kd};
                w_triplets.emplace_back(i, i, value2);
                ww.emplace_back(value2);
                c++;
            //}
        }
    }
    if (z_ != z_) {
        ERROR_STREAM("z_ contains nan");
    }
    DEBUG_STREAM("w_triplets size: " << w_triplets.size());
    auto maxx = std::max_element(std::begin(ww), std::end(ww));
    auto minn = std::min_element(std::begin(ww), std::end(ww));
    DEBUG_STREAM("W_w max, min: " << *maxx << ", " << *minn);

    w_.setFromTriplets(w_triplets.begin(), w_triplets.end());
    w_.makeCompressed();

//    if ((z_.array() != 0).count() != num_certain_points - depth_zeros) {
//        ERROR_STREAM("z non zeros: " << (z_.array() != 0).count() << ", should be "
//                                     << num_certain_points - depth_zeros
//                                     << "depth_zeros: " << depth_zeros);
//        throw std::runtime_error("z  not set up correctly");
//    }
    if (w_.nonZeros() != w_triplets.size()) {
        ERROR_STREAM("w non zeros: " << w_.nonZeros() << ", should be " << w_triplets.size());
        throw std::runtime_error("W Cost Matrix not set up correctly");
    }
    return true;
}

bool Solver::setS() {
    std::vector<TripT> S_triplets;
    S_triplets.reserve((params_.neighbours + 1) * dim_);
    std::vector<float> eijs;
    int count_n = 0;
    for (int p = 0; p < dim_; p++) {
        double sum_eij = 0;

        for (int x = 1; x < 3; x++) {
            /*
             * Calc Grey Diff to neighbours right and left to pixel
             */

            int pnext_lr = p + pow(-1, x);
            int pnext_tb = p + pow(-1, x) * data_.width;

            float eij_lr{addToSTriplet(S_triplets, p, pnext_lr, NeighbourCases::leftright)};
            float eij_tb{addToSTriplet(S_triplets, p, pnext_tb, NeighbourCases::topbottom)};
            sum_eij += eij_lr;
            sum_eij += eij_tb;
            eijs.emplace_back(eij_lr);
            eijs.emplace_back(eij_tb);
            if (eij_lr == 0)
                count_n++;
            if (eij_tb == 0)
                count_n++;
        }

        if (params_.neighbours == 8) {

            for (int x = 1; x < 3; x++) {
                /*
                 * Calc Grey Diff to neighours to top left and right
                 */

                int pnext_tlr = p - data_.width + pow(-1, x);
                int pnext_blr = p + data_.width + pow(-1, x);

                float eij_tlr{addToSTriplet(S_triplets, p, pnext_tlr, NeighbourCases::toplr)};
                float eij_blr{addToSTriplet(S_triplets, p, pnext_blr, NeighbourCases::bottomlr)};

                if (eij_tlr == 0) {
                    count_n++;
                } else {
                    eijs.emplace_back(eij_tlr);
                    sum_eij += eij_tlr;
                }
                if (eij_blr == 0) {
                    count_n++;
                } else {
                    eijs.emplace_back(eij_blr);
                    sum_eij += eij_blr;
                }
            }
        }
        S_triplets.emplace_back(p, p, sum_eij);
    }

    if ((count_n != 2 * (dim_ / data_.width) + 2 * data_.width && params_.neighbours == 4) ||
        ((count_n != 2 * (dim_ / data_.width) + 2 * data_.width + 4 && params_.neighbours == 8))) {
        ERROR_STREAM("Neighbour corrected: " << count_n);
        if (params_.neighbours == 4)
            ERROR_STREAM("neigh should be: " << 2 * (dim_ / data_.width) + 2 * data_.width);
        if (params_.neighbours == 8)
            ERROR_STREAM("neigh should be: " << 2 * (dim_ / data_.width) + 2 * data_.width + 4);
        throw std::runtime_error("Neighbour Correction not correctly");
    }

    auto maxx = std::max_element(std::begin(eijs), std::end(eijs));
    auto minn = std::min_element(std::begin(eijs), std::end(eijs));

    DEBUG_STREAM("EIJ max min: " << *maxx << ", " << *minn);
    s_.setFromTriplets(S_triplets.begin(), S_triplets.end());
    s_.makeCompressed();

    if (s_.nonZeros() != S_triplets.size()) {
        throw std::runtime_error("Smoothness Matrix not set correctly");
    }
    if (s_.nonZeros() !=
        ((params_.neighbours + 1) * dim_ - 2 * (dim_ / data_.width) - 2 * data_.width)) {
        ERROR_STREAM("s nonzeros: " << s_.nonZeros());
        ERROR_STREAM("s nonzeros should be: " << ((params_.neighbours + 1) * dim_ -
                                                  2 * (dim_ / data_.width) - 2 * data_.width));
        throw std::runtime_error("Smoothness Matrix not set correctly");
    }
    return true;
}

float Solver::addToSTriplet(std::vector<TripT>& S, const int p, const int pnext,
                            const NeighbourCases nc) {
    bool neigh;
    neigh = neighbourTest(p, pnext, nc);
    if (!neigh) {
        return 0;
    } else {
        const float eij{diff(p, pnext)};
        S.emplace_back(p, pnext, -eij);
        return eij;
    }
}

bool Solver::neighbourTest(const int p, int pnext, const NeighbourCases nc) {
    if (((abs((p % data_.width) - (pnext % data_.width)) > 1) || pnext < 0) &&
        (nc == NeighbourCases::leftright || nc == NeighbourCases::bottomlr ||
         nc == NeighbourCases::toplr)) {
        /*
         * Criteria for left right border pass
         */
        pnext = p;
        return false;
    }
    if (((floor(p / data_.width) == 0) && (pnext < 0)) &&
        (nc == NeighbourCases::topbottom || nc == NeighbourCases::toplr)) {
        /*
         * Criteria for top pass
         */
        pnext = p;
        return false;
    }
    if ((pnext >= dim_) && (nc == NeighbourCases::topbottom || nc == NeighbourCases::bottomlr)) {
        /*
         * Criteria for bottom pass
         */
        pnext = p;
        return false;
    }
    return true;
}

float Solver::diff(const int i, const int j) {
    float delta = abs(data_.image(i) - data_.image(j));
    if (params_.ks == 0) {
        return 0.25;
    }
    float ks = 1 / (params_.ks);
    float eij = 1 / (params_.ks * sqrt(2 * M_1_PI)) * exp(-1 / 2 * (ks * delta) * (ks * delta));
    // return sqrt(eij);
    return sqrt(eij);
}

int Solver::countCertainPoints() {
    return (data_.certainty.array() >= params_.certainty_threshhold).count();
}

bool Solver::solveEigen(Data& results) {

    bool successfull;
    Eigen::ConjugateGradient<SpMatT> eigen_solver;
    if (params_.max_iterations > 0) {
        INFO_STREAM("Max Iterations set to " << params_.max_iterations);
        eigen_solver.setMaxIterations(params_.max_iterations);
    }
    if (params_.tolerance > 0) {
        INFO_STREAM("Tolerance set to " << params_.tolerance);
        eigen_solver.setTolerance(params_.tolerance);
    }

    eigen_solver.compute(a_);
    Eigen::ComputationInfo info{eigen_solver.info()};

    if (info != Eigen::ComputationInfo::Success) {
        ERROR_STREAM("Compute A not Successfull!!");
    }

    Eigen::VectorXf res;
    res.setZero(dim_);
    res = eigen_solver.solveWithGuess(this->b_, data_.depth);

    info = eigen_solver.info();
    if (info == Eigen::ComputationInfo::Success) {
        INFO_STREAM("Compute Successfull");
    } else if (info == Eigen::ComputationInfo::InvalidInput) {
        INFO_STREAM("Invalid input");
    } else if (info == Eigen::ComputationInfo::NumericalIssue) {
        INFO_STREAM("Numerical issue");
    } else if (info == Eigen::ComputationInfo::NoConvergence) {
        INFO_STREAM("No Convergence");
    }
    DEBUG_STREAM("MRF res max, min: " << res.maxCoeff() << ", " << res.minCoeff());

    if (res == res) {
        results.depth = res;
        results.certainty = data_.depth - results.depth;
        DEBUG_STREAM("RESULT size: " << res.size());
        DEBUG_STREAM("RESULT  Zeros: " << res.size() - res.nonZeros());
        successfull = true;
    } else {
        ERROR_STREAM("RESULT contain nan: ");
        results.depth = data_.depth;
        successfull = false;
    }

    // results.depth *= norm_factor_;
    //    DEBUG_STREAM("MRF res z  max, min: " << results.depth.maxCoeff() << ", "
    //                                         << results.depth.minCoeff());
    //        INFO_STREAM("MRF Depth result max, min: " << data_.depth.maxCoeff() << ", "
    //                                              << data_.depth.minCoeff());
    return successfull;
}

std::ostream& operator<<(std::ostream& os, const Solver& s){
	os << "Solver: "<< std::endl;
	os << "dim: "<< s.dim_<< std::endl;
	os << "num_certain_points: "<< s.num_certain_points<< std::endl;
	os << "z_ size: "<< s.z_.size() << std::endl;
	os << "z_ minmax: "<< s.z_.minCoeff()<< ", " << s.z_.maxCoeff() << std::endl;
	os << "z_ non Zeros: "<< (s.z_.array() != 0).count() << std::endl;
	os << "b_ size: "<< s.b_.size() << std::endl;
	os << "b_ minmax: "<< s.b_.minCoeff()<< ", "<< s.b_.maxCoeff() << std::endl;
	os << "b_ non Zeros: "<< (s.b_.array() != 0).count() << std::endl;

}
} // end of mrf namespace
