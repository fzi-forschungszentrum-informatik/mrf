#include <limits>
#include <solver.hpp>
#include <generic_logger/generic_logger.hpp>

namespace mrf {

Solver::Solver(const Data& data_in, const Params& params_in) {
    data_ = data_in;
    params_ = params_in;
    xd_ = data_in.depth.cast<double>();
    dim_ = data_.depth.size();
    results_string_.clear();
    DEBUG_STREAM("Data in: " << data_);
    DEBUG_STREAM("Params in: " << params_);
}

Solver& Solver::operator=(Solver& mrf_in) {
    this->data_ = mrf_in.data_;
    this->params_ = mrf_in.params_;
    return *this;
}

bool Solver::solve(Data& results, std::stringstream& res_string) {
    DEBUG_STREAM("mrf::Solver::solve");
    bool success{false};

    success = solveCeres(results);
    if (success) {
        INFO_STREAM("SUCCESSFULL SOLVING:   " << success);
    } else {
        ERROR_STREAM("SUCCESSFULL SOLVING:   " << success);
    }
    res_string << results_string_.str();
    return success;
}

float Solver::neighbourDiff(const int p, const int pnext, const NeighbourCases& nc) {
    if (((abs((p % data_.width) - (pnext % data_.width)) > 1) || pnext < 0) &&
        (nc == NeighbourCases::leftright || nc == NeighbourCases::bottomlr ||
         nc == NeighbourCases::toplr)) {
        /*
         * Criteria for left right border pass
         */
        return -1;
    }
    if (((floor(p / data_.width) == 0) && (pnext < 0)) &&
        (nc == NeighbourCases::topbottom || nc == NeighbourCases::toplr)) {
        /*
         * Criteria for top pass
         */
        return -1;
    }
    if ((pnext >= dim_) && (nc == NeighbourCases::topbottom || nc == NeighbourCases::bottomlr)) {
        /*
         * Criteria for bottom pass
         */
        return -1;
    }
    return diff(p, pnext);
}

float Solver::diff(const int i, const int j) {
    const float delta = abs(data_.image(i) - data_.image(j));

    if (delta < params_.discont_thresh) {
        return 1;
    } else {
        return 0;
    }

    if (params_.ks == 0) {
        return 1;
    }
    float ks = 1 / (params_.ks);
    float eij = 1 / (params_.ks * sqrt(2 * M_1_PI)) * exp(-1 / 2 * (ks * delta) * (ks * delta));
    // return sqrt(eij);
    return sqrt(eij);
}


bool Solver::solveCeres(Data& results) {

    DEBUG_STREAM("Solve Ceres");
    ceres::Problem problem;
    int count_wrong_neighbours{0};
    for (size_t i = 0; i < dim_; i++) {
        /**
         * Distance costs
         */
        problem.AddResidualBlock(CostFunctor::create(static_cast<double>(data_.depth(i)), params_.kd),
                                 new ceres::HuberLoss(static_cast<double>(data_.certainty(i))), &xd_(i));
                                 //ceres::ScaledLoss(new ceres::TrivialLoss,static_cast<double>(data_.certainty(i))

        /**
         *  Smoothness costs
         */
        for (size_t j = 1; j < 3; j++) {
            const signed int pnext_lr{i + pow(-1, j)};
            const signed int pnext_tb{i + pow(-1, j) * data_.width};
            const float eij_lr{neighbourDiff(i, pnext_lr, NeighbourCases::leftright)};
            const float eij_tb{neighbourDiff(i, pnext_tb, NeighbourCases::topbottom)};
            if (eij_lr != -1) { // eij_lr != 0 &&
                problem.AddResidualBlock(SmoothFunctor::create(eij_lr * params_.ks), nullptr,
                                         &xd_(i), &xd_(pnext_lr));
            } else {
                count_wrong_neighbours++;
            }

            if (eij_tb != -1) { // eij_tb != 0 &&
                problem.AddResidualBlock(SmoothFunctor::create(eij_tb * params_.ks), nullptr,
                                         &xd_(i), &xd_(pnext_tb));
            } else {
                count_wrong_neighbours++;
            }
        }
    }
    DEBUG_STREAM("Wrong Neighbours: " << count_wrong_neighbours);

    /**
     * Check parameters
     */
    std::string err_str;
    if (params_.ceres_options.IsValid(&err_str)) {
        INFO_STREAM("All Residuals set up correctly");
    } else {
        ERROR_STREAM(err_str);
    }

    /**
     * Solve problem
     */
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    params_.ceres_options.max_num_iterations = params_.max_iterations;
    params_.ceres_options.minimizer_progress_to_stdout = true;
    params_.ceres_options.num_threads = 8;
    ceres::Solve(params_.ceres_options, &problem, &summary);
    results_string_ << summary.FullReport() << std::endl;

    results.depth = xd_.cast<float>(); ///< Parse data
    return true;
}


std::ostream& operator<<(std::ostream& os, const Solver& s) {
    os << "Solver: " << std::endl;
    os << "dim: " << s.dim_ << std::endl;
    os << "Results: "<< s.results_string_.str() << std::endl;
}
} // end of mrf namespace
