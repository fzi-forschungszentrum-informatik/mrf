#include "mrf_solver_eigen_cg.hpp"

namespace mrf {


MrfSolverEigenCg::MrfSolverEigenCg(MrfDataEigenCg&& in){
	data_ = in.clone();
	dim_ = data_.z.size();
	num_seed_points_ = data_.indices.size();
	init();
}
MrfSolverEigenCg::MrfSolverEigenCg(MrfDataEigenCg& in){
	data_ = in;
	dim_ = data_.z.size();
	num_seed_points_ = data_.indices.size();
	init();
}


//SolverType MrfSolverEigenCg::getId() const {
//    return ID;
//}

void MrfSolverEigenCg::init() {
    setCostW();
    setSmoothnessS();
    setAandB();
}

void MrfSolverEigenCg::setParameters(const int ks, const int kd, const int neighbourhood) {
    data_.ks = ks;
    data_.kd = kd;
    if(neighbourhood == 4 || neighbourhood == 8)  neighbourhood_ = neighbourhood;
    num_solving_ = 1;
}

void MrfSolverEigenCg::setMaxIterations(const int maxit) {
    max_iterations_ = maxit;
}

void MrfSolverEigenCg::setTolerance(const int tol) {
    tolerance_ = tol;
}

void MrfSolverEigenCg::setPrior(Eigen::VectorXf& prior) {

	x_ = prior;
    prior_loaded_ = true;
}

void MrfSolverEigenCg::getDepth(Eigen::VectorXf& depths) {
    solve();
    depths = x_;
}

void MrfSolverEigenCg::solve() {
    if (max_iterations_ > 0) {
        cg.setMaxIterations(max_iterations_);
    }
    if (tolerance_ > 0 ) {
        cg.setTolerance(tolerance_);
    }
    cg.compute(a_);

    int i = 0;
    while (num_solving_ > i) {
        if (!(prior_loaded_ && i == 0)) {
            x_ = cg.solve(b_);
        } else {
            x_ = cg.solveWithGuess(b_, x_);
        }
        i++;
    }
}

void MrfSolverEigenCg::setAandB() {
    a_ = s_.transpose() * s_ + w_.transpose() * w_;
    b_ = w_ * w_.transpose() * data_.z;
}

void MrfSolverEigenCg::setCostW() {
    std::vector<TripT> w_triplets;
    w_triplets.resize(num_seed_points_);
    for (int i = 0; i < num_seed_points_; i++) {
        const int& index =data_.indices[i];
        w_triplets[i] = TripT(index, index, data_.z(index) * data_.kd);
    }
    w_.setFromTriplets(w_triplets.begin(), w_triplets.end());
    w_.makeCompressed();
}

void MrfSolverEigenCg::setSmoothnessS() {
    std::vector<TripT> S_triplets;
    S_triplets.reserve(5 * dim_);
    for (int p = 0; p < dim_; p++) {
        NeighbourCases nc;
        double sum_eij = 0;

        for (int x = 1; x < 3; x++) {
            /*
             * Calc Grey Diff to neighbours right and left to pixel
             */
            bool neigh;
            nc = NeighbourCases::leftright;
            int pnext = p + pow(-1, x);
            neigh = neighbourTest(p, pnext, nc);
            const float eij = calcGrayDiff(p, pnext);

            if (neigh) {
                S_triplets.push_back(TripT(p, pnext, -eij));
                sum_eij += eij;
            }
        }
        for (int x = 1; x < 3; x++) {
            /*
             * Calc Grey Diff to neighbours on top and to bottom,
             */
            bool neigh;
            nc = NeighbourCases::topbottom;
            int pnext = p + pow(-1, x) * data_.width;
            neigh = neighbourTest(p, pnext, nc);
            const float eij = calcGrayDiff(p, pnext);
            if (neigh) {
                S_triplets.push_back(TripT(p, pnext, -eij));
                sum_eij += eij;
            }
        }
        if (neighbourhood_ == 8) {
            for (int x = 1; x < 3; x++) {
                /*
                 * Calc Grey Diff to neighours to top left and right
                 */
                bool neigh;
                nc = NeighbourCases::toplr;
                int pnext = p - data_.width + pow(-1, x);
                neigh = neighbourTest(p, pnext, nc);
                const float eij = calcGrayDiff(p, pnext);
                if (neigh) {
                    S_triplets.push_back(TripT(p, pnext, -eij));
                    sum_eij += eij;
                }
            }
            for (int x = 1; x < 3; x++) {
                /*
                 * Calc Grey Diff to neighours to bottom left and right
                 */
                bool neigh;
                nc = NeighbourCases::bottomlr;
                int pnext = p + data_.width + pow(-1, x);
                neigh = neighbourTest(p, pnext, nc);
                const float eij = calcGrayDiff(p, pnext);
                if (neigh) {
                    S_triplets.push_back(TripT(p, pnext, -eij));
                    sum_eij += eij;
                }
            }
        }
        S_triplets.push_back(TripT(p, p, sum_eij));
    }
    s_.setFromTriplets(S_triplets.begin(), S_triplets.end());
    s_.makeCompressed();
}

bool MrfSolverEigenCg::neighbourTest(const int p, int pnext, const NeighbourCases nc) {
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

float MrfSolverEigenCg::calcGrayDiff(const int i, const int j) {
    float delta = abs((data_.image(i) - data_.image(j)));
    float eij = exp(-data_.ks * sqrt((delta * delta)));
    return sqrt(eij);
}
}
