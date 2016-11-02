#include <mrf_eigen_cg.hpp>
#include <generic_logger/generic_logger.hpp>

namespace mrf {

MrfEigenCg::MrfEigenCg(const Eigen::VectorXf& image_in, const Eigen::VectorXf& z_in,
                       const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
                       const int kd_in) {
    setData(image_in, z_in, indices_in, width_in, ks_in, kd_in);
}
MrfEigenCg::MrfEigenCg(const Eigen::MatrixXf& image_in, const Eigen::Matrix3Xf& points_in,
                       const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
                       const int kd_in) {
    setData(image_in, points_in, indices_in, width_in, ks_in, kd_in);
}

void MrfEigenCg::setData(const Eigen::VectorXf& image_in, const Eigen::VectorXf& z_in,
                         const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
                         const int kd_in) {
    if (!(ks_in > 0 && kd_in > 0))
        throw std::runtime_error("Invalid ks kd inputs ");
    if (!(width_in > 0 && width_in < image_in.size()))
        throw std::runtime_error("Invalid MrfData width/image input ");
    if (!(indices_in.size() > 0))
        throw std::runtime_error("Invalid MrfData indices");
    if (!(image_in.size() > indices_in.size()))
        throw std::runtime_error("Invalid MrfData image/indices input");

    if (image_in.size() == z_in.cols() && indices_in.size() == z_in.nonZeros()) {
        image_ = image_in;
        z_ = z_in;
    } else if (indices_in.size() == z_in.cols()) {
        image_ = image_in;
        z_.setZero(image_in.size());
        for (int i = 0; i < indices_in.size(); i++) {
            z_(i) = z_in(i);
        }
    } else {
        throw std::runtime_error("Invalid MrfDataEigenCg Inputs ");
    }
    image_ = image_in;
    indices_ = indices_in;
    width_ = width_in;
    ks_ = ks_in;
    kd_ = kd_in;
    num_seed_points_ = indices_.size();
    dim_ = image_.size();
    init();
}

void MrfEigenCg::setData(const Eigen::MatrixXf& image_in, const Eigen::Matrix3Xf& points_in,
                         const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
                         const int kd_in) {
    if (!(ks_in > 0 && kd_in > 0))
        throw std::runtime_error("Invalid ks kd inputs ");
    if (!(width_in > 0 && width_in < image_in.size()))
        throw std::runtime_error("Invalid MrfData width/image input ");
    if (!(indices_in.size() > 0))
        throw std::runtime_error("Invalid MrfData indices");
    if (!(image_in.size() > indices_in.size()))
        throw std::runtime_error("Invalid MrfData image/indices input");

    image_ = image_in;
    indices_ = indices_in;
    width_ = width_in;
    ks_ = ks_in;
    kd_ = kd_in;
    num_seed_points_ = indices_.size();
    dim_ = image_in.size();
    setZ(points_in);
    init();
}

// SolverType MrfSolverEigenCg::getId() const {
//    return ID;
//}

void MrfEigenCg::init() {
    DEBUG_STREAM("Init");
    setCostW();
    setSmoothnessS();
    setAandB();
}

void MrfEigenCg::setParameters(const int ks_in, const int kd_in, const int neighbourhood) {
    if (!(ks_in > 0 && kd_in > 0)) {
        ERROR_STREAM("Ks=  " << ks_in << ", kd= " << kd_in);
        throw std::runtime_error("Invalid ks kd inputs ");
    }

    ks_ = ks_in;
    kd_ = kd_in;
    if (neighbourhood == 4 || neighbourhood == 8)
        neighbourhood_ = neighbourhood;
    num_solving_ = 1;
}

void MrfEigenCg::setMaxIterations(const int maxit) {
    if (!(maxit > 0))
        throw std::runtime_error("Max Iterations must be greater than 0 ");
    max_iterations_ = maxit;
}

void MrfEigenCg::setTolerance(const int tol) {
    if (!(tol > 0))
        throw std::runtime_error("Tolerance must be greater than 0 ");
    tolerance_ = tol;
}

void MrfEigenCg::setPrior(Eigen::VectorXf& prior) {
    x_ = prior;
    prior_loaded_ = true;
}

void MrfEigenCg::getDepth(Eigen::VectorXf& depths) {
    solve();
    depths = x_;
}

void MrfEigenCg::solve() {
    DEBUG_STREAM("solve");
    if (max_iterations_ > 0) {
        cg.setMaxIterations(max_iterations_);
    }
    if (tolerance_ > 0) {
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

    x_ = x_ * z_norm_max_;
    DEBUG_STREAM("x_ min,max : " << x_.maxCoeff() << ", min: " << x_.minCoeff());
}

void MrfEigenCg::setAandB() {
    DEBUG_STREAM("set a and b");
    a_.resize(dim_,dim_);
    a_.setZero();
    b_.setZero(dim_);

    a_ = s_.transpose() * s_ + w_.transpose() * w_;
    b_ = w_ * w_.transpose() * z_;

    DEBUG_STREAM("Non Zeros in A : " << a_.nonZeros());
    DEBUG_STREAM("Non Zeros in b: " << b_.count());
}

void MrfEigenCg::setZ(const Eigen::Matrix3Xf& points_in) {
    z_.setZero(dim_);
    for (int i = 0; i < indices_.size(); i++) {
        const int& index = indices_(i);
        if (index > z_.size())
            throw std::runtime_error("Index > z_ ");
        z_(index) = points_in.col(i).norm();
    }
    DEBUG_STREAM("z non zeros: "<< z_.count());
    DEBUG_STREAM("z max min "<< z_.maxCoeff() <<", "<< z_.minCoeff());
    z_norm_max_ = z_.maxCoeff();
    z_ = z_ / z_norm_max_;
}

void MrfEigenCg::setCostW() {
    DEBUG_STREAM("set Cost");
    w_.resize(dim_, dim_);
    w_.setZero();
    int maximum{indices_.col(0).maxCoeff()};
    DEBUG_STREAM("max index: " << maximum);
    std::vector<TripT> w_triplets;
    w_triplets.resize(indices_.size());
    for (int i = 0; i < indices_.size(); i++) {
        const int& index = indices_(i);
        w_triplets[i] = TripT(index, index, z_(index) * kd_);
    }
    w_.setFromTriplets(w_triplets.begin(), w_triplets.end());
    w_.makeCompressed();
    DEBUG_STREAM("w non zeros: "<< w_.nonZeros());
}

void MrfEigenCg::setSmoothnessS() {
    DEBUG_STREAM("set smooth");
    DEBUG_STREAM("Image max min: "<< image_.maxCoeff() << ", "<< image_.minCoeff());
    s_.resize(dim_, dim_);
    s_.setZero();
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
            int pnext = p + pow(-1, x) * width_;
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
                int pnext = p - width_ + pow(-1, x);
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
                int pnext = p + width_ + pow(-1, x);
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
    DEBUG_STREAM("s nonzeros "<< s_.nonZeros());
}

bool MrfEigenCg::neighbourTest(const int p, int pnext, const NeighbourCases nc) {
    if (((abs((p % width_) - (pnext % width_)) > 1) || pnext < 0) &&
        (nc == NeighbourCases::leftright || nc == NeighbourCases::bottomlr ||
         nc == NeighbourCases::toplr)) {
        /*
         * Criteria for left right border pass
         */
        pnext = p;
        return false;
    }
    if (((floor(p / width_) == 0) && (pnext < 0)) &&
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

float MrfEigenCg::calcGrayDiff(const int i, const int j) {
    float delta = abs((image_(i) - image_(j)));
    float eij = exp(-ks_ * sqrt((delta * delta)));
    return sqrt(eij);
}
}
