#pragma once


#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <eigen3/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h>

namespace mrf {

using SpMatT = Eigen::SparseMatrix<float>;
using TripT = Eigen::Triplet<float>;
enum NeighbourCases { topbottom, leftright, toplr, bottomlr };

class MrfEigenCg {
public:
    //static const SolverType ID = SolverType::EIGEN_CONJUGATE_GRADIENT;

    MrfEigenCg(){};
    MrfEigenCg(const Eigen::VectorXf& image_in, const Eigen::VectorXf& z_in,
               const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
               const int kd_in);
    MrfEigenCg(const Eigen::MatrixXf& image_in, const Eigen::Matrix3Xf& points_in,
               const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
               const int kd_in);

    void setData(const Eigen::MatrixXf& image_in, const Eigen::Matrix3Xf& points_in,
                 const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
                 const int kd_in);
    void setData(const Eigen::VectorXf& image_in, const Eigen::VectorXf& z_in,
                 const Eigen::VectorXi& indices_in, const int width_in, const int ks_in,
                 const int kd_in);

    void setParameters(const int ks, const int kd, const int neighbourhood);
    void setMaxIterations(const int);
    void setTolerance(const int);

    void setPrior(Eigen::VectorXf& prior);
    void getDepth(Eigen::VectorXf&);

private:
    void init();
    void solve();
    void setZ(const Eigen::Matrix3Xf& points_in);
    void setCostW();
    void setSmoothnessS();
    void setAandB();
    bool neighbourTest(const int p, int pnext, const NeighbourCases nc);
    float calcGrayDiff(const int i, const int j);

    Eigen::MatrixXf image_;
    Eigen::VectorXf z_;
    Eigen::VectorXi indices_;
    int z_norm_max_;
    int width_;
    int ks_;
    int kd_;
    bool prior_loaded_;
    int dim_;
    int num_seed_points_;
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
