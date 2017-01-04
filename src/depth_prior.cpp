#include "depth_prior.hpp"

namespace mrf {

using DataType = double;
using Point = pcl_ceres::Point<DataType>;
using PixelMapT = std::map<Pixel, Point, PixelLess>;
using RayMapT = std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>;
using treeT = std::unique_ptr<flann::Index<flann::L2_Simple<DataType>>>;
using EigenT = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DistanceType = flann::L2_Simple<DataType>;

flann::Matrix<DataType> convertEigen2FlannRow(const EigenT& mEigen) {
    flann::Matrix<DataType> mFlann(new DataType[mEigen.size()], mEigen.rows(), mEigen.cols());

    for (size_t n = 0; n < (unsigned)mEigen.size(); ++n) {
        *(mFlann.ptr() + n) = *(mEigen.data() + n);
    }
    return mFlann;
}

bool insideTriangle(const Pixel& p, const Eigen::Vector2i& first, const Eigen::Vector2i& second,
                    const Eigen::Vector2i& third) {
    const Eigen::Vector2i& P{p.col, p.row};
    const Eigen::Vector2i& AC{first - third};
    const Eigen::Vector2i& BA{second - first};
    const Eigen::Vector2i& CB{third - second};
    const Eigen::Vector2i& PA{P - first};
    const Eigen::Vector2i& PB{P - second};
    const Eigen::Vector2i& PC{P - third};

    const int u{(BA[0] * PA[1] - (BA[1] * PA[0]))};
    const int v{(BA[0] * (-1 * AC[1])) - (BA[1] * (-1 * AC[0]))};
    if (u * v >= 0) {
        const int u2{(AC[0] * PC[1] - (AC[1] * PC[0]))};
        const int v2{(AC[0] * (-1 * CB[1])) - (AC[1] * (-1 * CB[0]))};
        if (u2 * v2 >= 0) {
            const int u3{(CB[0] * PB[1] - (CB[1] * PB[0]))};
            const int v3{(CB[0] * (-1 * BA[1])) - (CB[1] * (-1 * BA[0]))};
            if (u3 * v3 >= 0) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int> getTriangleNeighbours(std::vector<int>& neighbours_in,
                                       const Eigen::Matrix2Xi& coordinates, const Pixel& p) {
    if (neighbours_in.size() < 3) {
        return std::vector<int>{-1, -1, -1};
    }
    std::vector<int> neighbours{-1, -1, -1};
    size_t i{0};
    while (i < (neighbours_in.size() - 2)) {
        const Eigen::Vector2i& first_coorinate{coordinates(0, neighbours_in[i]),
                                               coordinates(1, neighbours_in[i])};
        neighbours[0] = neighbours_in[i];
        size_t j{1 + i};
        while (j < (neighbours_in.size() - 1)) {
            const Eigen::Vector2i& second_coorinate{coordinates(0, neighbours_in[j]),
                                                    coordinates(1, neighbours_in[j])};
            neighbours[1] = neighbours_in[j];
            size_t n{j + 1};
            while (n < (neighbours_in.size())) {
                const Eigen::Vector2i& third_coorinate{coordinates(0, neighbours_in[n]),
                                                       coordinates(1, neighbours_in[n])};
                neighbours[2] = neighbours_in[n];
                if (insideTriangle(p, first_coorinate, second_coorinate, third_coorinate)) {
                    return neighbours;
                }
                n++;
            }
            j++;
        }
        i++;
    }
    return std::vector<int>{-1, -1, -1};
}

std::vector<int> getNeighbours(const Eigen::Matrix2Xi& coordinates, const treeT& tree,
                               const Pixel& p, const int num_neigh) {
    DistanceType::ElementType queryData[] = {static_cast<DistanceType::ElementType>(p.col),
                                             static_cast<DistanceType::ElementType>(p.row)};

    const flann::Matrix<DistanceType::ElementType> query(queryData, 1, 2);
    std::vector<std::vector<int>> indices_vec;
    std::vector<std::vector<DataType>> dist_vec;
    tree->knnSearch(query, indices_vec, dist_vec, num_neigh, flann::SearchParams(32));
    return indices_vec[0];
}

double pointIntersection(const Eigen::ParametrizedLine<double, 3>& ray,
                         const Eigen::Matrix3Xd& neighbours) {
    const Eigen::Vector3d& q0_world{neighbours.col(0)};
    const Eigen::Vector3d& q1_world{neighbours.col(1)};
    const Eigen::Vector3d& q2_world{neighbours.col(2)};

    Eigen::Vector3d direction_1_norm{q0_world - q1_world}; //.normalized();
    Eigen::Vector3d direction_2_norm{q2_world - q1_world}; //.normalized();

    Eigen::Vector3d normal = (direction_1_norm.cross(direction_2_norm)).normalized();

    Eigen::Hyperplane<double, 3> plane(normal, q0_world);
    Eigen::Vector3d p_int{ray.intersectionPoint(plane)};
    return (p_int - ray.origin()).norm();
}

void addSeedPoints(const RayMapT& rays, const PixelMapT& projection, Eigen::MatrixXd& depth_est,
                   Eigen::MatrixXd& certainty) {
    for (auto const& el : projection) {
        const Pixel& p{el.first};
        certainty(p.row, p.col) = 1;
        const Eigen::ParametrizedLine<double, 3>& ray{rays.at(p)};
        const Eigen::Hyperplane<double, 3> plane(ray.direction(), el.second.position);
        depth_est(p.row, p.col) =
            (rays.at(p).intersectionPoint(plane) - rays.at(p).origin()).norm();
    }
}

void getDepthEst(const RayMapT& rays, const PixelMapT& projection, const size_t& rows,
                 const size_t& cols, const Parameters::Initialization type,
                 const int neighborsearch, Eigen::MatrixXd& depth_est, Eigen::MatrixXd& certainty) {
    if (type == Parameters::Initialization::none) {
        addSeedPoints(rays, projection, depth_est, certainty);
        return;
    }
    double max_depth{0};
    double sum{0};
    size_t i{0};
    Eigen::Matrix2Xi coordinates(2, projection.size());
    for (auto const& el : projection) {
        coordinates(0, i) = el.first.col;
        coordinates(1, i++) = el.first.row;
        const Pixel& p{el.first};
        const Eigen::ParametrizedLine<double, 3>& ray{rays.at(p)};
        const Eigen::Hyperplane<double, 3> plane(ray.direction(), el.second.position);
        double val{(rays.at(p).intersectionPoint(plane) - rays.at(p).origin()).norm()};
        sum += val;
        if (val > max_depth) {
            max_depth = val;
        }
    }

    if (type == Parameters::Initialization::mean_depth) {
        LOG(INFO) << "Mean Depth";
        double mean_depth{sum / projection.size()};
        LOG(INFO) << "mean depth is " << mean_depth;
        depth_est = mean_depth * Eigen::MatrixXd::Ones(rows, cols);
        certainty = 0 * Eigen::MatrixXd::Ones(rows, cols);
        addSeedPoints(rays, projection, depth_est, certainty);
        return;
    }

    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_;
    flann::Matrix<DistanceType::ElementType> flann_dataset{
        convertEigen2FlannRow(coordinates.transpose().cast<DataType>())};
    kd_index_ptr_ =
        std::make_unique<flann::Index<DistanceType>>(flann_dataset, flann::KDTreeIndexParams(8));
    kd_index_ptr_->buildIndex(flann_dataset);

    if (type == Parameters::Initialization::nearest_neighbor) {
        LOG(INFO) << "nearest_neighbor Depth";
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {

                std::vector<int> neighbours{
                    getNeighbours(coordinates, kd_index_ptr_, Pixel(col, row), 1)};
                const Pixel p(coordinates(0, neighbours[0]), coordinates(1, neighbours[0]));
                const Eigen::ParametrizedLine<double, 3>& ray{rays.at(p)};
                const Eigen::Hyperplane<double, 3> plane(ray.direction(),
                                                         projection.at(p).position);
                depth_est(row, col) =
                    (rays.at(p).intersectionPoint(plane) - rays.at(p).origin()).norm();
                if (depth_est(row, col) > max_depth)
                    depth_est(row, col) = max_depth;
                if (depth_est(row, col) < 0)
                    depth_est(row, col) = 0;
                if (size_t(p.row) == row && size_t(p.col) == col) {
                    certainty(row, col) = 0.1;
                } else {
                    certainty(row, col) = 0.1;
                }
            }
        }
        addSeedPoints(rays, projection, depth_est, certainty);
        return;
    }

    if (type == Parameters::Initialization::triangles) {
        LOG(INFO) << "triangles Depth";
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {

                std::vector<int> all_neighbours{
                    getNeighbours(coordinates, kd_index_ptr_, Pixel(col, row), neighborsearch)};
                std::vector<int> triangle_neighbours{
                    getTriangleNeighbours(all_neighbours, coordinates, Pixel(col, row))};

                if (triangle_neighbours[0] == -1) {
                    depth_est(row, col) = 0;
                    certainty(row, col) = 0;
                } else {
                    Eigen::Matrix3Xd neighbours_points(3, 3);
                    for (size_t i = 0; i < 3; i++) {
                        Pixel c(coordinates(0, triangle_neighbours[i]),
                                coordinates(1, triangle_neighbours[i]));
                        neighbours_points.col(i) = projection.at(c).position;
                    }
                    depth_est(row, col) =
                        pointIntersection(rays.at(Pixel(col, row)), neighbours_points);
                    if (depth_est(row, col) > max_depth)
                        depth_est(row, col) = max_depth;
                    if (depth_est(row, col) < 0)
                        depth_est(row, col) = 0;
                    certainty(row, col) = 0.2;
                }
            }
        }
        addSeedPoints(rays, projection, depth_est, certainty);
    }
}
}
