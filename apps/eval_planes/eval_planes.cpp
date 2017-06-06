#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>

#include "camera.h"
#include "camera_model_ortho.h"
#include "downsample.hpp"
#include "evaluate.hpp"
#include "io.hpp"
#include "quality.hpp"
#include "solver.hpp"

namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;

/** @brief Remove points that are close to the sensor.
 *  @param in Input pointcloud
 *  @param out Output pointcloud
 *  @param threshold Points that are further away will be kept */
template <typename T>
void removeClosePoints(const typename pcl::PointCloud<T>::Ptr& in,
                       typename pcl::PointCloud<T>::Ptr& out,
                       const float threshold) {
    std::vector<int> to_keep;
    for (size_t c = 0; c < in->size(); c++) {
        if (in->points[c].getVector3fMap().norm() > threshold) {
            to_keep.push_back(c);
        }
    }
    size_t before{in->size()};
    pcl::copyPointCloud(*in, to_keep, *out);
    LOG(INFO) << "Removed " << before - out->size() << " points closer than " << threshold
              << " meters from original cloud";
}

/** @brief Stores parameters for and information about the evaluation */
struct EvalParameters {

    static constexpr char del = ','; ///< Delimiter
    inline static std::string header() {
        std::ostringstream oss;
        // clang-format off
        oss << "iteration" << del
        		<< "equidistant" << del
        		<< "random_rate" << del
        		<< "skip_rows" << del
        		<< "skip_cols";
        return oss.str();
        // clang-format on
    }

    inline friend std::ostream& operator<<(std::ostream& os, const EvalParameters& p) {
        // clang-format off
        return os << p.iteration << del
        		<< p.equidistant << del
				<< p.random_rate << del
        		<< p.skip_rows << del
        		<< p.skip_cols;
        // clang-format on
    }

    bool equidistant{true};  ///< Downsample equidistant or random
    double random_rate{0.5}; ///< Downsampling: Percentage of total points to keep
    size_t skip_rows{10};    ///< Downsampling: Keep every n-th row
    size_t skip_cols{10};    ///< Downsampling: Keep every n-th column
    size_t iteration{0};     ///< Current number of iterations running different parameter sets
    bool use_plane_init{false};

    std::string storage_container_file_name;
    float min_dist_m;
};

/** @brief Main evaluation function.
 *  @param p_mrf Parameters for the mrf optimization
 *  @param p_eval Parameters for the evaluation
 *  @param rows Rows of the artificial camera
 *  @param cols Columns of the artificial camera
 *  @return Result info, quality and parameters of the mrf optimization */
std::string evaluate(const mrf::Parameters& p_mrf,
                     const EvalParameters& p_eval,
                     const size_t& rows,
                     const size_t& cols) {

    LOG(INFO) << "Load image";

    const double col_cut{0.3};
    const double row_cut{0.5};
    cv::Mat image_depth{cv::Mat::zeros(rows, cols, cv::DataType<double>::type)}; // image_depth
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < col_cut * cols; col++) {
            image_depth.at<double>(row, col) = 1;
        }
    }
    for (size_t row = 0; row < row_cut * rows; row++) {
        for (size_t col = col_cut * cols; col < cols; col++) {
            image_depth.at<double>(row, col) = (double)row / (row_cut * rows);
        }
    }
    for (size_t row = row_cut * rows; row < rows; row++) {
        for (size_t col = col_cut * cols; col < cols; col++) {
            image_depth.at<double>(row, col) = 0;
        }
    }

    image_depth = image_depth * 100;
    LOG(INFO) << "Image_depth size: " << image_depth.cols << "x" << image_depth.rows << "="
              << image_depth.rows * image_depth.cols;


    LOG(INFO) << "Depth: " << image_depth.depth() << ", channels: " << image_depth.channels();
    // image.convertTo(image, cv::Vec3f::depth);

    LOG(INFO) << "Load camera model";
    std::shared_ptr<mrf::CameraModelOrtho> cam{new mrf::CameraModelOrtho(cols, rows)};

    LOG(INFO) << "Load cloud";

    LOG(INFO) << "Create cloud from depth image_depth";
    using PointIn = pcl::PointXYZINormal;
    using PointOut = pcl::PointXYZRGBNormal;
    using CloudIn = pcl::PointCloud<PointIn>;
    CloudIn::Ptr cl{new CloudIn};
    cl->height = rows;
    cl->width = cols;
    cl->resize(cl->width * cl->height);

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            Eigen::Vector3d support, direction;
            cam->getViewingRay(Eigen::Vector2d(c, r), support, direction);
            const Eigen::ParametrizedLine<double, 3> ray(support, direction);
            cl->at(c, r).getVector3fMap() = ray.pointAt(image_depth.at<double>(r, c)).cast<float>();
            cl->at(c, r).intensity = static_cast<float>(image_depth.at<double>(r, c));
        }
    }

    LOG(INFO) << "Cloud size: " << cl->width << "x" << cl->height << "=" << cl->height * cl->width;
    Eigen::Matrix3Xd point_matrix{cl->getMatrixXfMap().template topRows<3>().template cast<double>()};
    LOG(INFO) << "Cloud points; " << point_matrix.leftCols<3>();
    LOG(INFO) << "Cloud max x,y,z: " << point_matrix.row(0).maxCoeff() << ", " << point_matrix.row(1).maxCoeff() << ", "
              << point_matrix.row(2).maxCoeff();

    using namespace mrf;
    LOG(INFO) << "Downsample cloud";


    CloudIn::Ptr cl_downsampled{new CloudIn};
    if (p_eval.use_plane_init) {
        LOG(INFO) << "Use only three points, one for each plane";
        std::vector<size_t> p_cols;
        std::vector<size_t> p_rows;
        p_cols.emplace_back(col_cut / 2 * cols);
        p_rows.emplace_back(rows / 2);
        p_cols.emplace_back(col_cut * cols + (1 - col_cut) * cols / 2);
        p_rows.emplace_back(rows * row_cut / 2);
        p_cols.emplace_back(col_cut * cols + (1 - col_cut) * cols / 2);
        p_rows.emplace_back(rows * row_cut + rows * row_cut / 2);
        cl_downsampled->reserve(p_cols.size());
        for (size_t i = 0; i < p_cols.size(); i++) {
            Eigen::Vector3d support, direction;
            PointIn p;
            cam->getViewingRay(Eigen::Vector2d(p_cols[i], p_rows[i]), support, direction);
            const Eigen::ParametrizedLine<double, 3> ray(support, direction);
            p.getVector3fMap() = ray.pointAt(image_depth.at<double>(p_rows[i], p_cols[i])).cast<float>();
            cl_downsampled->push_back(p);
        }

    } else {
        if (p_eval.equidistant) {
            cl_downsampled = downsampleEquidistant<PointIn>(cl, p_eval.skip_cols, p_eval.skip_rows);
        } else {
            cl_downsampled = downsampleRandom<PointIn>(cl, p_eval.random_rate);
        }
    }


    LOG(INFO) << "Downsampled cloud: h x w = s: " << cl_downsampled->height << " x " << cl_downsampled->width << " = "
              << cl_downsampled->size();

    LOG(INFO) << "Estimate ground truth normals";
    cl->height = 1;
    cl->width = cl->size();
    // cl = mrf::estimateNormals<PointIn, PointIn>(cl, p_mrf.radius_normal_estimation, true);

    // removeClosePoints<PointIn>(cl, cl, p_eval.min_dist_m);
    image_depth.convertTo(image_depth, 5);
    // removeClosePoints<PointIn>(cl_downsampled, cl_downsampled, p_eval.min_dist_m);
    const Data<PointIn> ground_truth{cl, image_depth, Eigen::Affine3d::Identity()};
    const Data<PointIn> in{cl_downsampled, image_depth, Eigen::Affine3d::Identity()};

    LOG(INFO) << "Call MRF library";

    Data<PointOut> out;
    Solver s{cam, p_mrf};
    const ResultInfo result_info{s.solve(in, out)};
    const mrf::Quality q{evaluate<PointIn, PointOut>(ground_truth, out, cam)};
    LOG(INFO) << "Quality" << std::endl << Quality::header() << std::endl << q;

    LOG(INFO) << "Export data";
    boost::filesystem::path path_name{"/tmp/eval_planes/data/" + std::to_string(p_eval.iteration) + "/"};
    boost::filesystem::create_directories(path_name);
    exportDepthImage(Data<PointIn>{cl_downsampled, image_depth, Eigen::Affine3d::Identity()},
                     cam,
                     path_name.string() + "downsampled_");
    exportImage(out.image, path_name.string() + "out_", true, false, true);
    exportCloud<PointOut>(out.cloud, path_name.string() + "out_");
    exportImage(in.image, path_name.string() + "in_", true, false, true);
    exportCloud<PointIn>(in.cloud, path_name.string() + "in_");
    exportImage(ground_truth.image, path_name.string() + "ground_truth_", true, false, true);
    exportCloud<PointIn>(ground_truth.cloud, path_name.string() + "ground_truth_");

    exportResultInfo(result_info, path_name.string() + "result_info_");
    cv::imwrite(path_name.string() + "depth_error.png", createOutput(cv::abs(q.depth_error)));

    std::ostringstream oss;
    oss << result_info << ResultInfo::del << q << ResultInfo::del << p_mrf;
    return oss.str();
}

/** @brief Creates map of command line options
 *  @param argc Number of arguments
 *  @param argv Arguments
 *  @return Map of the options */
bpo::variables_map makeOptions(int argc, char** argv) {
    bpo::options_description options_description("Supported Parameters");
    // clang-format off
    options_description.add_options()
    		("help,h", "Produce help message")
			("input,i", bpo::value<std::string>()->default_value("/tmp/eval_planes/"), "Path to input folder")
			("output,o", bpo::value<std::string>()->default_value("/tmp/eval_planes/"), "Path to log output")
			("parameters,p", bpo::value<std::string>()->default_value(""), "Path to parameters file")
			("storage_container,c", bpo::value<std::string>()->default_value("/tmp/calibResultMap.bin"), "Path to storage container")
			("samples,s", bpo::value<size_t>()->default_value(3), "Number of samples to use")
			("offset,off", bpo::value<size_t>()->default_value(0), "Naming offset");
    // clang-format on
    bpo::variables_map options;
    bpo::store(bpo::command_line_parser(argc, argv).options(options_description).run(), options);
    bpo::notify(options);
    if (options.count("help")) {
        std::cout << options_description << std::endl;
        std::exit(EXIT_SUCCESS);
    }
    if (options.count("input") == 0) {
        std::cerr << "Error: No input specified." << std::endl;
        std::cout << options_description << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (options.count("output") == 0) {
        std::cerr << "Error: No output specified." << std::endl;
        std::cout << options_description << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return options;
}


int main(int argc, char** argv) {

    google::InitGoogleLogging("eval_planes");
    google::InstallFailureSignalHandler();
    bpo::variables_map options{makeOptions(argc, argv)};

    std::ofstream ofs("/tmp/eval_planes/eval_init.log");
    ofs << EvalParameters::header() << EvalParameters::del << mrf::ResultInfo::header() << mrf::ResultInfo::del
        << mrf::Quality::header() << mrf::ResultInfo::del << mrf::Parameters::header() << std::endl;

    EvalParameters p_eval;
    mrf::Parameters p_mrf(options.at("parameters").as<std::string>());

    /// none
    // p_mrf.use_functor_distance = false;
    // p_mrf.use_functor_normal = false;
    // p_mrf.use_functor_normal_distance = false;
    // p_mrf.use_functor_smoothness_distance = false;
    /// dingler
    // p_mrf.use_functor_distance = true;
    // p_mrf.use_functor_normal = false;
    // p_mrf.use_functor_normal_distance = false;
    // p_mrf.use_functor_smoothness_distance = true;
    /// ours
    // p_mrf.use_functor_distance = true;
    // p_mrf.use_functor_normal = false;
    // p_mrf.use_functor_normal_distance = true;
    // p_mrf.use_functor_smoothness_distance = false;

    ofs << p_eval << EvalParameters::del << evaluate(p_mrf, p_eval, 100, 150) << std::endl;

    ofs.close();

    return 0;
}
