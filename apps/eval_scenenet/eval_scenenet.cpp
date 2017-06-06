#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera_model_pinhole.h"
#include "downsample.hpp"
#include "evaluate.hpp"
#include "io.hpp"
#include "noise.hpp"
#include "quality.hpp"
#include "solver.hpp"

namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;

/** @brief Stores paths of the image, depth image and instance image */
struct PathContainer {
    inline PathContainer(const bfs::path& photo_, const bfs::path& depth_, const bfs::path& instance_)
            : photo{photo_}, depth{depth_}, instance{instance_} {};
    bfs::path photo;
    bfs::path depth;
    bfs::path instance;
};

/** @brief Stores parameters for and intermation about the evaluation cycles */
struct EvalParameters {

    static constexpr char del = ',';                             ///< Delimiter
    enum class NoiseType { None, DepthNoise, CalibrationNoise }; ///< Type of noise which is added to the depth image
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

    size_t iteration{0};                  ///< Current number of iterations running different parameter sets
    bool equidistant{true};               ///< Downsample equidistant or random
    NoiseType noisetype{NoiseType::None}; ///< Type of noise added to depth image
    float sigma{10};                      ///< Standard deviation of depth noise
    float sigma_rot{0};                   ///< Standard deviation of rotation noise
    float sigma_trans{0};                 ///< Standard deviation of translation noise
    double random_rate{0.5};              ///< Downsampling: Percentage of total points to keep
    size_t skip_rows{10};                 ///< Downsampling: Keep every n-th row
    size_t skip_cols{10};                 ///< Downsampling: Keep every n-th column
};

/** @brief Evaluate
 *  @param pc Location of the image files
 *  @param p_mrf Parameters for the mrf optimization
 *  @param p_eval Parameters for the evaluation
 *  @param test_name Optional test name
 *  @param feature_nr Which features in the image to use
 *  @return String featuring some information about the results */
std::string evaluate(const PathContainer& pc,
                     const mrf::Parameters& p_mrf,
                     const EvalParameters& p_eval,
                     const std::string& test_name = "",
                     const size_t feature_nr = 2) {

    using namespace mrf;
    boost::filesystem::path path_name{"/tmp/eval_scenenet/data/" + std::to_string(p_eval.iteration) + "/"};
    boost::filesystem::create_directories(path_name);

    LOG(INFO) << "Foto path: " << pc.photo.string();
    LOG(INFO) << "Depth path: " << pc.depth.string();
    LOG(INFO) << "Loading and converting images";
    cv::Mat img_photo{cv::imread(pc.photo.string(), cv::IMREAD_COLOR)};
    cv::Mat img_depth{cv::imread(pc.depth.string(), cv::IMREAD_ANYDEPTH)};
    cv::Mat img_instance{cv::imread(pc.instance.string(), cv::IMREAD_ANYDEPTH)};
    LOG(INFO) << "Depth photo: " << img_photo.depth() << ", depth depth: " << img_depth.depth()
              << ", depth instance: " << img_instance.depth();
    LOG(INFO) << "Channels photo: " << img_photo.channels() << ", channels depth: " << img_depth.channels()
              << ", channels instance: " << img_instance.channels();

    /**
     * \attention Units in dataset are in mm! Need to convert to meters
     */
    img_depth.convertTo(img_depth, cv::DataType<double>::type, 0.001);
    img_photo.convertTo(img_photo, cv::Vec3f::depth);
    img_instance.convertTo(img_instance, cv::DataType<float>::type);

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

    cv::minMaxLoc(img_depth, &minVal, &maxVal, &minLoc, &maxLoc);
    LOG(INFO) << "Groundtruth img_depth min + max values: " << minVal << " + " << maxVal;


    LOG(INFO) << "Export converted images";
    exportImage(img_photo, path_name.string() + "rgb_", true);
    exportImage(img_instance, path_name.string() + "instance_", true, false, true);
    exportImage(img_depth, path_name.string() + "depth_", true, false, true);

    const size_t rows = img_depth.rows;
    const size_t cols = img_depth.cols;
    std::vector<cv::Mat> channels;
    cv::split(img_photo, channels);
    channels.emplace_back(img_instance);
    cv::Mat_<cv::Vec4f> img_multi_channel(rows, cols);
    cv::merge(channels, img_multi_channel);

    LOG(INFO) << "Create camera model";
    const double focal_length{cols / (2 * std::tan(M_PI / 6))};
    std::shared_ptr<CameraModelPinhole> cam{new CameraModelPinhole(cols, rows, focal_length, cols / 2, rows / 2)};

    LOG(INFO) << "Create cloud from depth image";
    using PointIn = pcl::PointXYZINormal;
    using CloudIn = pcl::PointCloud<PointIn>;
    CloudIn::Ptr cl{new CloudIn};
    cl->height = rows;
    cl->width = cols;
    cl->resize(cl->width * cl->height);
    cv::Mat img;
    cv::cvtColor(img_photo, img, CV_BGR2GRAY);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            Eigen::Vector3d support, direction;
            cam->getViewingRay(Eigen::Vector2d(c, r), support, direction);
            const Eigen::ParametrizedLine<double, 3> ray(support, direction);
            cl->at(c, r).getVector3fMap() = ray.pointAt(img_depth.at<double>(r, c)).cast<float>();
            cl->at(c, r).intensity = img.at<float>(r, c);
        }
    }
    cl = mrf::estimateNormals<PointIn, PointIn>(cl, p_mrf.radius_normal_estimation, false);

    const Data<PointIn> ground_truth{cl, img_depth};
    LOG(INFO) << "Export ground truth data";
    exportData(ground_truth, path_name.string() + "ground_truth_");

    LOG(INFO) << "Downsample cloud";
    CloudIn::Ptr cl_downsampled;
    if (p_eval.equidistant) {
        cl_downsampled = downsampleEquidistant<PointIn>(cl, p_eval.skip_cols, p_eval.skip_rows);
    } else {
        cl_downsampled = downsampleRandom<PointIn>(cl, p_eval.random_rate);
    }

    if (p_eval.noisetype == EvalParameters::NoiseType::CalibrationNoise) {
        LOG(INFO) << "Add Calibration Noise";
        cl_downsampled = addCalibrationNoise<PointIn>(cl_downsampled, p_eval.sigma_trans, p_eval.sigma_rot);
    } else if (p_eval.noisetype == EvalParameters::NoiseType::DepthNoise) {
        LOG(INFO) << "Add Depth Noise";
        cl_downsampled = addDepthNoise<PointIn>(cl_downsampled, p_eval.sigma);
    }

    const Data<PointIn> in{cl_downsampled, img_multi_channel};
    exportCloud<PointIn>(in.cloud, path_name.string() + "in_");
    LOG(INFO) << "Export downsampled data";
    exportDepthImage(
        Data<PointIn>{cl_downsampled, img_depth}, cam, path_name.string() + "depth_downsampled_", true, false, true);
    exportOverlay(Data<PointIn>{cl_downsampled, img_photo}, cam, path_name.string() + "downsampled_overlay_");

    cv::Mat image_in;
    switch (feature_nr) { //> can be put into better structure for future
    case 0:
        image_in = img;
        LOG(INFO) << "Use Gray features only";
        break;
    case 1:
        image_in = img_photo;
        LOG(INFO) << "Use RGB features only";
        break;
    case 2:
        image_in = img_multi_channel;
        LOG(INFO) << "Use RGB and instance features";
        break;
    }

    LOG(INFO) << "Call MRF library";
    using PointOut = pcl::PointXYZRGBNormal;
    Data<PointOut> out;
    Solver s{cam, p_mrf};
    const ResultInfo result_info{s.solve(in, out)};
    const mrf::Quality q{mrf::evaluate<PointIn, PointOut>(ground_truth, out, cam)};
    LOG(INFO) << "Quality" << std::endl << Quality::header() << std::endl << q;

    LOG(INFO) << "Export output data";
    exportImage(out.image, path_name.string() + "out_", true, false, true);
    exportCloud<PointOut>(out.cloud, path_name.string() + "out_");

    LOG(INFO) << "Export depth error";
    exportImage(cv::abs(q.depth_error), path_name.string() + "depth_error_", true, true, true);

    LOG(INFO) << "Export result info";
    exportResultInfo(result_info, path_name.string() + "result_info_");


    std::ostringstream oss;
    oss << result_info << ResultInfo::del << q << ResultInfo::del << p_mrf << ResultInfo::del << maxVal
        << ResultInfo::del << minVal;
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
			("input,i", bpo::value<std::string>()->default_value("/datasets/scenenet_rgb-d/val/0/0"), "Path to input folders")
			("output,o", bpo::value<std::string>()->default_value("/tmp/eval_scenenet_rgb-d"), "Path to log output")
			("parameters,p", bpo::value<std::string>()->default_value(""), "Path to parameters file")
			("samples,s", bpo::value<size_t>()->default_value(3), "Number of samples to use")
			("offset,off", bpo::value<size_t>()->default_value(0), "Naming offset");
    // clang-format on
    bpo::variables_map options;
    bpo::store(bpo::command_line_parser(argc, argv).options(options_description).run(), options);
    bpo::notify(options);
    if (!options.count("help")) {
        std::cout << options_description << std::endl;
        std::exit(EXIT_SUCCESS);
    }
    if (!options.count("input")) {
        std::cerr << "Error: No input specified." << std::endl;
        std::cout << options_description << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (!options.count("output")) {
        std::cerr << "Error: No output specified." << std::endl;
        std::cout << options_description << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return options;
}

/** @brief Extract locations of images, depth images and instance images from a given path
 *  @param p Input path
 *  @return Vector of locations of images, depth images and instance images */
std::vector<PathContainer> extractData(const bfs::path& p) {

    if (!bfs::exists(p) || !bfs::is_directory(p)) {
        std::cerr << "Error: Path '" << p.string() << "' is not a directory." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const bfs::path photo{p / "photo"};
    if (!bfs::exists(photo) || !bfs::is_directory(photo)) {
        std::cerr << "Error: Path '" << photo.string() << "' is not a directory." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const bfs::path depth{p / "depth"};
    if (!bfs::exists(depth) || !bfs::is_directory(depth)) {
        std::cerr << "Error: Path '" << depth.string() << "' is not a directory." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const bfs::path instance{p / "instance"};
    if (!bfs::exists(instance) || !bfs::is_directory(instance)) {
        std::cerr << "Error: Path '" << instance.string() << "' is not a directory." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<bfs::path> photos;
    bfs::directory_iterator dir_iter_photo(photo), dir_end_photo;
    for (; dir_iter_photo != dir_end_photo; ++dir_iter_photo) {
        photos.push_back(dir_iter_photo->path());
    }
    std::sort(photos.begin(), photos.end());

    std::vector<bfs::path> depths;
    bfs::directory_iterator dir_iter_depths(depth), dir_end_depths;
    for (; dir_iter_depths != dir_end_depths; ++dir_iter_depths) {
        depths.push_back(dir_iter_depths->path());
    }
    std::sort(depths.begin(), depths.end());
    PCHECK(photos.size() == depths.size());

    std::vector<bfs::path> instances;
    bfs::directory_iterator dir_iter_instances(instance), dir_end_instances;
    for (; dir_iter_instances != dir_end_instances; ++dir_iter_instances) {
        instances.push_back(dir_iter_instances->path());
    }
    std::sort(instances.begin(), instances.end());
    PCHECK(photos.size() == instances.size());

    std::vector<PathContainer> paths;
    paths.reserve(photos.size());
    for (size_t c = 0; c < photos.size(); c++) {
        paths.emplace_back(photos[c], depths[c], instances[c]);
    }
    return paths;
}

int main(int argc, char** argv) {

    google::InitGoogleLogging("eval_scenenet");
    google::InstallFailureSignalHandler();
    bpo::variables_map options{makeOptions(argc, argv)};

    const bfs::path p(options.at("input").as<std::string>());
    const std::vector<PathContainer> paths{extractData(p)};

    std::ofstream ofs_init("/tmp/eval_scenenet/eval_init.log");
    ofs_init << EvalParameters::header() << EvalParameters::del << mrf::ResultInfo::header() << mrf::ResultInfo::del
             << mrf::Quality::header() << mrf::ResultInfo::del << mrf::Parameters::header() << EvalParameters::del
             << "gt_depth_max " << EvalParameters::del << "gt_depth_min" << EvalParameters::del << "sample_nr"
             << std::endl;

    mrf::Parameters p_mrf(options.at("parameters").as<std::string>());

    EvalParameters p_eval;
    p_eval.noisetype = EvalParameters::NoiseType::None;
    p_eval.equidistant = true;

    /// none(only init)
    // p_mrf.use_functor_distance = false;
    // p_mrf.use_functor_normal = false;
    // p_mrf.use_functor_normal_distance = false;
    // p_mrf.use_functor_smoothness_distance = false;
    /// normal
    // p_mrf.use_functor_distance = true;
    // p_mrf.use_functor_normal = false;
    // p_mrf.use_functor_normal_distance = true;
    // p_mrf.use_functor_smoothness_distance = false;
    /// Diebel
    // p_mrf.use_functor_distance = true;
    // p_mrf.use_functor_normal = false;
    // p_mrf.use_functor_normal_distance = false;
    // p_mrf.use_functor_smoothness_distance = true;

    ofs_init << p_eval << EvalParameters::del << evaluate(paths[0], p_mrf, p_eval, "init") << EvalParameters::del
             << std::endl;
    ofs_init.close();
    return EXIT_SUCCESS;
}
