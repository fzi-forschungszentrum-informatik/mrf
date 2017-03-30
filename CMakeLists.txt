cmake_minimum_required(VERSION 2.8.3)

project(mrf)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(catkin REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
	filesystem
	program_options
)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS PCL Eigen3 Boost OpenCV GLOG
)

set(glog_DIR glog)
include_directories(
  include/mrf
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${Eigen3_LIBRARY_DIRS}
)

add_library(${PROJECT_NAME}
	include/${PROJECT_NAME}/solver.hpp
	src/camera.cpp
	src/parameters.cpp
        src/camera_model_pinhole.cpp
        src/image_preprocessing.cpp
        src/camera_model.cpp
        src/neighbors.cpp
        src/prior.cpp
        src/smoothness_weight.cpp
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${OpenCV_LIBS} ${GLOG_LIBRARY} ${CERES_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# tests
#if(CATKIN_ENABLE_TESTING)
#  catkin_add_gtest(${PROJECT_NAME}_gtest test/my_test.cpp)
#  if(TARGET ${PROJECT_NAME}_gtest)
#   target_link_libraries(
#      ${PROJECT_NAME}_gtest
#      ${catkin_LIBRARIES}
#    )
#  endif()
#endif()
