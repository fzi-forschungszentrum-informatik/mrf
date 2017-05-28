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

file(GLOB_RECURSE PROJECT_HEADER_FILES_INC RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "include/*.h" "include/*.hpp")
file(GLOB PROJECT_SOURCE_FILES_INC RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "src/*.h" "src/*.hpp")
file(GLOB PROJECT_SOURCE_FILES_SRC RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "src/*.cpp")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS PCL Eigen3 Boost OpenCV GLOG
)

set(glog_DIR glog)
include_directories(
  include/${PROJECT_NAME}
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

# Declare a cpp library
add_library(${PROJECT_NAME}
    ${PROJECT_HEADER_FILES_INC}
    ${PROJECT_SOURCE_FILES_INC}
    ${PROJECT_SOURCE_FILES_SRC}
)
add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
		${PCL_LIBRARIES}
		${Eigen3_LIBRARIES}
		${OpenCV_LIBRARIES}
		${GLOG_LIBRARY}
		${CERES_LIBRARIES}
)
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
