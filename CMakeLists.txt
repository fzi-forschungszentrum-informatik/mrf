cmake_minimum_required(VERSION 2.8.3)

project(mrf)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

set(CMAKE_MODULE_PATH ~/datkin_ws/src/CMAKE)
MESSAGE( STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

# find dependencies
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

# message generation
#add_message_files(
#  DIRECTORY msg
#  FILES MyMessage.msg
#)
#generate_messages(
#)

# export information to downstream packages
# must be called *before* the targets to generate them in the desired location in the devel space
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS PCL Eigen3 Boost OpenCV GLOG
)

set(glog_DIR glog)
# setup targets
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${Eigen3_LIBRARY_DIRS}
)

add_library(${PROJECT_NAME} include/${PROJECT_NAME}/solver.hpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${OpenCV_LIBS} ${GLOG_LIBRARY})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
#add_executable(foo_bin src/bin.cpp)
#add_dependencies(foo_bin foo_lib)


# install artifacts
#install(
#  DIRECTORY cmake
#  DESTINATION share/${PROJECT_NAME})
#install(
#  DIRECTORY include/
#  DESTINATION include
#)
#install(
#  TARGETS foo_lib foo_bin
#  ARCHIVE DESTINATION lib
#  LIBRARY DESTINATION lib
#  RUNTIME DESTINATION bin
#)

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
