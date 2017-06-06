add_executable(eval_planes eval_planes/eval_planes.cpp)
target_link_libraries(eval_planes
        mrf
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
	${GLOG_LIBRARY}
	${CERES_LIBRARIES}
	${yaml-cpp_LIBRARIES}
)

add_executable(eval_scenenet eval_scenenet/eval_scenenet.cpp)
target_link_libraries(eval_scenenet
        mrf
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
	${GLOG_LIBRARY}
	${CERES_LIBRARIES}
	${yaml-cpp_LIBRARIES}
)
