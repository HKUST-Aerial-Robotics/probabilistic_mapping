cmake_minimum_required(VERSION 2.8.3)

project(mapping)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Og -fopenmp -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -fopenmp -march=native -DEIGEN_DONT_PARALLELIZE")

#include(/usr/local/share/OpenCV/OpenCVConfig.cmake)

find_package(catkin REQUIRED COMPONENTS
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    dynamic_reconfigure
    geometry_msgs
    rosbag
)
find_package(Boost COMPONENTS thread serialization REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(Ceres REQUIRED)
#CUDA
find_package(CUDA 9 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
#set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
#SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;" -std=c++11 --generate-code arch=compute_50,code=sm_50)
FILE(GLOB MOTIONSTEREO_SRC motion_stereo/*.*)
CUDA_COMPILE(MOTIONSTEREO ${MOTIONSTEREO_SRC})

catkin_package(
    DEPENDS
    CATKIN_DEPENDS
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${catkin_INCLUDE_DIRS}
    )

set(LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
#    ${CUDA_nppi_LIBRARY}
    ${CERES_LIBRARIES}
    ${catkin_LIBRARIES}
)

add_executable(evaluate evaluate.cc ${MOTIONSTEREO} ${MOTIONSTEREO_SRC})
target_link_libraries(evaluate ${LIBS})
