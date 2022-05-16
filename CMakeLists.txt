cmake_minimum_required(VERSION 3.19)
project(stitch_c)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(stitch_c orb_algo.h orb_algo.cpp image_stitch.h image_stitch.cpp main.cpp)
target_link_libraries(stitch_c ${OpenCV_LIBS})