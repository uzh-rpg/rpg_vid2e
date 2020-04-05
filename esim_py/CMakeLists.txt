cmake_minimum_required(VERSION 3.1)
project(esim_py)

add_subdirectory(pybind11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_library(libesim STATIC src/esim.cpp)

pybind11_add_module(esim_py src/bindings.cpp)

target_link_libraries(esim_py PRIVATE libesim ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} Eigen3::Eigen)
