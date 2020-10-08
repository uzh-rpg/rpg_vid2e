cmake_minimum_required(VERSION 3.3)
project(esim_py)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_library(libesim STATIC src/esim.cpp)

pybind11_add_module(esim_py src/bindings.cpp)

target_link_libraries(esim_py PRIVATE libesim ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${BOOST_SYSTEM_LIBRARY} Boost::filesystem Boost::system Eigen3::Eigen pybind11::embed)
