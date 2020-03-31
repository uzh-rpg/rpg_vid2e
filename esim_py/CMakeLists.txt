cmake_minimum_required(VERSION 2.8.12)
project(esim_py)

add_subdirectory(pybind11)

include_directories(include include/eigen3.3.7)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem REQUIRED)

add_library(libesim STATIC src/esim.cpp)

pybind11_add_module(esim_py src/bindings.cpp)

target_link_libraries(esim_py PRIVATE libesim ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY})
