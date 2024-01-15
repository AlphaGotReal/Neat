cmake_minimum_required(VERSION 3.4)
project(neat)
add_subdirectory(pybind11)
pybind11_add_module(neat src/utils.cpp)
