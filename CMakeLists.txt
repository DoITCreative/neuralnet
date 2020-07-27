cmake_minimum_required(VERSION 3.17.1)
project(neuralnet)
include_directories(headers)
add_executable(main main.cpp neuron.cpp feedForwardNetwork.cpp)
