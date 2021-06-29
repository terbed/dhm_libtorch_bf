cmake_minimum_required(VERSION 3.17)
project(dhm_dnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH /opt/libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA 10.1 REQUIRED)

add_executable(dhm_dnn main.cpp neuralnetwork.cpp neuralnetwork.h)

target_link_libraries(dhm_dnn PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(dhm_dnn PUBLIC ${OpenCV_LIBS})
target_link_libraries(dhm_dnn PUBLIC ${CUDA_LIBRARIES})