cmake_minimum_required(VERSION 3.5)
project(240306_mesh_deduplicator_cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture compute_86")
find_package(CUDA)

include_directories(include)

if(CUDA_FOUND)
    message(STATUS "CUDA found")
    set(cuda_src 
    include/kernel.cuh
    src/kernel.cu)
    cuda_add_library(cuda_lib ${cuda_src})

else()
    message(STATUS "CUDA not found")
endif()

LIST(APPEND SRC_FILES
    include/host_deduplication.hpp 
    src/main.cpp
)

add_executable(run ${SRC_FILES})
target_link_libraries(run cuda_lib)