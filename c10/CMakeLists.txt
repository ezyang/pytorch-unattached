cmake_minimum_required(VERSION 3.10)
project(c10)

set(CMAKE_CXX_STANDARD 11)

set(C10_INSTALL_INCLUDE_SUBDIR "include")

add_library(c10_cuda OBJECT
        c10/cuda/CUDAStorage.cpp
        c10/cuda/CUDAStorage.h
        )

add_library(c10 SHARED
        c10/Tensor.cpp
        c10/ArrayRef.cpp
        c10/TypeId.cpp
        c10/Assert.h
        c10/Assert.cpp
        c10/guts/TensorImpl.cpp
        c10/guts/CPUStorage.cpp
        c10/guts/CPUStorage.h
        c10/guts/Retainable.cpp
        c10/guts/Retainable.h)

target_include_directories(c10 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${C10_INSTALL_INCLUDE_SUBDIR}>
        )

install(DIRECTORY c10 DESTINATION "${C10_INSTALL_INCLUDE_SUBDIR}"
        FILES_MATCHING PATTERN "*.h")