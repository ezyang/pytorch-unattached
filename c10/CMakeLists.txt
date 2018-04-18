cmake_minimum_required(VERSION 3.2)
project(c10)

set(CMAKE_CXX_STANDARD 11)

option(BUILD_TEST "Build C++ test binaries (need gtest)" ON)

# TODO: make this optional
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# NB: This stuff was cargo-culted from ATen.  Halp!!
set(C10_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
set(C10_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
set(C10_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")

set(C10_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
set(C10_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
set(C10_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
set(C10_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")

add_library(c10 SHARED
        c10/Tensor.cpp
        c10/Tensor.h
        c10/ArrayRef.cpp
        c10/ArrayRef.h
        c10/TypeId.cpp
        c10/TypeId.h
        c10/Error.cpp
        c10/Error.h
        c10/Optional.h
        c10/AlignOf.h
        c10/guts/TensorImpl.cpp
        c10/guts/TensorImpl.h
        c10/guts/Retainable.cpp
        c10/guts/Retainable.h
        c10/cpu/CPUStorage.cpp
        c10/cpu/CPUStorage.h
        c10/cpu/CPUAllocator.cpp
        c10/cpu/CPUAllocator.h
        c10/cpu/CPUTensorImpl.cpp
        c10/cpu/CPUTensorImpl.h
        c10/SmallVector.cpp
        c10/SmallVector.h
        c10/DimVector.h
        c10/cpu/CPUContext.cpp
        c10/cpu/CPUContext.h
    c10.h
        c10/DataType.cpp
        c10/DataType.h
        c10/guts/Storage.cpp
        c10/guts/Storage.h
        c10/cpu/op/CPUAll.cpp
        c10/cpu/op/CPUAll.h
        c10/op/All.cpp
        c10/op/All.h
        c10/Utils.cpp
        c10/Utils.h
        c10/Functions.cpp
        c10/Functions.h
        c10/ThreadContext.cpp
        c10/ThreadContext.h
        c10/General.cpp
        c10/General.h)
target_include_directories(c10 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${C10_INSTALL_INCLUDE_SUBDIR}>
        )

install(DIRECTORY c10 DESTINATION "${C10_INSTALL_INCLUDE_SUBDIR}"
  FILES_MATCHING PATTERN "*.h")

install(TARGETS c10 EXPORT c10_targets
  RUNTIME DESTINATION "${C10_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${C10_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${C10_INSTALL_LIB_DIR}")

install(EXPORT c10_targets DESTINATION share/cmake/c10
  FILE c10_targets.cmake
  COMPONENT dev)

# TODO: I don't really want the tests to build if someone is building the library. It's a waste
# of time
add_executable(c10_basic
        test/basic.cpp)
target_link_libraries(c10_basic c10)

if (BUILD_TEST)
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF)
  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(BUILD_GTEST ON)
  set(INSTALL_GTEST OFF)
  # We currently don't need gmock right now.
  set(BUILD_GMOCK OFF)
  # For Windows, we will check the runtime used is correctly passed in.
  if (NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
    set(gtest_force_shared_crt ON)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party/googletest googletest)
  include_directories(${PROJECT_SOURCE_DIR}/../third_party/googletest/googletest/include)

  foreach(test_src
      c10/DataType_test.cpp
      c10/op/All_test.cpp
      )
    # Without directory or longest extension
    get_filename_component(test_name "${test_src}" NAME_WE)
    add_executable("${test_name}" "${test_src}")
    target_link_libraries("${test_name}" c10 gtest_main)
  endforeach()

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()
