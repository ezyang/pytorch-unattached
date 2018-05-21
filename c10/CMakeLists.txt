cmake_minimum_required(VERSION 3.2)
project(c10)

set(CMAKE_CXX_STANDARD 17)

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
    c10.cpp
    c10/Tensor.cpp
    c10/ArrayRef.cpp
    c10/Error.cpp
    c10/Optional.cpp
    c10/AlignOf.cpp
    c10/guts/TensorImpl.cpp
    c10/guts/IntrusivePtr.cpp
    c10/guts/Metaprogramming.cpp
    c10/guts/Macros.cpp
    c10/guts/C++17.cpp
    c10/guts/IdWrapper.cpp
    c10/guts/TypeId.cpp
    c10/guts/caffe2/common.cc
    c10/guts/scope_guard.cpp
    c10/guts/Any.cpp
    c10/cpu/CPUStorage.cpp
    c10/cpu/CPUAllocator.cpp
    c10/cpu/CPUTensorImpl.cpp
    c10/dispatch/Dispatcher.cpp
    c10/dispatch/impl/DispatchTable.cpp
    c10/dispatch/impl/DispatchKey.cpp
    c10/dispatch/OpId.cpp
    c10/dispatch/OpRegistration.cpp
    c10/dispatch/OpSchema.cpp
    c10/dispatch/TensorTypeId.cpp
    c10/dispatch/TensorTypeIdRegistration.cpp
    c10/dispatch/OpSchemaRegistration.cpp
    c10/stack_bindings/CallStack.cpp
    c10/stack_bindings/StackBasedOperator.cpp
    c10/stack_bindings/StackBasedOperatorRegistry.cpp
    c10/SmallVector.cpp
    c10/DimVector.cpp
    c10/cpu/CPUContext.cpp
    c10/DataType.cpp
    c10/guts/Storage.cpp
    c10/cpu/op/CPUAll.cpp
    c10/op/All.cpp
    c10/op/OpSchemaDefs.cpp
    c10/Utils.cpp
    c10/Functions.cpp
    c10/ThreadContext.cpp
    c10/General.cpp
    c10/KeywordArgs.cpp
    c10/Context.cpp
    c10/dispatch/OpSig.cpp c10/dispatch/OpSig.h)
target_include_directories(c10 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${C10_INSTALL_INCLUDE_SUBDIR}>
    )
target_include_directories(c10 SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../third_party/flat_hash_map>
        $<INSTALL_INTERFACE:${C10_INSTALL_INCLUDE_SUBDIR}/../third_party/flat_hash_map>
        )
target_compile_options(c10 PRIVATE -Wall -Wextra -Wold-style-cast -Wno-missing-braces -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-overflow=5 -Wundef -fdiagnostics-show-option -Wconversion -Wpedantic -Wno-gnu-zero-variadic-macro-arguments -Werror)
set_property(TARGET c10 PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(c10_dispatch_example c10/dispatch/DispatcherExample.cpp)
target_compile_options(c10_dispatch_example PRIVATE -Wall -Wextra -Wold-style-cast -Wno-missing-braces -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-overflow=5 -Wundef -fdiagnostics-show-option -Wconversion -Wpedantic -Wno-gnu-zero-variadic-macro-arguments -Werror)
target_link_libraries(c10_dispatch_example c10)
set_property(TARGET c10_dispatch_example PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(c10_proto_example c10/dispatch/ProtoParserExample.cpp)
target_compile_options(c10_proto_example PRIVATE -Wall -Wextra -Wold-style-cast -Wno-missing-braces -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-overflow=5 -Wundef -fdiagnostics-show-option -Wconversion -Wpedantic -Wno-gnu-zero-variadic-macro-arguments -Werror)
target_link_libraries(c10_proto_example c10)
set_property(TARGET c10_proto_example PROPERTY POSITION_INDEPENDENT_CODE ON)

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
      c10/cpu/op/CPUAll_test.cpp
      c10/dispatch/DispatcherExample.cpp
      c10/Registry_test.cpp
      c10/dispatch/OpSchema_test.cpp
      c10/dispatch/OpSig_test.cpp
      )
    # Without directory or longest extension
    get_filename_component(test_name "${test_src}" NAME_WE)
    add_executable("${test_name}" "${test_src}")
    target_link_libraries("${test_name}" c10 gtest_main)
  endforeach()

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()
