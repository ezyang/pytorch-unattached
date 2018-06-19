cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)

option(BUILD_TEST "Build C++ test binaries (need gtest)" ON)

# TODO: make this optional
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_library(c10 OBJECT
    c10.cpp
    c10/Tensor.cpp
    c10/ArrayRef.cpp
    c10/Error.cpp
    c10/Optional.cpp
    c10/AlignOf.cpp
    c10/guts/TensorImpl.cpp
    c10/guts/IntrusivePtr.cpp
    c10/guts/scope_guard.cpp
    c10/guts/Any.cpp
    c10/cpu/CPUStorage.cpp
    c10/cpu/CPUAllocator.cpp
    c10/cpu/CPUTensorImpl.cpp
    c10/dispatch/Dispatcher.cpp
    c10/dispatch/impl/DispatchTable.cpp
    c10/dispatch/KernelRegistration.cpp
    c10/dispatch/OpSchemaRegistration.cpp
    c10/stack_bindings/ParameterStack.cpp
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
set_property(TARGET c10 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(c10 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flat_hash_map>)
target_include_directories(c10 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/caffe2/c10_prototype>)

add_executable(c10_dispatch_example $<TARGET_OBJECTS:c10> c10/dispatch/DispatcherExample.cpp)
target_include_directories(c10_dispatch_example PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flat_hash_map>)
target_include_directories(c10_dispatch_example PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/caffe2/c10_prototype>)
set_property(TARGET c10_dispatch_example PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(c10_dispatch_example caffe2_library)

add_executable(c10_proto_example $<TARGET_OBJECTS:c10> c10/dispatch/ProtoParserExample.cpp)
target_include_directories(c10_proto_example PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flat_hash_map>)
target_include_directories(c10_proto_example PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/caffe2/c10_prototype>)
target_link_libraries(c10_proto_example caffe2_library)
set_property(TARGET c10_proto_example PROPERTY POSITION_INDEPENDENT_CODE ON)

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
  #add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party/googletest googletest)
  #include_directories(${PROJECT_SOURCE_DIR}/../third_party/googletest/googletest/include)

  foreach(test_src
      c10/DataType_test.cpp
      c10/op/All_test.cpp
      c10/cpu/op/CPUAll_test.cpp
      c10/dispatch/DispatcherExample.cpp
      c10/Registry_test.cpp
      c10/dispatch/OpSig_test.cpp
      )
    # Without directory or longest extension
    get_filename_component(test_name "${test_src}" NAME_WE)
    add_executable("${test_name}" $<TARGET_OBJECTS:c10> "${test_src}")
    target_include_directories(${test_name} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flat_hash_map>)
    target_include_directories(${test_name} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/caffe2/c10_prototype>)
    target_link_libraries("${test_name}" gtest_main caffe2_library)
  endforeach()

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()
