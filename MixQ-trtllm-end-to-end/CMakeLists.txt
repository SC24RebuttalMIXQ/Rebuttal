# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
cmake_minimum_required(VERSION 3.1)

# Enable C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
set(TARGET_NAME trt_llm_custom_plugins)
project(${TARGET_NAME})

set(CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pthread ")

set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

if(NOT DEFINED TRT_INCLUDE_DIR)
  set(TRT_INCLUDE_DIR "/usr/local/tensorrt/include")
  if(NOT EXISTS ${TRT_INCLUDE_DIR})
    # In case of TensorRT installed from a deb package.
    set(TRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
  endif()
endif()
message(STATUS "tensorrt include path: ${TRT_INCLUDE_DIR}")
if(DEFINED TRT_LLM_INCLUDE_DIR)
  message(
    STATUS "openai_triton/manual_plugin example has been self-contained "
           "and TRT_LLM_INCLUDE_DIR is now unnecessary to specify the path of "
           "C++ runtime source files.")
endif()

if(NOT DEFINED TRT_LIB_DIR)
  set(TRT_LIB_DIR "/usr/local/tensorrt/lib")
  if(NOT EXISTS ${TRT_INCLUDE_DIR})
    # In case of TensorRT installed from a deb package.
    set(TRT_LIB_DIR "/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
  endif()
endif()
find_library(
  TRT_LIB_PATH nvinfer
  HINTS ${TRT_LIB_DIR}
  NO_DEFAULT_PATH)
find_library(TRT_LIB_PATH nvinfer REQUIRED)
message(STATUS "TRT_LIB_DIR: ${TRT_LIB_DIR}")
message(STATUS "Found nvinfer library: ${TRT_LIB_PATH}")

if(NOT DEFINED TRT_LLM_LIB_DIR)
  # Find at tensorrt_llm/libs.
  set(TRT_LLM_LIB_DIR /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs)

  list(APPEND TRT_LLM_LIB_DIR "../../../tensorrt_llm/libs")
endif()
find_library(TRT_LLM_LIB_PATH nvinfer_plugin_tensorrt_llm
             HINTS ${TRT_LLM_LIB_DIR} NO_DEFAULT_PATH)
find_library(TRT_LLM_LIB_PATH nvinfer_plugin_tensorrt_llm REQUIRED)
message(STATUS "Found nvinfer_plugin_tensorrt_llm library: ${TRT_LLM_LIB_PATH}")

find_library(TRT_LLM_COMMON_LIB_PATH th_common HINTS ${TRT_LLM_LIB_DIR}
                                                     NO_DEFAULT_PATH)
find_library(TRT_LLM_COMMON_LIB_PATH th_common REQUIRED)
message(STATUS "Found th_common library: ${TRT_LLM_COMMON_LIB_PATH}")

include_directories(/app/tensorrt_llm/include)
 
# Declare the target library.
add_library(
  ${TARGET_NAME} SHARED
  MixQPlugins.cpp
  ImplMixQPlugin.cpp
 )

add_subdirectory(kernel)
add_subdirectory(weightonlykernel)



target_link_libraries(
  ${TARGET_NAME} PUBLIC cuda ${CUDA_LIBRARIES} ${TRT_LLM_LIB_PATH}
                        ${TRT_LLM_COMMON_LIB_PATH} ${TRT_LIB_PATH} cublas 
                        kernel weightonlykernel)

if(NOT MSVC)
  set_property(TARGET ${TARGET_NAME} PROPERTY LINK_FLAGS "-Wl,--no-undefined")
endif()


target_include_directories(${TARGET_NAME} PUBLIC /usr/local/cuda/include)
target_include_directories(${TARGET_NAME} PUBLIC ${TRT_INCLUDE_DIR})
