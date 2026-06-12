cmake_minimum_required(VERSION 3.15)

project(BEM3dLaplace LANGUAGES CXX)

option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_MATLAB "Build MATLAB bindings" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(easybem_core SHARED
    src/BEM3dLaplace.cpp
    src/BEM3dElasticity.cpp
    src/func.cpp
    src/quadrature.cpp
)

set_target_properties(
    easybem_core
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)  

target_include_directories(easybem_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()
