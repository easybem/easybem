cmake_minimum_required(VERSION 3.15)

project(BEM3dLaplace LANGUAGES CXX)

option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_MATLAB "Build MATLAB bindings" OFF)

option(USE_CUDA "Build with CUDA support" ON)

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

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(easybem_core PRIVATE
      src/BEM3dLaplaceGPU.cu
      src/meshTopology.cpp
    )
    target_compile_definitions(easybem_core PRIVATE
        USE_CUDA
    )
    target_link_libraries(easybem_core PRIVATE
        CUDA::cudart
    )
    set_target_properties(easybem_core PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 61
        #CUDA_ARCHITECTURES native
    )
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_MATLAB)
    add_subdirectory(matlab)
endif()
