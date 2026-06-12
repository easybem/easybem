find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(
    easybem
    bindings.cpp
)

target_link_libraries(
    easybem
    PRIVATE
        easybem_core
)

target_include_directories(
    easybem
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
