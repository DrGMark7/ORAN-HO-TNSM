cmake_minimum_required(VERSION 3.16)
project(oran_cplex_optimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(OpenMP REQUIRED)

if(NOT CPLEX_ROOT)
  message(FATAL_ERROR "Set CPLEX_ROOT to the CPLEX Studio 22.1.1 installation")
endif()

set(CPLEX_INCLUDE "${CPLEX_ROOT}/cplex/include")
set(CONCERT_INCLUDE "${CPLEX_ROOT}/concert/include")
set(CPLEX_LIB_DIR "${CPLEX_ROOT}/cplex/lib/x86-64_linux/static_pic")
set(CONCERT_LIB_DIR "${CPLEX_ROOT}/concert/lib/x86-64_linux/static_pic")

foreach(required
    "${CPLEX_INCLUDE}/ilcplex/ilocplex.h"
    "${CONCERT_INCLUDE}/ilconcert/iloenv.h"
    "${CPLEX_LIB_DIR}/libcplex.a"
    "${CPLEX_LIB_DIR}/libilocplex.a"
    "${CONCERT_LIB_DIR}/libconcert.a")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing required CPLEX file: ${required}")
  endif()
endforeach()

add_library(oran_cplex_optimizer SHARED
  cplex_optimizer.cpp
  feasible_space_counter.cpp
  large_feasible_space_counter.cpp)
target_compile_definitions(oran_cplex_optimizer PRIVATE IL_STD)
target_include_directories(oran_cplex_optimizer PRIVATE "${CPLEX_INCLUDE}" "${CONCERT_INCLUDE}")
target_link_libraries(oran_cplex_optimizer PRIVATE
  OpenMP::OpenMP_CXX
  "${CPLEX_LIB_DIR}/libilocplex.a"
  "${CONCERT_LIB_DIR}/libconcert.a"
  "${CPLEX_LIB_DIR}/libcplex.a"
  pthread dl m)
target_compile_options(oran_cplex_optimizer PRIVATE
  $<$<CONFIG:Release>:-O3>)
set_target_properties(oran_cplex_optimizer PROPERTIES OUTPUT_NAME "oran_cplex_optimizer")

