cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(fastwave)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
else()
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O3 -mcpu=cortex-a55 -mtune=cortex-a55 -ffast-math -funroll-loops -fopenmp -fdata-sections -ffunction-sections")
endif()

if(AX650)
    add_definitions(-DAX650)
    if(NOT BSP_MSP_DIR)
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/toolchains/ax650n_bsp_sdk/msp/out)
    endif()
    if(NOT EXISTS ${BSP_MSP_DIR})
        message(FATAL_ERROR "BSP_MSP_DIR ${BSP_MSP_DIR} not exist.")
    endif()
    include_directories(${BSP_MSP_DIR}/include)
    link_directories(${BSP_MSP_DIR}/lib)
    set(MSP_LIBS ax_sys ax_engine ax_interpreter)
else()
    message(FATAL_ERROR "Use -DAX650")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(fastwave_ax650 src/main.cpp)
target_link_libraries(fastwave_ax650 ${MSP_LIBS} pthread m gomp)

install(TARGETS fastwave_ax650 RUNTIME DESTINATION ./)
set_target_properties(fastwave_ax650 PROPERTIES INSTALL_RPATH "$ORIGIN/")
