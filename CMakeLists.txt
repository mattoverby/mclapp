# Copyright Matt Overby 2021.
# Distributed under the MIT License.
cmake_minimum_required(VERSION 3.0)

project(mclapp_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(DownloadProject)

option(MCL_DEBUG "Compile in test mode" OFF)
option(MCL_BUILD_EXAMPLES "Compile examples" ON)
option(MCL_HEADLESS "Compile without render libs" OFF)
option(MCL_USE_IMGUI "Compile with IMGUI" OFF)

add_definitions(-DMCL_APP_OUTPUT_DIR="${CMAKE_BINARY_DIR}/output")
add_definitions(-DMCL_APP_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if (MCL_USE_IMGUI)
    add_definitions(-DMCL_APP_USE_IMGUI)
endif()

# I know you should not set build type directly and instead let the user
# set it, but this is way more convenient.
if (MCL_DEBUG)
	add_definitions(-DMCL_DEBUG)
	set(CMAKE_BUILD_TYPE Debug)
	if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wundef -Wextra -Wno-long-long -Wlogical-op -Wshadow -Wmaybe-uninitialized")
	endif()
else()
	set(CMAKE_BUILD_TYPE Release)
endif()

set(MCLAPP_INCLUDE_DIRS "")

# thread building blocks (CPU threading)
find_package(TBB REQUIRED)
set(MCLAPP_INCLUDE_DIRS "${MCLAPP_INCLUDE_DIRS};${TBB_INCLUDE_DIRS}")

# Eigen, header only
find_package(EIGEN REQUIRED)
set(MCLAPP_INCLUDE_DIRS "${MCLAPP_INCLUDE_DIRS};${EIGEN_INCLUDE_DIR}")

# mclgeom, header only
download_project(PROJ mclgeom
	GIT_REPOSITORY https://github.com/mattoverby/mclgeom.git
	GIT_TAG main
	PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/external/
	UPDATE_DISCONNECTED 1
	QUIET)
set(MCLAPP_INCLUDE_DIRS "${MCLAPP_INCLUDE_DIRS};${mclgeom_SOURCE_DIR}/include")

# Cereal (serialization), header only
download_project(PROJ cereal
	GIT_REPOSITORY https://github.com/USCiLab/cereal.git
	GIT_TAG master
	PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/external/
	UPDATE_DISCONNECTED 1
	QUIET)
set(MCLAPP_INCLUDE_DIRS "${MCLAPP_INCLUDE_DIRS};${cereal_SOURCE_DIR}/include")

# libigl, geom processing and viewer
# TBH I'm not sure how to use download_project correctly
# with libigl's sophisticated build scripts.
# Oh well, this works.
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
#option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    MCL_USE_IMGUI)
set(LIBIGL_WITH_OPENGL_GLFW_IMGUI                                   MCL_USE_IMGUI)
option(LIBIGL_WITH_PNG               "Use PNG"                      ON)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 ON)
download_project(PROJ libigl
	GIT_REPOSITORY https://github.com/libigl/libigl.git
	GIT_TAG main
	PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/external/
	UPDATE_DISCONNECTED 1
	QUIET)
list(APPEND CMAKE_MODULE_PATH "${libigl_SOURCE_DIR}/cmake")
include(libigl)
set(LIBIGL_INCLUDE_DIRS
	${glad_SOURCE_DIR}/include
	${libigl_SOURCE_DIR}/include
	${libigl_SOURCE_DIR}/external
	${libigl_SOURCE_DIR}/external/stb
	${libigl_SOURCE_DIR}/external/imgui
	${libigl_SOURCE_DIR}/external/imgui/examples
	${libigl_SOURCE_DIR}/external/libigl-imgui
	${libigl_SOURCE_DIR}/external/imgui/backends)
set(MCLAPP_INCLUDE_DIRS "${MCLAPP_INCLUDE_DIRS};${LIBIGL_INCLUDE_DIRS}")

# Create the library
file(GLOB MCLAPP_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/include/MCL/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/MCL/*.cpp)
file(COPY # why do I have to do this?
	${CMAKE_CURRENT_SOURCE_DIR}/data/igl_stb_image_export.h
	DESTINATION
	${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(MCLAPP_INCLUDE_DIRS "${MCLAPP_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MCL_USE_IMGUI)
    set(IMGUI_LIB igl::opengl_glfw_imgui)
else()
    message("MCL USE IMGUI OFF")
    set(IMGUI_LIB "")
endif()

# dependent libs
set(MCLAPP_LIBRARIES
	igl::core
	igl::png
	igl::opengl_glfw
    ${IMGUI_LIB}
	${TBB_LIBRARIES}
	stdc++fs)

if (LIBIGL_WITH_TRIANGLE)
	set(MCLAPP_LIBRARIES igl::triangle ${MCLAPP_LIBRARIES})
endif()

add_library(mclapp ${MCLAPP_SRC})
target_link_libraries(mclapp PUBLIC ${MCLAPP_LIBRARIES})
target_include_directories(mclapp SYSTEM PUBLIC ${MCLAPP_INCLUDE_DIRS})

# Examples
if (MCL_BUILD_EXAMPLES)
    add_executable(scaf ${CMAKE_CURRENT_SOURCE_DIR}/examples/scaf.cpp)
    target_link_libraries(scaf mclapp)
    
    add_executable(mesh ${CMAKE_CURRENT_SOURCE_DIR}/examples/mesh.cpp)
    target_link_libraries(mesh mclapp)
endif()

