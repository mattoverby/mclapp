# Copyright Matt Overby 2021.
# Distributed under the MIT License.
cmake_minimum_required(VERSION 3.11)

project(mclapp_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
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

# Libigl
# includes Eigen as well
option(LIBIGL_WITH_OPENGL "Use OpenGL" ON)
option(LIBIGL_WITH_OPENGL_GLFW "Use GLFW" ON)
option(LIBIGL_WITH_PNG "Use PNG" ON)
option(LIBIGL_WITH_TRIANGLE "Use Triangle" ON)
if (MCL_USE_IMGUI)
    option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use imgui" ON)
endif()
include(libigl)

# thread building blocks (CPU threading)
find_package(TBB REQUIRED)

# mclgeom, header only
download_project(PROJ mclapp_mclgeom
	GIT_REPOSITORY https://github.com/mattoverby/mclgeom.git
	GIT_TAG main
	QUIET)
list(APPEND MCLAPP_INCLUDE_DIRS ${mclapp_mclgeom_SOURCE_DIR}/include)

# Cereal (serialization), header only
download_project(PROJ mclapp_cereal
	GIT_REPOSITORY https://github.com/USCiLab/cereal.git
	GIT_TAG master
	QUIET)
list(APPEND MCLAPP_INCLUDE_DIRS ${mclapp_cereal_SOURCE_DIR}/include)

# Create the library
file(GLOB MCLAPP_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/include/MCL/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/MCL/*.cpp)
list(APPEND MCLAPP_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MCL_USE_IMGUI)
    set(IMGUI_LIB igl::opengl_glfw_imgui)
else()
    message("MCL USE IMGUI OFF")
    set(IMGUI_LIB "")
endif()

# dependent libs
set(MCLAPP_LIBRARIES
    igl::core
    igl::opengl_glfw
	igl::png
	igl::triangle
    ${IMGUI_LIB}
	${TBB_LIBRARIES}
	stdc++fs)

add_library(mclapp ${MCLAPP_SRC})
target_link_libraries(mclapp PUBLIC ${MCLAPP_LIBRARIES})
target_include_directories(mclapp SYSTEM PUBLIC ${MCLAPP_INCLUDE_DIRS})

# Examples
if (MCL_BUILD_EXAMPLES)
    add_executable(scaf ${CMAKE_CURRENT_SOURCE_DIR}/examples/scaf.cpp)
    target_link_libraries(scaf mclapp)
    
    add_executable(mesh ${CMAKE_CURRENT_SOURCE_DIR}/examples/mesh.cpp)
    target_link_libraries(mesh mclapp)

	add_executable(arap ${CMAKE_CURRENT_SOURCE_DIR}/examples/arap.cpp)
    target_link_libraries(arap mclapp)
endif()

