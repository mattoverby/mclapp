// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include <iostream>

#include "MCL/Application.hpp"
#include "MCL/MeshData.hpp"

#include <igl/readOBJ.h>

#ifdef MCL_APP_USE_IMGUI
    #include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#endif

using namespace Eigen;

bool custom_gui_options()
{
#ifdef MCL_APP_USE_IMGUI
	if (ImGui::CollapsingHeader("custom options", ImGuiTreeNodeFlags_DefaultOpen))
	{
	    if (ImGui::Button("test button")) { std::cout << "Hello!" << std::endl; }
	}
#endif
    return false; // true to update rendering
}

int main(int argc, char *argv[])
{
	(void)(argc);
	(void)(argv);

	//std::string obj = MCL_APP_ROOT_DIR "/data/sphere.obj";
	std::string obj = MCL_APP_ROOT_DIR "/data/bunny_lowres.obj";
	MatrixXd V;
	MatrixXi F;
	if (!igl::readOBJ(obj, V, F))
		return EXIT_FAILURE;

	// MeshData is a global singleton that stores geometry info
	// and computes prim/facet/etc mappings.
	mcl::MeshData &meshdata = mcl::MeshData::get();
	meshdata.add_mesh(V, F);

	// Create app that directly interfaces with MeshData
	mcl::Application app;
	app.options.name = "bunny"; // optional
	app.draw_gui_callback = &custom_gui_options; // optional
    app.start();

	return EXIT_SUCCESS;
}
