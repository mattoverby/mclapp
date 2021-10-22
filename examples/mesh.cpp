// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include <iostream>

#include "MCL/Application.hpp"
#include "MCL/MeshData.hpp"

#include <igl/readOBJ.h>

using namespace Eigen;

int main(int argc, char *argv[])
{
	(void)(argc);
	(void)(argv);

	std::string obj = MCL_APP_ROOT_DIR "/data/sphere.obj";
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
	app.options.name = "sphere";
    app.start();

	return EXIT_SUCCESS;
}
