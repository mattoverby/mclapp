// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>

#include <iostream>

#include "MCL/ReadSeamedObj.hpp"
#include "MCL/Logger.hpp"
#include "MCL/Application.hpp"
#include "MCL/MeshData.hpp"
#include "MCL/AssertHandler.hpp"

#include <igl/opengl/glfw/Viewer.h>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/mapping_energy_with_jacobians.h>
#include <igl/polar_svd.h>
#include <igl/triangle/scaf.h>

using namespace Eigen;
using namespace mcl;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXi;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXd;
typedef Eigen::Matrix<double,2,3> Matrix23d;

class Optimizer
{
public:
	igl::triangle::SCAFData scaf_data;
	RowMatrixXd X;
	void init();
	void step();
};

int main(int argc, char *argv[])
{
	(void)(argc);
	(void)(argv);

	std::string obj = MCL_APP_ROOT_DIR "/data/camel_b.obj";
	MatrixXd V;
	MatrixXi F;
	if (!igl::readOBJ(obj, V, F))
		return EXIT_FAILURE;

	// MeshData is a global singleton that stores geometry info
	// and computes prim/facet/etc mappings.
	MeshData &meshdata = MeshData::get();
	meshdata.add_parameterized_mesh(V, F);

	Optimizer solver;
	solver.init();

	// Create app that directly interfaces with MeshData
	Application app;
	app.options.render_UV = true;
	app.solve_frame = [&](RowMatrixXd &X)
	{
		solver.step();
		X = solver.X;
	};
    app.start();

	std::cout << "Press space to start/stop the solver" << std::endl;

	// Export per-iteration timings/counters to csv
	Logger &log = Logger::get();
	log.write_all("octopus"); // output/octopus/octopus.csv

	return EXIT_SUCCESS;
}

static inline void end_frame()
{
	const MeshData &meshdata = MeshData::get();
	mclSetCounter("num_verts", meshdata.get_rest().rows());
	mclSetCounter("num_triangles", meshdata.get_faces().rows());
}

void Optimizer::init()
{
	MicroTimer t;
	const MeshData &meshdata = MeshData::get();
	const RowMatrixXd &V = meshdata.get_rest(); // 3D verts
	const RowMatrixXi &F = meshdata.get_elements(); // faces
	X = meshdata.get_initializer(); // 2D UV initializer
	std::cout << meshdata.print_stats() << std::endl;

	mclAssert(V.cols()==3);
	mclAssert(F.cols()==3);
	mclAssert(X.cols()==2);

	scaf_data = igl::triangle::SCAFData();
	Eigen::VectorXi b; Eigen::MatrixXd bc;
	igl::triangle::scaf_precompute(V, F, X,
		scaf_data,
		igl::MappingEnergyType::SYMMETRIC_DIRICHLET,
		b, bc, 0);

	Logger &log = Logger::get();
	log.end_frame = end_frame;
	end_frame(); // add meshdata to log at iter = -1
	mclSetValue("init_ms", t.elapsed_ms());
}

void Optimizer::step()
{
	mclStartFrame();
	igl::triangle::scaf_solve(scaf_data, 1);
	X = scaf_data.w_uv.topRows(X.rows());
	// end of step(), end_frame is called internally
}
