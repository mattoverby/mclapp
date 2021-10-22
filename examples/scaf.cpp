// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include <iostream>

#include "MCL/Logger.hpp"
#include "MCL/Application.hpp"
#include "MCL/MeshData.hpp"
#include "MCL/AssertHandler.hpp"
#include "MCL/RenderCache.hpp"

#include <igl/triangle/scaf.h>
#include <igl/readOBJ.h>

using namespace Eigen;
using namespace mcl;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXi;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXd;

class Optimizer
{
public:
	igl::triangle::SCAFData scaf_data;
	RowMatrixXd X;
	void init();
	void step();
	void draw_scaf();
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
	log.write_all("camel"); // output/camel/camel_log.csv

	return EXIT_SUCCESS;
}

void Optimizer::init()
{
    // mclStart begins a named timer. When the scope
    // ends, the timer is stopped.
	mclStart("Optimizer::init");

	const MeshData &meshdata = MeshData::get();
	const RowMatrixXd &V = meshdata.get_rest(); // 3D verts
	const RowMatrixXi &F = meshdata.get_elements(); // faces
	X = meshdata.get_initializer(); // 2D UV initializer
	std::cout << meshdata.print_stats() << std::endl;

    // Asserts with or without message.
	mclAssert(V.cols()==3, "Bad cols in V");
	mclAssert(F.cols()==3);
	mclAssert(X.cols()==2);

	scaf_data = igl::triangle::SCAFData();
	Eigen::VectorXi b; Eigen::MatrixXd bc;
	igl::triangle::scaf_precompute(V, F, X,
		scaf_data,
		igl::MappingEnergyType::SYMMETRIC_DIRICHLET,
		b, bc, 0);

	draw_scaf();
}

void Optimizer::step()
{
    // A frame can be a timestep, iteration, whatever.
    // Counters/timers/etc are recorded per-frame.
    // As a special case, the runtime of this
    // function is recorded in seconds.
	mclStartFrame();
	
	// Step the integrator
	igl::triangle::scaf_solve(scaf_data, 1);
	X = scaf_data.w_uv.topRows(X.rows());
	
	// Counters (ints) and values (doubles) can
	// be recorded per-frame. Total/max/avg for all
	// frames is reported when log.write_... is called.
    mclSetCounter("example_counter", X.size());
    mclSetValue("example_value", X.norm());
    
    draw_scaf();
}

void Optimizer::draw_scaf()
{
    // RenderCache is a singleton that's used for
    // visual debugging. Any data added to RenderCache
    // will be drawn on the next frame and cleared.
    // It not efficient.
    RenderCache &cache = RenderCache::get();
    int nsv = scaf_data.w_uv.rows() - X.rows();
    MatrixXd V_scaf = scaf_data.w_uv.bottomRows(nsv);
    MatrixXd C_scaf = MatrixXd::Ones(scaf_data.s_T.rows(), 3);
    C_scaf *= 0.7; // gray
    cache.add_triangles(V_scaf, scaf_data.s_T, C_scaf);
}
