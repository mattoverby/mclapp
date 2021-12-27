// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include <iostream>

#include "MCL/Logger.hpp"
#include "MCL/Application.hpp"
#include "MCL/MeshData.hpp"
#include "MCL/AssertHandler.hpp"
#include "MCL/RenderCache.hpp"
#include "MCL/ReadMSH4.hpp"
#include "MCL/XForm.hpp"
#include "MCL/Centerize.hpp"

#include <igl/arap.h>

using namespace Eigen;
using namespace mcl;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXi;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXd;

class Deformer
{
public:
	igl::ARAPData arap_data;
	RowMatrixXd X;
	void init();
	void step();
	void render_pins();
	std::vector<Vector3d> pin_loc;
};

int main(int argc, char *argv[])
{
	(void)(argc);
	(void)(argv);

	std::string msh = MCL_APP_ROOT_DIR "/data/dillo.msh";
	MatrixXd V0;
	MatrixXi T0;
	if (!mcl::readMSH4(msh, V0, T0)) {
		std::cerr << "Failed to read " << msh << std::endl;
		return EXIT_FAILURE;
	}

	// Rotate dillo mesh so it faces camera
	mcl::centerize(V0);
	XForm<double> xf = XForm<double>::make_rotate(90,Vector3d(1,0,0));
	xf.apply(V0);
	xf = XForm<double>::make_rotate(180,Vector3d(0,0,1));
	xf.apply(V0);

	// MeshData is a global singleton that stores geometry info
	// and computes prim/facet/etc mappings.
	RowMatrixXd V = V0;
	RowMatrixXi T = T0;
	MeshData &meshdata = MeshData::get();
	meshdata.add_mesh(V, T);

	// Initialize the deformer
	Deformer solver;
	solver.init();
	solver.render_pins();

	// Create app that directly interfaces with MeshData
	Application app;
	app.options.name = "dillo";
	app.solve_frame = [&](RowMatrixXd &X)
	{
		solver.step();
		solver.render_pins();
		X = solver.X;
	};
    app.start();

	std::cout << "Press space to start/stop the solver" << std::endl;

	// Export per-iteration timings/counters to csv
	Logger &log = Logger::get();
	log.write_all(app.options.name); // output/dillo/dillo_log.csv
	return EXIT_SUCCESS;
}

void Deformer::init()
{
    // mclStart begins a named timer. When the scope
    // ends, the timer is stopped.
	mclStart("Deformer::init");

	const MeshData &meshdata = MeshData::get();
	const RowMatrixXd &V = meshdata.get_rest(); // 3D verts
	const RowMatrixXi &T = meshdata.get_elements(); // tets
	X = meshdata.get_initializer();
	std::cout << meshdata.print_stats() << std::endl;

	// Compute bounding box
	AlignedBox<double,3> box;
	for (int i=0; i<(int)X.rows(); ++i)
		box.extend(V.row(i).transpose());

	// Get indices of feet
	pin_loc.clear();
	std::vector<int> feet_inds;
	for (int i=0; i<(int)X.rows(); ++i)
	{
		if (X(i,1) < box.min()[1] + 1e-2)
		{
			feet_inds.emplace_back(i);
			pin_loc.emplace_back(X.row(i));
		}
	}

	Eigen::VectorXi b = Map<VectorXi>(feet_inds.data(), feet_inds.size());
	bool success = igl::arap_precomputation(V,T,3,b,arap_data);
	mclAssert(success, "failed to init arap");
}

void Deformer::step()
{
/*
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
*/
}

void Deformer::render_pins()
{
	// RenderCache is a singleton that's used for
    // visual debugging. Any data added to RenderCache
    // will be drawn on the next frame and cleared.
    // It not efficient.
    RenderCache &cache = RenderCache::get();
	int np = arap_data.b.rows();
	for (int i=0; i<np; ++i)
		cache.add_point<3>(X.row(arap_data.b[i]));

	cache.add_sphere(Vector3d(0,1,0), 1);
}