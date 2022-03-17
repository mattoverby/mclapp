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
	void render();
	std::map<int,Vector3d> pins;
	std::vector<int> hand;
	AlignedBox<double,3> feet_box, hand_box;
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
	solver.render();

	// Create app that directly interfaces with MeshData
	Application app;
	app.options.name = "dillo";
	app.solve_frame = [&](RowMatrixXd &X)
	{
		solver.step();
		X = solver.X;
		solver.render();
	};
    app.start();

	std::cout << "Press space to start/stop the solver" << std::endl;

	// Export per-iteration timings to csv
	std::string out_csv = app.options.name+".csv";
    mcl::WriteLogCSV(out_csv.c_str());
	return EXIT_SUCCESS;
}

void Deformer::init()
{
    mcl::ResetLog();
    mcl::TimedScope("Deformer::init");

	const MeshData &meshdata = MeshData::get();
	const RowMatrixXd &V = meshdata.get_rest(); // 3D verts
	const RowMatrixXi &T = meshdata.get_elements(); // tets
	X = meshdata.get_initializer();
	std::cout << meshdata.print_stats() << std::endl;

	// Compute bounding box
	AlignedBox<double,3> box;
	for (int i=0; i<(int)X.rows(); ++i)
		box.extend(V.row(i).transpose());

	// Get indices of left hand and feet
	pins.clear();
	hand.clear();
	feet_box.setEmpty();
	hand_box.setEmpty();
	std::vector<int> b_inds;
	for (int i=0; i<(int)X.rows(); ++i)
	{
		if (X(i,0) > box.max()[0] - 1e-2)
		{
			pins.emplace(i,X.row(i));
			hand.emplace_back(i);
			b_inds.emplace_back(i);
			hand_box.extend(X.row(i).transpose());
		}
		if (X(i,1) < box.min()[1] + 1e-2)
		{
			pins.emplace(i,X.row(i));
			b_inds.emplace_back(i);
			feet_box.extend(X.row(i).transpose());
		}
	}

	feet_box.extend(feet_box.min()-Vector3d::Ones()*0.01);
	feet_box.extend(feet_box.max()+Vector3d::Ones()*0.01);

	Eigen::VectorXi b = Map<VectorXi>(b_inds.data(), b_inds.size());
	bool success = igl::arap_precomputation(V,T,3,b,arap_data);
	mclAssert(success, "failed to init arap");
}

void Deformer::step()
{
    // A frame can be a timestep, iteration, whatever.
    // Timers/etc are recorded per-frame.
    // As a special case, the runtime of this
    // function is recorded in seconds.
    mcl::IncrementFrame();

	// Get the number of frames, stored in logger
	// and updated by the mclStartFrame function.
	int num_frames = mcl::Logger::get().frame_number();
	
	// Slowly move hand
	for (int i=0; i<(int)hand.size() && num_frames<30; ++i)
	{
		mclAssert(pins.count(hand[i])>0);
		pins[hand[i]] += Vector3d(0.01,0,0);
	}

	// Set pins
	RowMatrixXd bc(pins.size(), 3);
	mclAssert((int)pins.size() == (int)arap_data.b.rows());
	for (int i=0; i<(int)arap_data.b.rows(); ++i)
	{
		int idx = arap_data.b[i];
		mclAssert(pins.count(idx)>0);
		bc.row(i) = pins[idx];
	}
	
	// Solve arap
	igl::arap_solve(bc, arap_data, X);
}

void Deformer::render()
{
	// Mesh rendering already handled by Application and MeshData
	// Add auxiliary and debug rendering with RenderCache
	Vector3d color = Vector3d(0.7,0.1,0.1);
	RenderCache &cache = RenderCache::get();

	// Update hand box
	hand_box.setEmpty();
	int n_hand = hand.size();
	for (int i=0; i<n_hand; ++i)
		hand_box.extend(X.row(hand[i]).transpose());

	// Set render cache
	Vector3d c_offset = Vector3d(0,0.05,0.2); // looks nicer
	double r = hand_box.sizes().maxCoeff()*0.5;
	cache.add_sphere(hand_box.center()+c_offset, r, 3, color);
	cache.add_box(feet_box.min(), feet_box.max(), color);
}
