// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include "Headless.hpp"
#include "MeshData.hpp"
#include "RenderCache.hpp"

#include "MCL/AssertHandler.hpp"
#include "MCL/Logger.hpp"
//#include <chrono>
#include <experimental/filesystem>
#include <igl/writeOBJ.h>

namespace mcl
{

static inline void make_canary(const std::string &canary_file)
{
	mclAssert(canary_file.length()>0);
	std::experimental::filesystem::path p(canary_file);
	std::experimental::filesystem::create_directories(p.parent_path());
	//using namespace std::chrono;
	//const auto p1 = system_clock::now();
	//double timestamp = duration_cast<seconds>(p1.time_since_epoch()).count();
	//canary_file = output_dir + "canary_"+std::to_string(timestamp)+".txt";
	std::ofstream ksf;
	ksf.open(canary_file.c_str());
	ksf << "Delete this file to quit headless run";
	ksf.close();
}

static inline bool canary_is_alive(const std::string &canary_file)
{
	if (canary_file.length() == 0)
		return true;

	std::ifstream f(canary_file.c_str());
	return bool(f.good());
}

void Headless::start()
{
    if (options.name.size()==0) { options.name = "mclapp"; }
	std::string prefix = mcl::Logger::make_write_prefix(options.name);

	std::string canary_file = prefix + "canary.txt";
    make_canary(canary_file);
	mclAssert(canary_is_alive(canary_file));

    const MeshData &meshdata = MeshData::get();
    RowMatrixXd x = meshdata.get_initializer();

    mcl::Logger &log = mcl::Logger::get();
    if (options.export_frame_obj)
    {
        save_frame_obj(prefix, log.curr_frame, x);
    }

	while (true)
	{
        if (pre_solve_callback != nullptr)
            pre_solve_callback();

        if (solve_frame != nullptr)
            solve_frame(x);

        if (post_solve_callback != nullptr)
            post_solve_callback();

        if (options.export_frame_obj)    
            save_frame_obj(prefix, log.curr_frame, x);

		// Clear render cache so it doesn't grow, even
		// though we aren't rendering anything.
		RenderCache::get().clear();

		if (options.quit_next_frame || !canary_is_alive(canary_file))
			break;
	}

	// Remove canary if exists
	if (canary_is_alive(canary_file))
		std::experimental::filesystem::remove(canary_file);
}

void Headless::save_frame_obj(const std::string &prefix, int frame_idx, const RowMatrixXd &X)
{
    const MeshData &meshdata = MeshData::get();
	const RowMatrixXi &F = meshdata.get_faces();
	mclAssert(X.rows() == meshdata.get_rest().rows());
	Eigen::MatrixXd V = Eigen::MatrixXd::Zero(X.rows(), 3);
	if (X.cols() == 2) { V.leftCols(2) = X; }
	else { V = X; }
	std::string fn = prefix+options.name+"_"+std::to_string(frame_idx)+".obj";
	igl::writeOBJ(fn, V, F);
}

} // end namespace mcl
