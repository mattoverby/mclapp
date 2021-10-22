// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_INTERFACE_HPP
#define MCL_INTERFACE_HPP 1

#include <igl/opengl/glfw/Viewer.h>


namespace mcl
{

class Interface
{
public:
	typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXi;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXd;

    struct Options
    {
   		std::string name; // optional test name
		bool export_frame_obj; // export obj file each frame
		bool animate; // solve frames continuously
		bool quit_next_frame;
		// Application only:
		int start_mesh_color; // starting color from list
		bool flat_shading; // non-smooth triangles
		bool render_UV; // if parameterized mesh, render UV
		bool show_gui; // press g to toggle

        Options() :
   			name(""),
			export_frame_obj(false),
    		animate(false),
			quit_next_frame(false),
			start_mesh_color(0),
			flat_shading(true),
			render_UV(false),
			show_gui(true)
            {}
    } options;

    virtual void start() = 0;
    virtual void quit() { options.quit_next_frame = true; }

	// Get new vertex locations for the next frame (X)
	std::function<void(RowMatrixXd&)> solve_frame;
	std::function<void()> pre_solve_callback;
	std::function<void()> post_solve_callback;
};

} // end namespace mcl

#endif
