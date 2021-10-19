
#include "Screenshot.hpp"
#ifndef MCL_HEADLESS
#include <GLFW/glfw3.h>
#include <igl/png/writePNG.h>
#include <iomanip>

namespace mcl
{

void Screenshot::save_frame(igl::opengl::glfw::Viewer &viewer, std::string fn)
{
	if (fn.size()==0)
	{
		std::stringstream ss;
		ss << MCL_APP_OUTPUT_DIR << "/" << std::setfill('0') << std::setw(5) << frame_counter << ".png";
		frame_counter++;
		fn = ss.str();
	}

	int w, h;
	glfwGetFramebufferSize(viewer.window, &w, &h);

    // Allocate temporary buffers
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(w,h);
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(w,h);
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(w,h);
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(w,h);

    // Draw the scene in the buffers
    viewer.core().draw_buffer(viewer.data(),false,R,G,B,A);
	if (render_background) {
		A.fill(255);
	}

    // Save it to a PNG
    igl::png::writePNG(R,G,B,A,fn.c_str());
}

} // ns gini

#endif