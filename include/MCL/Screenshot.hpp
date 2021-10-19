// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_SCREENSHOT_HPP
#define MCL_SCREENSHOT_HPP 1

#include <igl/opengl/glfw/Viewer.h>

namespace mcl
{

class Screenshot
{
public:
	int frame_counter;
	bool render_background;
	bool rendered_init;
	Screenshot() :
		frame_counter(0),
		render_background(0),
		rendered_init(0)
		{}

	// If filename is empty, it's auto computed from the frame_counter.
	// Otherwise, the frame_counter is NOT incremented.
	void save_frame(igl::opengl::glfw::Viewer &viewer, std::string fn="");
};

} // end namespace mcl

#endif
