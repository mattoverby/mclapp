// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_APPLICATION_HPP
#define MCL_APPLICATION_HPP 1

#include <functional>
#include <Eigen/Core>
#include <igl/opengl/glfw/ViewerPlugin.h>
#include "Interface.hpp"

namespace mcl
{

class Application : public igl::opengl::glfw::ViewerPlugin, public Interface
{
public:
	using Interface::RowMatrixXi;
	using Interface::RowMatrixXd;

	Application();
	~Application();

	void init(igl::opengl::glfw::Viewer *viewer_);
	igl::opengl::glfw::Viewer &get_viewer() { return *viewer; }

    void start() override;
	void redraw(const RowMatrixXd &X);

	std::function<void(char key)> key_pressed_callback;
	std::function<bool()> draw_gui_callback; // true if update render

protected:
	bool pre_draw();
	bool post_draw();
	bool key_pressed(unsigned int key, int mod);

    std::shared_ptr<igl::opengl::glfw::Viewer> viewer_ptr; // used if start() is called

	void append_mesh(const RowMatrixXd &X,
		RowMatrixXd &V, RowMatrixXi &F,
		RowMatrixXd &C, RowMatrixXd &N);

	void append_inverted_elements(const RowMatrixXd &X);
};

} // end namespace mcl

#endif
