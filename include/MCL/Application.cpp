// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include "Application.hpp"
#include "MeshData.hpp"
#include "Screenshot.hpp"
#include "RenderCache.hpp"
#include "Logger.hpp"
#include "MCL/AssertHandler.hpp"

#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <igl/png/readPNG.h>
#include <igl/opengl/glfw/Viewer.h>
#ifdef MCL_APP_USE_IMGUI
    #include <igl/opengl/glfw/imgui/ImGuiMenu.h>
    #include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
    #include <igl/opengl/glfw/imgui/ImGuiTraits.h>
#endif

#include <experimental/filesystem>

namespace mcl
{

using namespace Eigen;

struct Texture
{
	typedef Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> MatType;
	MatType R, G, B, A;
	void clear() { R = MatType(); G = MatType(); B = MatType(); A = MatType(); }
};

struct RuntimeOptions
{
	bool solve_next_frame;
	bool solved_last_frame;
	bool screenshot_each_frame;
	std::vector<Vector3d> mesh_colors; // default per-mesh colors
	int scale_uv;
	int matcap_index; // 0 = none
	std::vector<std::string> matcap_labels;
	std::vector<Texture> matcaps;
	Texture ref_tex;
#ifdef MCL_APP_USE_IMGUI
	igl::opengl::glfw::imgui::ImGuiMenu gui;
#endif
	int gui_plugin_idx;
	mcl::Screenshot screenshotter;
	mcl::Application *app_ptr;
	igl::opengl::glfw::Viewer *viewer_ptr;
	Interface::RowMatrixXd X;
	void init_runtime();
	RuntimeOptions() :
		solve_next_frame(false),
		solved_last_frame(false),
		screenshot_each_frame(false),
		scale_uv(4),
		matcap_index(0),
		gui_plugin_idx(-1),
		app_ptr(nullptr),
		viewer_ptr(nullptr)
		{ init_runtime(); }
};
static RuntimeOptions runtime;

static inline void callback_draw_viewer_menu();

Application::Application()
{
	// Set some defaults based on input data
	const MeshData &meshdata = MeshData::get();
	if (meshdata.is_texture_param()) { options.render_UV = true; }
}

Application::~Application()
{
}

void Application::start()
{
    viewer_ptr = std::make_shared<igl::opengl::glfw::Viewer>();
    viewer_ptr->plugins.emplace_back(this);
    viewer_ptr->launch();
}

void Application::init(igl::opengl::glfw::Viewer *viewer_)
{
	if (options.name.size()==0) { options.name = "mclapp"; }
	runtime = RuntimeOptions();
	runtime.app_ptr = this;
	viewer = viewer_;
	runtime.viewer_ptr = viewer;

	// Use app runtime variables to deal with solve/animate etc.
	viewer->core().is_animating = true;

#ifdef MCL_APP_USE_IMGUI
	runtime.gui.callback_draw_viewer_menu = &callback_draw_viewer_menu;
	if (options.show_gui)
	{
    	runtime.gui_plugin_idx = viewer->plugins.size();
	    viewer->plugins.push_back(&runtime.gui);
	}
#endif

	const MeshData &meshdata = MeshData::get();
	mclAssert(meshdata.get_initializer().rows()>0, "you must init meshdata before app");
	runtime.X = meshdata.get_initializer();
	redraw(runtime.X);

	// Print help
	key_pressed((unsigned int)('h'), 0);
}

bool Application::pre_draw()
{
	if (!options.animate && !runtime.solve_next_frame)
		return false;

	runtime.solve_next_frame = false;

	if (solve_frame != nullptr)
	{
		if (pre_solve_callback != nullptr)
			pre_solve_callback();

		solve_frame(runtime.X);
		runtime.solved_last_frame = true;

		if (post_solve_callback != nullptr)
			post_solve_callback();

		redraw(runtime.X);
	}

	return false;
}

bool Application::post_draw()
{
	if (options.quit_next_frame)
	{ 
		glfwSetWindowShouldClose(viewer->window, true);
	}

	// If we are exporting frames and have not set the first frame yet,
	// we should do so now. Frames are otherwise exported post_draw
	if (runtime.screenshot_each_frame && runtime.screenshotter.frame_counter==0)
	{
		runtime.screenshotter.save_frame(*viewer);
	}

	if (runtime.screenshot_each_frame && runtime.solved_last_frame)
	{
		runtime.screenshotter.save_frame(*viewer);
	}

	// Should we stop animating?
	Logger &log = Logger::get();
	if (log.pause_simulation)
	{
		log.pause_simulation = false;
		options.animate = false;
	}

	runtime.solved_last_frame = false;
	return false;
}

static inline void toggle_gui()
{
    if (!runtime.app_ptr || !runtime.viewer_ptr)
        return;
#ifdef MCL_APP_USE_IMGUI
    runtime.app_ptr->options.show_gui = !runtime.app_ptr->options.show_gui;
    if (runtime.app_ptr->options.show_gui)
    {
        mclAssert(runtime.gui_plugin_idx == -1);
        runtime.gui_plugin_idx = runtime.viewer_ptr->plugins.size();
	    runtime.viewer_ptr->plugins.push_back(&runtime.gui);    
    }
    else
    {
        int pidx = runtime.gui_plugin_idx;
        mclAssert(pidx >= 0 && pidx < (int)runtime.viewer_ptr->plugins.size());
        runtime.viewer_ptr->plugins.erase(runtime.viewer_ptr->plugins.begin()+pidx);
        runtime.gui_plugin_idx = -1;
    }
#endif
}

static inline void cycle_matcap()
{
    runtime.matcap_index = (runtime.matcap_index + 1) % runtime.matcap_labels.size();
    //std::cout << "Loading matcap: " << runtime.matcap_index << "/" << runtime.matcaps.size() << std::flush;
    //std::cout << ": " << runtime.matcap_labels[runtime.matcap_index] << std::endl;
}

bool Application::key_pressed(unsigned int key_, int mod)
{
	(void)(mod);
	char key = std::tolower(char(key_));
	bool needs_redraw = false;

	//printf("pressed: %c\n", key);

	auto print_help = [&]()
	{
		std::stringstream ss;
		ss << "Keys:";
		ss << "\n\t h: help";
		ss << "\n\t space: toggle animate";
		ss << "\n\t p: solve frame";
    	ss << "\n\t m: cycle matcap";
#ifdef MCL_APP_USE_IMGUI
        ss << "\n\t g: toggle gui";
#endif
		printf("%s\n", ss.str().c_str());
	};

	switch (key)
	{
		case 'h' : { print_help(); } break;
		case 'p' : { runtime.solve_next_frame = true; } break;
		case ' ' : { options.animate = !options.animate; } break;
        case 'g' : { toggle_gui(); } break;
        case 'm' : { cycle_matcap(); needs_redraw = true; } break;
	}

	if (needs_redraw)
		redraw(runtime.X);

	if (key_pressed_callback != nullptr)
		key_pressed_callback(key);

	return false;
}

void Application::redraw(const RowMatrixXd &X)
{
	const MeshData &meshdata = MeshData::get();
	bool has_x = X.rows() == meshdata.get_rest().rows();

	double lighting_factor = 0.6;
	// if matcap, factor = 0.9
	if (meshdata.dim() == 2) { lighting_factor = 0; }

	// Update viewer settings
	viewer->data().clear();
	viewer->core().lighting_factor = lighting_factor;
	viewer->core().background_color = Vector4f(1,1,1,1);
	viewer->core().orthographic = false;
	viewer->data().line_color = Vector4f(0.2,0.2,0.2,0.2);
	viewer->data().double_sided = true;

	if (has_x)
	{
		mclAssert(X.rows() == meshdata.get_rest().rows());
		mclAssert(X.rows() == meshdata.get_initializer().rows());
		mclAssert(X.rows()>0);

		RowMatrixXd V, C, N;
		RowMatrixXi F;
		append_mesh(X, V, F, C, N);

		mclAssert(V.cols() == 3);
		viewer->data().set_face_based(true);
		viewer->data().set_mesh(V, F);
		if (C.rows()>0) { viewer->data().set_colors(C); }
		if (N.rows()>0) { viewer->data().set_normals(N); }
		
	    // Set texture from matcap if desired
        if (runtime.matcap_index > 0)
        {
            const Texture& tex = runtime.matcaps[runtime.matcap_index-1];
            //Texture::MatType texA = (tex.A.cast<float>() * mesh_opacity).cast<unsigned char>();
        	viewer->data().show_texture = true;
            viewer->data().use_matcap = true;
            viewer->core().lighting_factor = 0.9;
        	viewer->data().set_texture(tex.R, tex.G, tex.B, tex.A);
        }
        else if (meshdata.is_texture_param() && !options.render_UV)
	    {
			double rest_rad = meshdata.get_rest_radius();
		    double scale = (1.0 / rest_rad) * double(runtime.scale_uv);
		    viewer->data().show_texture = true;
		    viewer->core().lighting_factor = 0.9;
		    viewer->data().V_material_diffuse = MatrixXd::Ones(V.rows(), 3);
		    viewer->data().set_uv(X*scale, F); // scale tex for visibility
		    if (runtime.ref_tex.R.rows())
		    {
			    viewer->data().set_face_based(false);
			    viewer->data().set_texture(
				    runtime.ref_tex.R, runtime.ref_tex.G,
				    runtime.ref_tex.B, runtime.ref_tex.A);
		    }
	    }
	}
    else
    {
    	RowMatrixXd V, C;
		RowMatrixXi F;
        RenderCache &cache = RenderCache::get();
        cache.append_triangles(V, F, C);
        if (V.rows()>0)
        {
            viewer->data().set_mesh(V, F);
            viewer->data().set_colors(C);
        }
    }

	// Add points and lines from cache
    RenderCache &cache = RenderCache::get();
	MatrixXd pts, pt_colors;
	MatrixXd edge0, edge1, edge_colors;
	cache.append_points(pts, pt_colors);
	cache.append_lines(edge0, edge1, edge_colors);
	if (pts.rows()) { viewer->data().add_points(pts, pt_colors); }
	if (edge0.rows()) { viewer->data().add_edges(edge0, edge1, edge_colors); }
	cache.clear();
}

void Application::append_mesh(const RowMatrixXd &X,
    RowMatrixXd &V, RowMatrixXi &F,
	RowMatrixXd &C, RowMatrixXd &N)
{
	const MeshData &meshdata = MeshData::get();
	F = meshdata.get_faces();
	V = X;
	bool compute_colors = true;
	
	if (meshdata.is_texture_param() && !options.render_UV)
	{
	    compute_colors = false;
		V = meshdata.get_rest();
		//C = RowMatrixXd::Ones(F.rows(), 3)*0.7;
	}
	else if (X.cols() == 2)
	{
		V = RowMatrixXd::Zero(X.rows(), 3);
		V.leftCols(2) = X;
	}

    // Flat shading is determined by the normals
    // Use per_corner_normals for smooth shading.
    if (compute_colors)
    {
    	int n_meshes = meshdata.num_meshes();
    	C = RowMatrixXd::Zero(F.rows(), 3);
		const VectorXi &F_offset = meshdata.get_face_offsets();
		for (int i=0; i<n_meshes; ++i)
		{
			int nf = F_offset[i+1] - F_offset[i];
			int c_idx = (options.start_mesh_color + i) % int(runtime.mesh_colors.size());
			C.block(F_offset[i],0,nf,3).col(0).array() = runtime.mesh_colors[c_idx][0];
			C.block(F_offset[i],0,nf,3).col(1).array() = runtime.mesh_colors[c_idx][1];
			C.block(F_offset[i],0,nf,3).col(2).array() = runtime.mesh_colors[c_idx][2];
		}

	    RenderCache &cache = RenderCache::get();
	    cache.append_triangles(V, F, C);
    }

    // Use flat shading if rendering 2D
	if (meshdata.is_texture_param())
	{
	    if (options.flat_shading || options.render_UV) { igl::per_face_normals(V, F, N); } // 2D
	    else { igl::per_corner_normals(V, F, 50, N); } // TODO deal with seams
	}
	else if (options.flat_shading) { igl::per_face_normals(V, F, N); }
	else { igl::per_corner_normals(V, F, 50, N); }

} // end append mesh

void RuntimeOptions::init_runtime()
{
	mesh_colors = {
		Vector3d(166,206,227)*(1.0/255.0),
		Vector3d(168,213,128)*(1.0/255.0),
		Vector3d(251,154,153)*(1.0/255.0),
		Vector3d(253,191,111)*(1.0/255.0),
		Vector3d(202,178,214)*(1.0/255.0),
		Vector3d(255,255,153)*(1.0/255.0),
		Vector3d(200,200,200)*(1.0/255.0) // grayscale, always last
	};

	// Load matcaps
	matcap_labels.clear();
	matcaps.clear();
	matcap_labels.emplace_back("none");
	std::string mc_dir = MCL_APP_ROOT_DIR "/data/matcaps";
	for (const auto &file : std::experimental::filesystem::directory_iterator(mc_dir))
	{
		std::string fn = std::experimental::filesystem::absolute(file.path());
		std::string label = file.path().stem();
		Texture tex;
		if (igl::png::readPNG(fn, tex.R, tex.G, tex.B, tex.A))
		{
			matcaps.emplace_back(tex);
			matcap_labels.emplace_back(label);
		}
	}

	// Load reference texture
	std::string fn = MCL_APP_ROOT_DIR "/data/texture_bb.png";
	if (!igl::png::readPNG(fn, ref_tex.R, ref_tex.G, ref_tex.B, ref_tex.A))
	{
		std::cerr << "Failed to load " << fn << std::endl;
		ref_tex.clear();
	}
}

static inline void callback_draw_viewer_menu()
{
#ifdef MCL_APP_USE_IMGUI
	const MeshData &meshdata = MeshData::get();
	if (!runtime.app_ptr || !runtime.viewer_ptr)
		return;

	bool needs_render_update = false;

	// Scene info
	{
		std::stringstream header_text;
		header_text << runtime.app_ptr->options.name.c_str();
		header_text << "\n\t#verts: " << meshdata.get_rest().rows();
		header_text << "\n\t#elems: " << meshdata.get_elements().rows();
    	header_text << "\n\t#bnd elems: " << meshdata.get_facets().rows();
		ImGui::TextUnformatted(header_text.str().c_str());
	}

	if (ImGui::Button("screenshot"))
	{
		std::string png_out = "screenshot_" +
			std::to_string(runtime.screenshotter.frame_counter) + ".png";
		std::cout << "Saving \"" << png_out << "\"" << std::endl;
		runtime.screenshotter.save_frame(runtime.app_ptr->get_viewer(), png_out);
		runtime.screenshotter.frame_counter++;
	}
	if (ImGui::Checkbox("screenshot background", &runtime.screenshotter.render_background)){}
	if (ImGui::Checkbox("screenshot each frame", &runtime.screenshot_each_frame)){}
	if (ImGui::CollapsingHeader("solver", ImGuiTreeNodeFlags_DefaultOpen))
	{
        if (ImGui::Button("solve frame")) { runtime.solve_next_frame=true; }
		ImGui::Checkbox("animate (space)", &runtime.app_ptr->options.animate);		
	}
	if (ImGui::CollapsingHeader("rending", ImGuiTreeNodeFlags_DefaultOpen))
	{
	    bool show_lines = runtime.viewer_ptr->data().show_lines;
        if (ImGui::Checkbox("show lines (l)", &show_lines))
        {
            runtime.viewer_ptr->data().show_lines = show_lines;
            needs_render_update = true;
        }
        if (ImGui::Checkbox("flat shading", &runtime.app_ptr->options.flat_shading)) { needs_render_update = true; }
        if (ImGui::Checkbox("render UV", &runtime.app_ptr->options.render_UV)){ needs_render_update = true; }
        if (ImGui::SliderInt("scale UV", &runtime.scale_uv, 1, 10)){ needs_render_update = true; }
        if (ImGui::Combo("matcap (m)", &runtime.matcap_index, runtime.matcap_labels)) { needs_render_update = true; }
    }
    
    if (runtime.app_ptr->draw_gui_callback != nullptr)
    {
        if (runtime.app_ptr->draw_gui_callback()) { needs_render_update = true; }
    }
    
	if (needs_render_update) { runtime.app_ptr->redraw(runtime.X); }
#endif
} // end imgui menu

} // end ns mcl
