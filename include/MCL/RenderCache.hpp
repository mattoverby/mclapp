// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_RENDERCACHE_HPP
#define MCL_RENDERCACHE_HPP 1

#include "Singleton.hpp"
#include <Eigen/Geometry>
#include <vector>

namespace mcl
{

//
// Used for debugging
// Add points/lines that are rendered in Application on the next draw call.
// Very inefficient, yet convenient.
//
class RenderCache : public Singleton<RenderCache>
{
public:
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXd;
    typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXi;
    
	// Clears cached data
	void clear();
	
	template<int DIM>
	void add_point(
        const Eigen::Matrix<double,DIM,1> &p,
        const Eigen::Vector3d &c = Eigen::Vector3d(1,0,0));

	template<int DIM>
	void add_line(
        const Eigen::Matrix<double,DIM,1> &p0,
        const Eigen::Matrix<double,DIM,1> &p1,
        const Eigen::Vector3d &c = Eigen::Vector3d(1,0,0));

    void add_triangles(
        const Eigen::MatrixXd &V,  // cols = 2 or 3
        const Eigen::MatrixXi &F,  // cols = 3
        const Eigen::MatrixXd &C); // cols = 3

	void add_sphere(
		const Eigen::Vector3d &center,
		double radius,
		int subdiv=1,
        const Eigen::Vector3d &c=Eigen::Vector3d(1,0,0));

	template<int DIM>
    void add_box(
        const Eigen::Matrix<double,DIM,1> &bmin,
        const Eigen::Matrix<double,DIM,1> &bmax,
        const Eigen::Vector3d &c = Eigen::Vector3d(1,0,0));

    // Used by mcl::Application for rendering with for libigl
    void append_points(Eigen::MatrixXd &P, Eigen::MatrixXd &C);
    void append_lines(Eigen::MatrixXd &E0, Eigen::MatrixXd &E1, Eigen::MatrixXd &C);
    void append_triangles(RowMatrixXd &V, RowMatrixXi &F, RowMatrixXd &C);
    
protected:
    std::vector<Eigen::Vector3d> pts;
    std::vector<Eigen::Vector3d> pt_colors;
    std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d> > lines;
    std::vector<Eigen::Vector3d> line_colors;
    Eigen::MatrixXd tri_V, tri_C; // per-tri colors
    Eigen::MatrixXi tri_F;

}; // class rendercache

} // ns mcl

#endif
