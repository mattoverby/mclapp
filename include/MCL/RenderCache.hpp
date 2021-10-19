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

    // Add points to bottom of matrix (for libigl)
    void append_points(Eigen::MatrixXd &P, Eigen::MatrixXd &C);

    // Add lines to bottom of matrix (for libigl)
    void append_lines(Eigen::MatrixXd &E0, Eigen::MatrixXd &E1, Eigen::MatrixXd &C);
    
protected:
    std::vector<Eigen::Vector3d> pts;
    std::vector<Eigen::Vector3d> pt_colors;
    std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d> > lines;
    std::vector<Eigen::Vector3d> line_colors;

}; // class rendercache

} // ns mcl

#endif
