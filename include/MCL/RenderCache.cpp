// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include "RenderCache.hpp"

namespace mcl
{

// Clears cached data
void RenderCache::clear()
{
    pts.clear();
    pt_colors.clear();
    lines.clear();
    line_colors.clear();
}

template<int DIM>
void RenderCache::add_point(const Eigen::Matrix<double,DIM,1> &p,
    const Eigen::Vector3d &c)
{
    pt_colors.emplace_back(c);
    if (DIM==2) { pts.emplace_back(Eigen::Vector3d(p[0], p[1], 0)); }
    else if (DIM==3) { pts.emplace_back(p.template head<3>()); }
    else { pt_colors.pop_back(); }
}

template void mcl::RenderCache::add_point<2>(const Eigen::Vector2d&, const Eigen::Vector3d&);
template void mcl::RenderCache::add_point<3>(const Eigen::Vector3d&, const Eigen::Vector3d&);

template<int DIM>
void RenderCache::add_line(const Eigen::Matrix<double,DIM,1> &p0, const Eigen::Matrix<double,DIM,1> &p1,
    const Eigen::Vector3d &c)
{
    using namespace Eigen;
    line_colors.emplace_back(c);
    if (DIM==2) { lines.emplace_back(Vector3d(p0[0],p0[1],0), Vector3d(p1[0],p1[1],0)); }
    else if (DIM==3) { lines.emplace_back(p0.template head<3>(), p1.template head<3>()); }
    else { line_colors.pop_back(); }
}

template void mcl::RenderCache::add_line<2>(const Eigen::Vector2d&, const Eigen::Vector2d&, const Eigen::Vector3d&);
template void mcl::RenderCache::add_line<3>(const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&);

// Add points to bottom of matrix
void RenderCache::append_points(Eigen::MatrixXd &P, Eigen::MatrixXd &C)
{
    int np = std::min(pts.size(), pt_colors.size());
    if (np == 0) { return; }
    int np_prev = std::min(P.rows(), C.rows());
    P.conservativeResize(np_prev+np, 3);
    C.conservativeResize(np_prev+np, 3);
    for (int i=0; i<np; ++i)
    {
        P.row(np_prev+i) = pts[i];
        C.row(np_prev+i) = pt_colors[i];
    }
}

// Add lines to bottom of matrix
void RenderCache::append_lines(Eigen::MatrixXd &E0, Eigen::MatrixXd &E1, Eigen::MatrixXd &C)
{
    int ne = std::min(lines.size(), line_colors.size());
    if (ne == 0) { return; }
    int ne_prev = std::min(std::min(E0.rows(), E1.rows()), C.rows());
    E0.conservativeResize(ne_prev+ne, 3);
    E1.conservativeResize(ne_prev+ne, 3);
    C.conservativeResize(ne_prev+ne, 3);
    for (int i=0; i<ne; ++i)
    {
        E0.row(ne_prev+i) = lines[i].first;
        E1.row(ne_prev+i) = lines[i].second;
        C.row(ne_prev+i) = line_colors[i];
    }
}

} // ns mcl