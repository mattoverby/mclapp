// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include "RenderCache.hpp"
#include "MCL/AssertHandler.hpp"
#include <mutex>
#include <iostream>
#include <map>

namespace mcl
{

std::mutex write_mutex;

// Clears cached data
void RenderCache::clear()
{
    std::lock_guard<std::mutex> guard(write_mutex);
    pts.clear();
    pt_colors.clear();
    lines.clear();
    line_colors.clear();
    tri_V = Eigen::MatrixXd();
    tri_F = Eigen::MatrixXi();
    tri_C = Eigen::MatrixXd();
}

template<int DIM>
void RenderCache::add_point(const Eigen::Matrix<double,DIM,1> &p,
    const Eigen::Vector3d &c)
{
    std::lock_guard<std::mutex> guard(write_mutex);
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
    std::lock_guard<std::mutex> guard(write_mutex);
    using namespace Eigen;
    line_colors.emplace_back(c);
    if (DIM==2) { lines.emplace_back(Vector3d(p0[0],p0[1],0), Vector3d(p1[0],p1[1],0)); }
    else if (DIM==3) { lines.emplace_back(p0.template head<3>(), p1.template head<3>()); }
    else { line_colors.pop_back(); }
}

template void mcl::RenderCache::add_line<2>(const Eigen::Vector2d&, const Eigen::Vector2d&, const Eigen::Vector3d&);
template void mcl::RenderCache::add_line<3>(const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&);

void RenderCache::add_triangles(
    const Eigen::MatrixXd &V,  // cols = 2 or 3
    const Eigen::MatrixXi &F,  // cols = 3
    const Eigen::MatrixXd &C_) // cols = 3
{
    std::lock_guard<std::mutex> guard(write_mutex);
    using namespace Eigen;
    if (V.rows()==0 || F.cols()!=3 || V.cols() > 3 || V.cols() < 2)
        return;

    int prev_v = tri_V.rows();
    tri_V.conservativeResize(prev_v+V.rows(), 3);
    tri_V.bottomRows(V.rows()).setZero();
    if (V.cols()==3) { tri_V.bottomRows(V.rows()) = V; }
    else { tri_V.block(prev_v,0,V.rows(),V.cols()) = V; }
    
    int prev_f = tri_F.rows();
    tri_F.conservativeResize(prev_f+F.rows(), 3);
    tri_F.bottomRows(F.rows()) = F;
    tri_F.bottomRows(F.rows()).array() += prev_v;

    MatrixXd C_tmp;
    const MatrixXd *C = nullptr;
    if (C_.rows() == F.rows()) { C = &C_; }
    else if (C_.rows() == V.rows() && C_.cols()==3)
    {
        C_tmp = MatrixXd::Zero(F.rows(),3);
        for (int i=0; i<(int)F.rows(); ++i)
        {
            C_tmp.row(i) += C_.row(F(i,0));
            C_tmp.row(i) += C_.row(F(i,1));
            C_tmp.row(i) += C_.row(F(i,2));
            C_tmp.row(i) *= 1.0 / 3.0;
        }
        C = &C_tmp;
    }
    else
    {
        C_tmp = MatrixXd::Ones(F.rows(),3);
        C = &C_tmp;
    }
    
    int prev_c = tri_C.rows();
    tri_C.conservativeResize(prev_c+C->rows(),3);
    tri_C.bottomRows(C->rows()) = *C;
}

// From https://github.com/caosdoar/spheres/blob/
// d16e148ca7d8346887da8f42dc2382a6e2c863c3/src/spheres.cpp
// License: MIT
class UnitSphereMesh
{
protected:
    struct Edge
    {
        uint32_t v0, v1;
        Edge(uint32_t v0_, uint32_t v1_) :
            v0(v0_ < v1_ ? v0_ : v1_),
            v1(v0_ < v1_ ? v1_ : v0_) {}
        inline bool operator <(const Edge &rhs) const
        {
            return v0 < rhs.v0 || (v0 == rhs.v0 && v1 < rhs.v1);
        }
    };
    static inline uint32_t subdivide_edge(uint32_t f0, uint32_t f1,
        const Eigen::Vector3d &v0, const Eigen::Vector3d &v1,
        UnitSphereMesh *io_mesh, std::map<Edge, uint32_t> &io_divisions)
    {
        const Edge edge(f0, f1);
        auto it = io_divisions.find(edge);
        if (it != io_divisions.end()) { return it->second; }
        Eigen::Vector3d v = ((v0+v1)*0.5).normalized();
        const uint32_t f = io_mesh->vertices.size();
        io_mesh->vertices.emplace_back(v);
        io_divisions.emplace(edge, f);
        return f;
    }
public:
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> faces;
    void make_icosahedron()
    {
        using namespace Eigen;
        const double t = (1.0 + std::sqrt(5.0)) / 2.0;
        vertices.clear();
        faces.clear();
        vertices.emplace_back(Vector3d(-1.0,  t, 0.0).normalized());
        vertices.emplace_back(Vector3d( 1.0,  t, 0.0).normalized());
        vertices.emplace_back(Vector3d(-1.0, -t, 0.0).normalized());
        vertices.emplace_back(Vector3d( 1.0, -t, 0.0).normalized());
        vertices.emplace_back(Vector3d(0.0, -1.0,  t).normalized());
        vertices.emplace_back(Vector3d(0.0,  1.0,  t).normalized());
        vertices.emplace_back(Vector3d(0.0, -1.0, -t).normalized());
        vertices.emplace_back(Vector3d(0.0,  1.0, -t).normalized());
        vertices.emplace_back(Vector3d( t, 0.0, -1.0).normalized());
        vertices.emplace_back(Vector3d( t, 0.0,  1.0).normalized());
        vertices.emplace_back(Vector3d(-t, 0.0, -1.0).normalized());
        vertices.emplace_back(Vector3d(-t, 0.0,  1.0).normalized());
        faces.emplace_back(0, 11, 5);
        faces.emplace_back(0, 5, 1);
        faces.emplace_back(0, 1, 7);
        faces.emplace_back(0, 7, 10);
        faces.emplace_back(0, 10, 11);
        faces.emplace_back(1, 5, 9);
        faces.emplace_back(5, 11, 4);
        faces.emplace_back(11, 10, 2);
        faces.emplace_back(10, 7, 6);
        faces.emplace_back(7, 1, 8);
        faces.emplace_back(3, 9, 4);
        faces.emplace_back(3, 4, 2);
        faces.emplace_back(3, 2, 6);
        faces.emplace_back(3, 6, 8);
        faces.emplace_back(3, 8, 9);
        faces.emplace_back(4, 9, 5);
        faces.emplace_back(2, 4, 11);
        faces.emplace_back(6, 2, 10);
        faces.emplace_back(8, 6, 7);
        faces.emplace_back(9, 8, 1);
    }
    void subdivide()
    {
        mclAssert(vertices.size()>0);
        mclAssert(faces.size()>0);
        using namespace Eigen;
        std::vector<Vector3d> prevV = vertices;
        std::vector<Vector3i> prevF = faces;
        uint32_t nf = prevF.size();

        //meshOut.vertices = meshIn.vertices;
        faces.clear();
        std::map<Edge, uint32_t> divisions; // Edge -> new vertex
        for (uint32_t i = 0; i < nf; ++i)
        {
            const uint32_t f0 = prevF[i][0];
            const uint32_t f1 = prevF[i][1];
            const uint32_t f2 = prevF[i][2];
            const Vector3d v0 = prevV[f0];
            const Vector3d v1 = prevV[f1];
            const Vector3d v2 = prevV[f2];
            const uint32_t f3 = subdivide_edge(f0, f1, v0, v1, this, divisions);
            const uint32_t f4 = subdivide_edge(f1, f2, v1, v2, this, divisions);
            const uint32_t f5 = subdivide_edge(f2, f0, v2, v0, this, divisions);
            faces.emplace_back(f0, f3, f5);
            faces.emplace_back(f3, f1, f4);
            faces.emplace_back(f4, f2, f5);
            faces.emplace_back(f3, f4, f5);
        }
    }
};

void RenderCache::add_sphere(
	const Eigen::Vector3d &center,
	double radius,
	int subdiv,
    const Eigen::Vector3d &c)
{
    using namespace Eigen;
    UnitSphereMesh mesh;
    mesh.make_icosahedron();

    for (int i=0; i<subdiv; ++i)
        mesh.subdivide();

    // Copy to Eigen matrices
    int nv = mesh.vertices.size();
    Eigen::MatrixXd V(nv, 3);
    for (int i=0; i<nv; ++i)
    {
        mesh.vertices[i] *= radius;
        mesh.vertices[i] += center;
        V.row(i) = mesh.vertices[i];
    }

    int nf = mesh.faces.size();
    Eigen::MatrixXi F(nf, 3);
    Eigen::MatrixXd C(nf, 3);
    for (int i=0; i<nf; ++i)
    {
        F.row(i) = mesh.faces[i];
        C.row(i) = c;
    }

    // Let add_triangles to do the rest
    add_triangles(V,F,C);
}

template<>
void RenderCache::add_box(
    const Eigen::Matrix<double,2,1> &bmin,
    const Eigen::Matrix<double,2,1> &bmax,
    const Eigen::Vector3d &color)
{
    using namespace Eigen;
    // d c
    // a b
    Eigen::MatrixXd V(4, 2);
    V.row(0) = bmin; // a
    V.row(1) = RowVector2d(bmax[0], bmin[1]); // b
    V.row(2) = bmax; // c
    V.row(3) = RowVector2d(bmin[0], bmax[1]); // d
    Eigen::MatrixXi F(2,3);
    F.row(0) = RowVector3i(0, 1, 3);
    F.row(1) = RowVector3i(1, 2, 3);
    Eigen::MatrixXd C(2,3);
    C.row(0) = color;
    C.row(1) = color;
    add_triangles(V,F,C);
}

template<>
void RenderCache::add_box(
    const Eigen::Matrix<double,3,1> &bmin,
    const Eigen::Matrix<double,3,1> &bmax,
    const Eigen::Vector3d &color)
{
    using namespace Eigen;

    // Bottom quad
    Vector3d a = bmin;
    Vector3d b(bmax[0], bmin[1], bmin[2]);
    Vector3d c(bmax[0], bmin[1], bmax[2]);
    Vector3d d(bmin[0], bmin[1], bmax[2]);
    // Top quad
    Vector3d e(bmin[0], bmax[1], bmin[2]);
    Vector3d f(bmax[0], bmax[1], bmin[2]);
    Vector3d g = bmax;
    Vector3d h(bmin[0], bmax[1], bmax[2]);

    Eigen::MatrixXd V(8, 3);
    V.row(0) = a;
    V.row(1) = b;
    V.row(2) = c;
    V.row(3) = d;
    V.row(4) = e;
    V.row(5) = f;
    V.row(6) = g;
    V.row(7) = h;
    Eigen::MatrixXi F(12, 3);
    F.row(0) = RowVector3i(0, 7, 4); // n = -x
    F.row(1) = RowVector3i(0, 3, 7); // n = -x
    F.row(2) = RowVector3i(1, 5, 6); // n = +x
    F.row(3) = RowVector3i(1, 6, 2); // n = +x
    F.row(4) = RowVector3i(0, 1, 2); // n = -y
    F.row(5) = RowVector3i(0, 2, 3); // n = -y
    F.row(6) = RowVector3i(4, 6, 5); // n = +y
    F.row(7) = RowVector3i(4, 7, 6); // n = +y
    F.row(8) = RowVector3i(1, 0, 4); // n = -z
    F.row(9) = RowVector3i(1, 4, 5); // n = -z
    F.row(10) = RowVector3i(3, 2, 7); // n = +z
    F.row(11) = RowVector3i(2, 6, 7); // n = +z
    Eigen::MatrixXd C(12,3);
    for (int i=0; i<12; ++i) { C.row(i)=color; }
    add_triangles(V,F,C);
}

// Add points to bottom of matrix
void RenderCache::append_points(Eigen::MatrixXd &P, Eigen::MatrixXd &C)
{
    std::lock_guard<std::mutex> guard(write_mutex);
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
    std::lock_guard<std::mutex> guard(write_mutex);
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

void RenderCache::append_triangles(RowMatrixXd &V, RowMatrixXi &F, RowMatrixXd &C)
{
    std::lock_guard<std::mutex> guard(write_mutex);
    if (tri_V.rows()==0 || tri_F.rows()==0 || tri_C.rows()==0)
        return;

    bool colors_per_vertex = C.rows() == V.rows();

    int prev_v = V.rows();
    int v_cols = std::min(3, (int)V.cols());
    if (v_cols == 0) { v_cols = tri_V.cols(); }
    mclAssert(tri_V.cols() >= v_cols);
    mclAssert(v_cols >= 2);
    V.conservativeResize(prev_v+tri_V.rows(), v_cols);
    V.bottomRows(tri_V.rows()) = tri_V.leftCols(v_cols);
    
    int prev_f = F.rows();
    F.conservativeResize(prev_f+tri_F.rows(), 3);
    F.bottomRows(tri_F.rows()) = tri_F;
    F.bottomRows(tri_F.rows()).array() += prev_v;
    mclAssert(tri_C.cols()==3, "TODO: opacity in RenderCache");

    if (C.rows()==0)
    {
        C = tri_C;
    }
    else if (colors_per_vertex)
    {
        std::vector<std::vector<int> > v_to_f(tri_V.rows(), std::vector<int>());
        for (int i=0; i<(int)tri_F.rows(); ++i)
        {
            v_to_f[tri_F(i,0)].emplace_back(i);
            v_to_f[tri_F(i,1)].emplace_back(i);
            v_to_f[tri_F(i,2)].emplace_back(i);
        }
        int prev_c = C.rows();
        int c_cols = C.cols();
        C.conservativeResize(prev_c+tri_V.rows(), c_cols);
        for (int i=0; i<(int)tri_V.rows(); ++i)
        {
            Eigen::RowVector3d ci = Eigen::RowVector3d::Zero();
            int nt = v_to_f[i].size();
            for (int j=0; j<nt; ++j)
            {
                ci += tri_C.row(v_to_f[i][j]);
            }
            ci *= 1.0 / std::max(1.0, double(nt));
            for (int j=0; j<3; ++j)
            {
                C(prev_c+i,j) = ci[j];
            }
            if (c_cols == 4) { C(prev_c+i,3) = 1; }
        }
    }
    else
    {   
        int prev_c = C.rows();
        C.conservativeResize(prev_c+tri_C.rows(), C.cols());
        mclAssert(tri_C.cols()==3);
        C.bottomRows(tri_C.rows()).leftCols(3) = tri_C;
    }
}

} // ns mcl
