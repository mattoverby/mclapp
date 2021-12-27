// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include "MeshData.hpp"
#include "MCL/AssertHandler.hpp"
#include "MCL/Sort.hpp"
#include "MCL/SignedVolume.hpp"

#include <cereal/cereal.hpp>
#include "MCL/EigenCereal.hpp"
#include <cereal/types/vector.hpp>

#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/doublearea.h>

#include <set>
#include <map>
#include <unordered_set>
#include <tbb/parallel_for.h>
#include <sstream>

namespace mcl
{

using namespace Eigen;

std::string MeshData::print_stats() const
{
	AlignedBox<double,3> box;
	for (int i=0; i<mV.rows(); ++i)
	{
		Vector3d vi = Vector3d::Zero();
		for (int j=0; j<std::min(3,(int)mV.cols()); ++j) { vi[j] = mV(i,j); }
		box.extend(vi);
	}

	std::stringstream ss;
	ss << "MeshData" << std::endl;
	ss << "\tmeshes: " << num_meshes() << std::endl;
	ss << "\tvertices: " << get_rest().rows() << std::endl;
	ss << "\telements: " << get_elements().rows() << std::endl;
	ss << "\tbnd edges: " << get_boundary_edges().rows() << std::endl;
	ss << "\tbnd faces: " << get_faces().rows() << std::endl;
	double box_bnds[3] = { box.sizes()[0], box.sizes()[1], box.sizes()[2] };
	ss << "\tbounds: " << box_bnds[0] << ", " << box_bnds[1] << ", " << box_bnds[2] << std::endl;
	return ss.str();
}

void MeshData::clear()
{
	mDim = 0;
	needs_init = true;

	mV = RowMatrixXd();
	mVinit = RowMatrixXd();
	mP = RowMatrixXi();
	mF = RowMatrixXi();
	m_masses = Eigen::VectorXd();
	m_rest_bmin_mbax = Eigen::VectorXd();

	m_mesh_measures = Eigen::VectorXd();
	m_V_offsets = Eigen::VectorXi();
	m_P_offsets = Eigen::VectorXi();
	m_F_offsets = Eigen::VectorXi();
	m_measures = Eigen::VectorXd();
	m_scalings = Eigen::VectorXd();
	m_bnd_edges = RowMatrixXi();
	m_bnd_verts = Eigen::VectorXi();

	m_V_graph.clear();
	m_faces_map.clear();
	m_tets_map.clear();
	m_E_to_F.clear();
}

void MeshData::initialize()
{
	needs_init = false;

	int n_mesh = num_meshes();
	mclAssert(n_mesh > 0, ("No dynamic meshes"));
	init_boundary_mappings();
	init_per_element_data();
	init_feature_mapping();

	// measures only used for texture param, for now
	m_mesh_measures = Eigen::VectorXd::Zero(n_mesh);
	for (int i=0; i<n_mesh; ++i)
	{
		// if param, m_rest_volumes are surface areas
		int p_start = m_P_offsets[i];
		int np = m_P_offsets[i+1] - p_start;
		m_mesh_measures[i] = m_measures.segment(p_start,np).sum();
	}

	// Bounding box of rest verts
	int nv = mV.rows();
	int vc = mV.cols();
	Eigen::AlignedBox<double,3> box;
	for (int i=0; i<nv; ++i)
	{
		Vector3d xi = Vector3d::Zero();
		xi.head(vc) = mV.row(i).transpose();
		box.extend(xi);
	}
	m_rest_bmin_mbax = Eigen::VectorXd::Zero(vc*2);
	m_rest_bmin_mbax.head(vc) = box.min().head(vc);
	m_rest_bmin_mbax.tail(vc) = box.max().head(vc);

	// Needs initializer?
	if (mVinit.rows() != mV.rows()) { mVinit = mV; }
}

bool MeshData::is_texture_param() const
{
	return mDim==2 && mVinit.cols()==2 && mV.cols()==3;
}

bool MeshData::is_dynamic_vert(int idx) const
{
	return (idx >= 0 && idx < mV.rows());
}

int MeshData::mesh_idx_from_element(int face_idx) const
{
	int num_mesh = num_meshes();
	mclAssert(m_P_offsets.rows() == num_mesh+1);
	for (int i=0; i<num_mesh; ++i)
	{
		if (face_idx >= m_P_offsets[i] && face_idx < m_P_offsets[i+1]) { return i; }
	}

	return -1;
}

const MeshData::RowMatrixXi& MeshData::get_faces() const
{
	if (mDim==2) { return mP; }
	else if (mDim==3) { return mF; }
	return mF;
}

const MeshData::RowMatrixXi& MeshData::get_boundary_edges() const
{
	mclAssert(!needs_init, ("must call initialize first"));
	return m_bnd_edges;
}

const Eigen::VectorXi& MeshData::get_boundary_verts() const
{
	mclAssert(!needs_init, ("must call initialize first"));
	return m_bnd_verts;
}


void MeshData::set_rest(const RowMatrixXd& V)
{
	needs_init = true;
	mV = V;
}

Eigen::Vector2i MeshData::get_adjacent_faces(int e0, int e1) const
{
	if (e1 < e0) { std::swap(e0,e1); }
	std::string hash = std::to_string(e0) + ' ' + std::to_string(e1);
	std::unordered_map<std::string, Eigen::Vector2i>::const_iterator it = m_E_to_F.find(hash);	
	mclAssert(it != m_E_to_F.end(), ("edge not found in get_adjacent_faces"));
	return it->second;
}

const Eigen::VectorXd& MeshData::get_mesh_measures() const
{
	mclAssert(!needs_init, ("must call initialize first"));
	return m_mesh_measures;
}

const Eigen::VectorXd& MeshData::get_measures() const
{
	mclAssert(!needs_init, ("must call initialize first"));
	return m_measures;
}

void MeshData::update_measure(int idx, double m)
{
	mclAssert(idx >= 0 && idx < (int)m_measures.rows());
	m_measures[idx] = m;
}

const MeshData::RowMatrixXd& MeshData::get_initializer() const
{
	if (mVinit.rows()==0) { return mV; }
	return mVinit;
}

const Eigen::VectorXd& MeshData::get_scalings() const
{
	mclAssert(!needs_init, ("must call initialize first"));
	return m_scalings;
}

const std::vector<std::vector<int> >& MeshData::get_graph() const
{
	mclAssert(!needs_init, ("must call initialize first"));
	return m_V_graph;
}

const Eigen::VectorXi& MeshData::get_face_offsets() const
{
	mclAssert(mDim == 2 || mDim == 3);
	if (mDim==2) { return m_P_offsets; }
	else if (mDim==3) { return m_F_offsets; }
	return m_V_offsets; // hush compiler...
}

int MeshData::get_face_index(int f0, int f1, int f2) const
{
	mcl::sort3(f0, f1, f2);
	std::string hash = std::to_string(f0) + ' ' + std::to_string(f1) + ' ' + std::to_string(f2);
	std::unordered_map<std::string, int>::const_iterator it = m_faces_map.find(hash);
	if (it == m_faces_map.end()) { return -1; }
	return it->second;
}

int MeshData::get_tet_index(int t0, int t1, int t2, int t3) const
{
	std::vector<int> tet = { t0, t1, t2, t3 };
	std::sort(tet.begin(), tet.end());
	std::string hash =
		std::to_string(tet[0]) + ' ' + std::to_string(tet[1]) + ' ' +
		std::to_string(tet[2]) + ' ' + std::to_string(tet[3]);
	std::unordered_map<std::string, int>::const_iterator it = m_tets_map.find(hash);
	if (it == m_tets_map.end()) { return -1; }
	return it->second;
}

int MeshData::add_mesh(const RowMatrixXd &V, const RowMatrixXi &P_, const RowMatrixXi &F, bool init)
{
	needs_init = true;
	mclAssert(V.rows() > 0 && (V.cols() == 2 || V.cols() == 3), "bad V");
	mclAssert(P_.rows() > 0 && (P_.cols() == 3 || P_.cols() == 4), "bad P");
	if (mDim==0) { mDim = V.cols(); }
	else { mclAssert(V.cols() == mDim, ("bad V dim")); }

	mclAssert(!is_texture_param(), "cannot mix parameterized with regular meshes");
	mclAssert(F.cols() == mDim, ("bad F cols"));
	mclAssert(P_.minCoeff() == 0, "P does not start at zero");

	// Is a cloth mesh, fill last col with -1
	RowMatrixXi P = P_;
	if (P_.cols() == 3 && V.cols()==3)
	{
		P.conservativeResize(P_.rows(), 4);
		P.col(3).array() = -1;
	}

	int nv = V.rows();
	int np = P.rows();
	int p_cols = P.cols();
	int nf = F.rows();
	int f_cols = F.cols();
	int nv_prev = mV.rows();
	int np_prev = mP.rows();
	int nf_prev = mF.rows();

	// First mesh
	if (nv_prev==0)
	{
		mV = V;
		mP = P;
		mF = F;
		m_V_offsets = VectorXi::Zero(2);
		m_V_offsets[1] = nv;
		m_P_offsets = VectorXi::Zero(2);
		m_P_offsets[1] = np;
		m_F_offsets = VectorXi::Zero(2);
		m_F_offsets[1] = nf;
		if (init) { initialize(); }
		return 0;
	}

	// Resize global data and append
	int mesh_idx = num_meshes();
	m_V_offsets.conservativeResize(mesh_idx+2);
	m_V_offsets[mesh_idx+1] = nv_prev + nv;
	m_P_offsets.conservativeResize(mesh_idx+2);
	m_P_offsets[mesh_idx+1] = np_prev + np;
	m_F_offsets.conservativeResize(mesh_idx+2);
	m_F_offsets[mesh_idx+1] = nf_prev + nf;

	mV.conservativeResize(nv_prev + nv, mDim);
	mV.bottomRows(nv) = V;
	mP.conservativeResize(np_prev + np, p_cols);
	mP.bottomRows(np) = P;
	mP.bottomRows(np).array() += nv_prev;
	mF.conservativeResize(nf_prev + nf, f_cols);
	mF.bottomRows(nf) = F;
	mF.bottomRows(nf).array() += nv_prev;

	if (init) { initialize(); }
	return mesh_idx;
}

int MeshData::add_mesh(const RowMatrixXd &V, const RowMatrixXi &P, bool init)
{
	mclAssert(V.rows() > 0 && (V.cols() == 2 || V.cols() == 3), ("bad V"));
	mclAssert(P.rows() > 0 && (P.cols() == 3 || P.cols() == 4), ("bad P"));
	if (mDim==0) { mDim = V.cols(); }
	else { mclAssert(V.cols() == mDim, ("bad dim")); }

	// If V = 3 and P = 3, then it's a cloth
	if (mDim == 3 && P.cols() == 3)
	{
		MatrixXi F = P;
		return add_mesh(V, P, F, init);
	}

	// Otherwise compute facets
	RowMatrixXi F;
	compute_facets(P, F);
	return add_mesh(V, P, F, init);
}

int MeshData::add_parameterized_mesh(const RowMatrixXd &V, const RowMatrixXi &F, const RowMatrixXd &UVinit, bool init)
{
	mclAssert(V.rows()>0 && F.rows()>0);
	mclAssert(V.cols()==3 && F.cols()==3);
	if (num_meshes() > 0)
	{
		printf("**MeshData TODO: Multiple parameterized meshes for packing\n");
		return -1;
	}

	mDim = 2;
	int nv = V.rows();
	int nf = F.rows();
	
	mV = V; // rest = 3D mesh
	mP = F;
	compute_facets(F, mF); // bnd edges

	m_V_offsets = VectorXi::Zero(2);
	m_V_offsets[1] = nv;
	m_P_offsets = VectorXi::Zero(2);
	m_P_offsets[1] = nf;
	m_F_offsets = VectorXi::Zero(2);
	m_F_offsets[1] = mF.rows(); // computed

	// Compute initial texture coords
	if (UVinit.rows() == V.rows())
	{
		mVinit = UVinit;
	}
	else
	{
		// Tutte embedding
		VectorXi bnd;
		igl::boundary_loop(mP, bnd);
		MatrixXd bnd_uv;
		igl::map_vertices_to_circle(mV, bnd, bnd_uv);
		MatrixXd UV;
		UV.resize(V.rows(),3);
		igl::harmonic(mP, bnd, bnd_uv, 1, UV);
		mVinit = UV.block(0,0,UV.rows(),2);

		// Scale initializer based on volume of the 3D mesh
		AlignedBox<double,3> box;
		for (int i=0; i<nv; ++i) { box.extend(mV.row(i).transpose()); }
		mVinit *= box.sizes().maxCoeff() * 0.5;
	}

	if (init) { initialize(); }
	return num_meshes()-1;
}


void MeshData::compute_masses(double density_kgd)
{
	auto Vi = [&](int idx) {
		int c = std::min(3, (int)mV.cols());
		Vector3d vi = Vector3d::Zero();
		for (int i=0; i<c; ++i) {vi[i] = mV(idx,i); }
		return vi;
	};

	bool use_default_density = density_kgd < 0;
	m_masses = VectorXd::Zero(mV.rows());

	int n_meshes = m_V_offsets.size() - 1;
	for (int i=0; i<n_meshes; ++i)
	{
		int p_start = m_P_offsets[i];
		int p_end = m_P_offsets[i + 1];
		int p_cols = mP.cols();

		if (use_default_density)
		{
			if (mDim==2 || p_cols==3) {
				density_kgd = 0.4;
			} else if (mDim == 3 && p_cols == 4) {
				density_kgd = 1100;
			}
		}

		for (int j=p_start; j<p_end; ++j)
		{
			std::vector<Vector3d> p_verts;
			for (int k=0; k<p_cols; ++k)
			{
				int pi = mP(j,k);
				p_verts.emplace_back(Vi(pi));
			}

			if (p_cols == 4)
			{
				Eigen::Matrix<double,3,3> E;
				E.col(0) = p_verts[1] - p_verts[0];
				E.col(1) = p_verts[2] - p_verts[0];
				E.col(2) = p_verts[3] - p_verts[0];
				double vol = std::abs(E.determinant()/6.0);
				double tet_mass = density_kgd * vol;
				m_masses[mP(j,0)] += tet_mass / 4.0;
				m_masses[mP(j,1)] += tet_mass / 4.0;
				m_masses[mP(j,2)] += tet_mass / 4.0;
				m_masses[mP(j,3)] += tet_mass / 4.0;
			}
			else if (p_cols == 3)
			{
				Vector3d e0 = p_verts[1] - p_verts[0];
				Vector3d e1 = p_verts[2] - p_verts[0];
				double area = 0.5 * (e0.cross(e1)).norm();
				double tri_mass = density_kgd * area;
				m_masses[mP(j,0)] += tri_mass / 3.0;
				m_masses[mP(j,1)] += tri_mass / 3.0;
				m_masses[mP(j,2)] += tri_mass / 3.0;
			}
		}
	}
} // end compute masses

double MeshData::get_rest_radius() const
{
	mclAssert(m_rest_bmin_mbax.rows()==4 || m_rest_bmin_mbax.rows()==6);
	Vector3d bmin = Vector3d::Zero();
	Vector3d bmax = Vector3d::Zero();
	int nc = mV.cols();
	bmin.head(nc) = m_rest_bmin_mbax.head(nc);
	bmax.head(nc) = m_rest_bmin_mbax.tail(nc);
	Vector3d c = 0.5 * (bmin + bmax);
	return (bmax - c).norm();
}

void MeshData::compute_facets(const RowMatrixXi &P, RowMatrixXi &F)
{
	if (P.cols() == 4)
	{
		struct FaceKey {
			FaceKey() : f(Vector3i::Zero()), f_sorted(Vector3i::Zero()) {}
			FaceKey(int f0, int f1, int f2)
			{
				f = Vector3i(f0, f1, f2);
				f_sorted = sort(f);
			}
			Vector3i f;
			Vector3i f_sorted;
			static Vector3i sort(const Vector3i& ff)
			{
				Vector3i r = ff;
				std::sort(r.data(), r.data()+3);
				return r;
			}
			bool operator<(const FaceKey& other) const{
				for (int i=0; i<3; ++i) {
					if (f_sorted[i] < other.f_sorted[i]) { return true; }
					if (f_sorted[i] > other.f_sorted[i]) { return false; }
				}
				return false;
			}
		};

		std::set<FaceKey> faces;
		std::set<FaceKey> faces_seen_twice;

		int n_tets = P.rows();
		int total_faces = 0;
		for (int t=0; t<n_tets; ++t)
		{
			total_faces += 4;
			int p0 = P(t,0);
			int p1 = P(t,1);
			int p2 = P(t,2);
			int p3 = P(t,3);
			FaceKey curr_faces[4] = {
				FaceKey( p0, p1, p3 ),
				FaceKey( p0, p2, p1 ),
				FaceKey( p0, p3, p2 ),
				FaceKey( p1, p2, p3 ) };

			for (int f=0; f<4; ++f)
			{
				const FaceKey &curr_face = curr_faces[f];

				// We've already seen the face at least twice.
				if (faces_seen_twice.count(curr_face)>0) { continue; }

				// Check that we don't already have the face
				std::set<FaceKey>::iterator it = faces.find(curr_face);
				if (it != faces.end()) {
					faces.erase(it);
					faces_seen_twice.insert(curr_face);
					continue;
				}

				// Otherwise, add it to faces
				faces.insert(curr_face);

			} // end loop faces

		} // end loop tets

		// Copy to matrix
		int nf = faces.size();
		mclAssert(nf > 0);
		F.resize(nf, 3);
		std::set<FaceKey>::const_iterator fit = faces.begin();
		for( int f_idx=0; fit != faces.end(); ++fit, ++f_idx ){
			F.row(f_idx) = fit->f;
		}
	}
	else if (P.cols() == 3)
	{
		F = MatrixXi();
		std::vector< std::vector<int> > loops;
		igl::boundary_loop(P, loops);
		int n_loop = loops.size();
		for (int i=0; i<n_loop; ++i)
		{
			std::vector<int> &loop = loops[i];
			int n_prev_edges = F.rows();
			int n_curr_edges = loop.size();
			mclAssert(n_curr_edges > 0);
			F.conservativeResize(n_curr_edges + n_prev_edges, 2);
			for (int j=1; j<n_curr_edges; ++j)
			{
				Vector2i e = Vector2i(loop[j-1], loop[j]);
				int e_idx = n_prev_edges + j - 1;
				F.row(e_idx) = e;
			}
			F.row(F.rows()-1) = RowVector2i(loop.back(),loop[0]);
		}
	}

} // end compute facets

void MeshData::init_per_element_data()
{
	int n_ele = mP.rows();
	if (n_ele == 0) { return; }
	if (mDim==2) { mclAssert(mP.cols()==3, "bad elements dim"); }
	else if (mDim==3) { mclAssert(mP.cols()>2 && mP.cols()<5, "bad elements dim"); }

	m_measures = VectorXd::Zero(n_ele);
	m_scalings = VectorXd::Zero(n_ele);
	bool is_param = is_texture_param();

	tbb::parallel_for(0, n_ele, [&](int i)
	{
		if (is_param)
		{
			mclAssert(mP.cols()==3 && mV.cols()==3);
			std::vector<Vector3d> v = {
				mV.row(mP(i,0)),
				mV.row(mP(i,1)),
				mV.row(mP(i,2)) };

			m_measures[i] = triangle_area(v[0],v[1],v[2]);
			m_scalings[i] = 1.0 / (0.5 * (v[0]-v[1]).norm()+(v[1]-v[2]).norm()+(v[2]-v[0]).norm());
		}
		else if (mDim == 2)
		{
			std::vector<Vector2d> v = {
				mV.row(mP(i,0)),
				mV.row(mP(i,1)),
				mV.row(mP(i,2)) };

			m_measures[i] = signed_triangle_area(v[0],v[1],v[2]);
			if (m_measures[i] <= 0) { m_measures[i] = 0; }
			m_scalings[i] = 1.0 / (0.5  * (v[0]-v[1]).norm()+(v[1]-v[2]).norm()+(v[2]-v[0]).norm());
		}
		else if(mDim == 3)
		{
			if (mP(i,3) < 0)
			{
				std::vector<Vector3d> v = {
					mV.row(mP(i,0)),
					mV.row(mP(i,1)),
					mV.row(mP(i,2)) };

				m_measures[i] = triangle_area(v[0],v[1],v[2]);
				m_scalings[i] = 1.0 / (0.5 * (v[0]-v[1]).norm()+(v[1]-v[2]).norm()+(v[2]-v[0]).norm());
				return;
			}

			std::vector<Vector3d> v = {
				mV.row(mP(i,0)),
				mV.row(mP(i,1)),
				mV.row(mP(i,2)),
				mV.row(mP(i,3)) };

			m_measures[i] = signed_tet_volume(v[0],v[1],v[2],v[3]);
			if (m_measures[i] <= 0) { m_measures[i] = 0; }
			m_scalings[i] = 1.0 / (0.5 * tet_surface_area(v[0],v[1],v[2],v[3]));
		}
	});

	// Verify
	mclAssert(m_measures.minCoeff() >= 0, "negative rest volume");
	mclAssert(m_scalings.minCoeff() >= 0, "negative surface area");

} // end compute per-element data

void MeshData::init_boundary_mappings()
{
	// If dim==2, boundary elements = edges
	if (mDim==2)
	{
		m_bnd_edges = mF;
	}
	else if (mDim == 3)
	{
		int nf = mF.rows();
		mclAssert(mF.cols() == 3, "faces.cols != 3");
		std::set<std::pair<int,int> > edge_set;
		for (int fi=0; fi<nf; ++fi)
		{
			std::pair<int,int> e0(mF(fi,0), mF(fi,1));
			std::pair<int,int> e1(mF(fi,1), mF(fi,2));
			std::pair<int,int> e2(mF(fi,2), mF(fi,0));
			if (e0.first < e0.second) { std::swap(e0.first, e0.second); }
			if (e1.first < e1.second) { std::swap(e1.first, e1.second); }
			if (e2.first < e2.second) { std::swap(e2.first, e2.second); }
			edge_set.emplace(e0);
			edge_set.emplace(e1);
			edge_set.emplace(e2);
		}

		int edge_idx = 0;
		m_bnd_edges.resize(edge_set.size(), 2);
		for (const std::pair<int,int>& edge : edge_set)
		{
			m_bnd_edges.row(edge_idx) = RowVector2i(edge.first, edge.second);
			edge_idx++;
		}
	}

	// Compute boundary verts from boundary edges
	{
		int ne = m_bnd_edges.rows();
		std::unordered_set<int> bnd_verts;
		std::vector<int> bnd_verts_vec;

		for (int i=0; i<ne; ++i)
		{
			int i0 = m_bnd_edges(i,0);
			int i1 = m_bnd_edges(i,1);
			if (bnd_verts.emplace(i0).second) { bnd_verts_vec.emplace_back(i0); }
			if (bnd_verts.emplace(i1).second) { bnd_verts_vec.emplace_back(i1); }
		}

		m_bnd_verts = Map<VectorXi>(bnd_verts_vec.data(), bnd_verts_vec.size());
	}
}

void MeshData::init_feature_mapping()
{
	// Elements graph
	{
		int nv = mV.rows();
		m_V_graph.clear();
		m_V_graph.resize(nv, std::vector<int>());

		auto idx_in_vec = [](const std::vector<int> &v, int idx)
		{
			std::vector<int>::const_iterator it = std::find(v.begin(), v.end(), idx);
			return (it != v.end());
		};

		// Create the graph
		int np = mP.rows();
		int pcols = mP.cols();
		for (int i=0; i<np; ++i)
		{
			for (int j=0; j<pcols; ++j)
			{
				int e0 = mP(i,j);
				int e1 = mP(i,(j+1)%pcols);
				if (e0 < 0 || e1 < 0) { continue; }
				if (!idx_in_vec(m_V_graph[e0],e1)) { m_V_graph[e0].emplace_back(e1); }
				if (!idx_in_vec(m_V_graph[e1],e0)) { m_V_graph[e1].emplace_back(e0); }
			}
		}
	}

	// Mapping for faces
	{
		m_faces_map.clear();
		const MatrixXi &faces = get_faces();
		int nf = faces.rows();
		for (int i=0; i<nf; ++i)
		{
			RowVector3i f = faces.row(i);
			mcl::sort3(f[0], f[1], f[2]);
			std::string hash =
				std::to_string(f[0]) + ' ' +
				std::to_string(f[1]) + ' ' +
				std::to_string(f[2]);
			m_faces_map.emplace(hash, i);
		}
	}

	// Mapping for tets
	m_tets_map.clear();
	if (mP.cols()==4)
	{
		int np = mP.rows();
		for (int i=0; i<np; ++i)
		{
			RowVector4i t = mP.row(i);
			std::vector<int> tet = { t[0], t[1], t[2], t[3] };
			std::sort(tet.begin(), tet.end());
			std::string hash =
				std::to_string(tet[0]) + ' ' +
				std::to_string(tet[1]) + ' ' +
				std::to_string(tet[2]) + ' ' +
				std::to_string(tet[3]);
			m_tets_map.emplace(hash, i);
		}
	}

	// Adjacent faces mapping
	{
		m_E_to_F.clear();
		const MatrixXi &faces = get_faces();
		int nf = faces.rows();
		for (int i=0; i<nf; ++i)
		{
			RowVector3i f = faces.row(i);
			for (int j=0; j<3; ++j)
			{
				int e0 = f[j];
				int e1 = f[(j+1)%3];
				if (e1 < e0) { std::swap(e0,e1); }
				std::string hash = std::to_string(e0) + ' ' + std::to_string(e1);
				std::unordered_map<std::string, Eigen::Vector2i>::iterator it = m_E_to_F.find(hash);

				if (it == m_E_to_F.end())
				{
					Vector2i adj_f(i, -1);
					m_E_to_F.emplace(hash, adj_f);
				}
				else
				{
					Vector2i &adj_f = it->second;
					// a) the face is already listed with the edge
					// b) the face is not listed
					// c) the edge has both faces listed (error)
					if (adj_f[0] == i || adj_f[1] == i) { continue; }
					else if (adj_f[0] == -1) { adj_f[0] = i; }
					else if (adj_f[1] == -1) { adj_f[1] = i; }
					else { mclAssert(false, ("Problem with edge to face mapping")); }
				}
			}
		}
	}

} // end init feature mapping

template <class Archive>
void MeshData::save(Archive& archive) const
{
	// Only serialize input data from add_mesh
	// since initialize() is called after reading.
	archive(
		mDim,
		needs_init,
		mV,
		mVinit,
		mP,
		mF,
		m_masses,
		m_rest_bmin_mbax,
		m_V_offsets,
		m_P_offsets,
		m_F_offsets
		);
}

template <class Archive>
void MeshData::load(Archive& archive)
{
	clear();

	// TODO archive everything correctly and avoid the need for init.
	// But for now just archive a subset of necessary data
	// and use the functions
	archive(
		mDim,
		needs_init,
		mV,
		mVinit,
		mP,
		mF,
		m_masses,
		m_rest_bmin_mbax,
		m_V_offsets,
		m_P_offsets,
		m_F_offsets
		);

	initialize();
}

template void MeshData::save<cereal::BinaryOutputArchive>(cereal::BinaryOutputArchive&) const;
template void MeshData::load<cereal::BinaryInputArchive>(cereal::BinaryInputArchive&);

} // end ns gini