// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_MESHDATA_HPP
#define MCL_MESHDATA_HPP 1

#include "Singleton.hpp"
#include <Eigen/Geometry>
#include <vector>
#include <unordered_map>

namespace mcl
{

//
// Stores all of the input mesh information.
// All const accessors are thread safe.
// Any non-const functions are not guaranteed to be.
//
class MeshData : public Singleton<MeshData>
{
public:
	typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXi;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXd;

	MeshData() { clear(); }

	// Prints details about mesh
	std::string print_stats() const;

	// Resets all mesh data
	void clear();

	int dim() const { return mDim; }
	int num_meshes() const { return std::max(0, (int)m_V_offsets.size()-1); }

	// Computes mesh quantities, connectivities, etc...
	void initialize();
	bool needs_initialize() const { return needs_init; }

	// General queries
	bool is_texture_param() const;
	bool is_dynamic_vert(int idx) const;
	int mesh_idx_from_element(int elem_idx) const;

	// Get mesh features
	const RowMatrixXi& get_faces() const; // 2D = same as elements, 3D = same as facets
	const RowMatrixXi& get_boundary_edges() const; // same as facets if 2D
	const Eigen::VectorXi& get_boundary_verts() const;
	const RowMatrixXi& get_elements() const { return mP; } // 2D = triangles, 3D = tets
	const RowMatrixXi& get_facets() const { return mF; } // boundary, 2D = edges, 3D = triangles
	Eigen::Vector2i get_adjacent_faces(int e0, int e1) const; // returns faces of edge (-1 if not found)

	// Get mesh quantities
	const Eigen::VectorXd& get_mesh_measures() const; // per-mesh total area or volume
	const Eigen::VectorXd& get_measures() const; // per-element 2D=area, 3D=volume
	void update_measure(int idx, double m); // change per-element measure
	const Eigen::VectorXd& get_scalings() const; // per-element 2D=inv perim, 3D=inv surface area
	const std::vector<std::vector<int> >& get_graph() const; // per-vertex non-directed connectivity of elems

	// Initializer is the starting point of the optimization.
	// If not set before optimization starts, rest shape is used.
	const RowMatrixXd& get_initializer() const;
	void set_initializer(const RowMatrixXd& V) { mVinit = V; }

	// Rest shape defines the energy of the mesh.
	// By default the rest shape is whatever is passed to add_mesh
	const RowMatrixXd& get_rest() const { return mV; }
	void set_rest(const RowMatrixXd& V);

	// Returns offsets [start, end] for elements of the stacked matrix. E.g.,
	// mesh0 = E.block(offsets[0], 0, offsets[1], dim+1)
	// mesh1 = E.block(offsets[1], 0, offsets[2], dim+1)
	// etc...
	const Eigen::VectorXi& get_element_offsets() const { return m_P_offsets; }
	const Eigen::VectorXi& get_vertex_offsets() const { return m_V_offsets; }
	const Eigen::VectorXi& get_facet_offsets() const { return m_F_offsets; }
	const Eigen::VectorXi& get_face_offsets() const;

	// Handy hash map to get the feature indices. Returns -1 if not found.
	// Winding order does not matter.
	int get_face_index(int f0, int f1, int f2) const;
	int get_tet_index(int t0, int t1, int t2, int t3) const;

	// Returns index of the mesh, or -1 if there was an error.
	// If init=true, initialize() is called immediately which can be slow for many large meshes.
	int add_mesh(const RowMatrixXd &V, const RowMatrixXi &P, bool init=true); // auto computes faces
	int add_mesh(const RowMatrixXd &V, const RowMatrixXi &P, const RowMatrixXi &F, bool init=true);

	// Special case: adds a mesh to be parameterized.
	// In this case, the 2D texture coords are the dynamic variable (initializer) and
	// the 3D mesh is the rest mesh (incl. rest_volume=area, rest_surf_area=perimeter, etc...).
	// The input mesh must be a 3D triangular mesh.
	// If UVinit is not given, one is computed (Tutte).
	// TODO: Some validation and error checking on compatibility.
	int add_parameterized_mesh(const RowMatrixXd &V, const RowMatrixXi &F,
		const RowMatrixXd &UVinit = RowMatrixXd(), bool init=true);
	
 	// Computes (volume-weighted) masses with unit-volume density.
	// If negative, defaults are used: 1100 for volumetric, 0.4 for cloth/2D
	// See: https://www.engineeringtoolbox.com/density-solids-d_1265.html
	// Masses are not needed for static solves.
	// If masses are not computed/set, an empty vector is returned
	void compute_masses(double density_kg = -1);
	void set_masses(const Eigen::VectorXd &m) { m_masses=m; }
	const Eigen::VectorXd& get_masses() const { return m_masses; }
	double get_rest_radius() const;

	// Serialize
	template<class Archive> void save(Archive& archive) const;
	template<class Archive> void load(Archive& archive);

protected:

	// Edges if 2D, triangles if 3D
	void compute_facets(const RowMatrixXi &P, RowMatrixXi &F);

	// Boundary edges, vertices, etc.
	void init_boundary_mappings();

	// Updates per-element internal data
	void init_per_element_data();

	// Mappings to edge/face/etc...
	void init_feature_mapping();

	int mDim;
	bool needs_init;

	// Dynamic:
	RowMatrixXd mV; // rest vertices
	RowMatrixXd mVinit; // initializer
	RowMatrixXi mP; // (internal) elements
	RowMatrixXi mF; // facets
	Eigen::VectorXd m_masses; // per-vertex masses
	Eigen::VectorXd m_rest_bmin_mbax; // [bmin, bmax]

	// Stacked (computed) quantities [dynamic, kinematic]
	// Most are computed on initialize
	Eigen::VectorXd m_mesh_measures; // mesh-measure
	Eigen::VectorXi m_V_offsets; // indices into stacked V
	Eigen::VectorXi m_P_offsets; // indices into stacked P
	Eigen::VectorXi m_F_offsets; // indices into stacked F
	Eigen::VectorXd m_measures; // rest volume/area
	Eigen::VectorXd m_scalings; // inv surface areas/perimeter
	RowMatrixXi m_bnd_edges; // boundary edges
	Eigen::VectorXi m_bnd_verts; // boundary vertices

	// Mappings
	std::vector<std::vector<int> > m_V_graph; // vertex-to-vertex connectivity
	std::unordered_map<std::string, int> m_faces_map; // inds to face, [f0,f1,f2] -> f, f0<f1<f2
	std::unordered_map<std::string, int> m_tets_map; // inds to tet, [t0,t1,...] -> t, t0<t1<...
	std::unordered_map<std::string, Eigen::Vector2i> m_E_to_F; // adj faces, [e0,e1] -> [f0,f1]

}; // class MeshData

} // ns mcl

#endif
