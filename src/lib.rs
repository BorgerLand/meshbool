use crate::boolean3::Boolean3;
use crate::common::{LossyFrom, Polygons};
use crate::meshboolimpl::{MeshBoolImpl, Relation};
use crate::shared::normal_transform;
use nalgebra::{Matrix3, Matrix3x4, Point3, UnitQuaternion, Vector2, Vector3};
use std::ops::{Add, AddAssign, BitXor, BitXorAssign, Neg, Sub, SubAssign};

pub use crate::common::AABB;
pub use crate::common::OpType;
pub use crate::polygon::triangulate;

mod boolean3;
mod boolean_result;
mod collider;
mod common;
mod constructors;
mod disjoint_sets;
mod edge_op;
mod face_op;
mod mesh_fixes;
mod meshboolimpl;
mod parallel;
mod polygon;
mod properties;
mod shared;
mod smoothing;
mod sort;
mod subdivision;
mod tree2d;
mod tri_dis;
mod utils;
mod vec;

#[test]
fn test() {
	use nalgebra::Vector3;

	//just make sure it don't crash at the sight of the simplest shapes
	let cube1 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), true);
	let cube2 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), false);

	let union = &cube1 + &cube2;
	println!("{:?}", union.get_mesh_gl_32(0));

	let difference = &cube1 - &cube2;
	println!("{:?}", difference.get_mesh_gl_32(0));

	let intersection = &cube1 ^ &cube2;
	println!("{:?}", intersection.get_mesh_gl_32(0));
}

fn halfspace(b_box: AABB, mut normal: Vector3<f64>, origin_offset: f64) -> MeshBool {
	normal.normalize_mut();
	let mut cutter =
		MeshBool::cube(Vector3::repeat(2.0), true).translate(Vector3::new(1.0, 0.0, 0.0));
	let size: f64 = (b_box.center() - normal * origin_offset).norm() + 0.5 * b_box.size().norm();
	cutter = cutter
		.scale(Vector3::repeat(size))
		.translate(Vector3::new(origin_offset, 0.0, 0.0));
	let y_deg: f64 = normal.z.asin().neg().to_degrees();
	let z_deg: f64 = normal.y.atan2(normal.x).to_degrees();
	return cutter.rotate(0.0, y_deg, z_deg);
}

///@brief Mesh input/output suitable for pushing directly into graphics
///libraries.
///
///The core (non-optional) parts of MeshGL are the triVerts indices buffer and
///the vertProperties interleaved vertex buffer, which follow the conventions of
///OpenGL (and other graphic libraries') buffers and are therefore generally
///easy to map directly to other applications' data structures.
///
///The triVerts vector has a stride of 3 and specifies triangles as
///vertex indices. For triVerts = [2, 4, 5, 3, 1, 6, ...], the triangles are [2,
///4, 5], [3, 1, 6], etc. and likewise the halfedges are [2, 4], [4, 5], [5, 2],
///[3, 1], [1, 6], [6, 3], etc.
///
///The triVerts indices should form a manifold mesh: each of the 3 halfedges of
///each triangle should have exactly one paired halfedge in the list, defined as
///having the first index of one equal to the second index of the other and
///vice-versa. However, this is not always possible - consider e.g. a cube with
///normal-vector properties. Shared vertices would turn the cube into a ball by
///interpolating normals - the common solution is to duplicate each corner
///vertex into 3, each with the same position, but different normals
///corresponding to each face. This is exactly what should be done in MeshGL,
///however we request two additional vectors in this case: mergeFromVert and
///mergeToVert. Each vertex mergeFromVert[i] is merged into vertex
///mergeToVert[i], avoiding unreliable floating-point comparisons to recover the
///manifold topology. These merges are simply a union, so which is from and to
///doesn't matter.
///
///If you don't have merge vectors, you can create them with the Merge() method,
///however this will fail if the mesh is not already manifold within the set
///tolerance. For maximum reliability, always store the merge vectors with the
///mesh, e.g. using the EXT_mesh_manifold extension in glTF.
///
///You can have any number of arbitrary floating-point properties per vertex,
///and they will all be interpolated as necessary during operations. It is up to
///you to keep track of which channel represents what type of data. A few of
///Manifold's methods allow you to specify the channel where normals data
///starts, in order to update it automatically for transforms and such. This
///will be easier if your meshes all use the same channels for properties, but
///this is not a requirement. Operations between meshes with different numbers
///of peroperties will simply use the larger numProp and pad the smaller one
///with zeroes.
///
///On output, the triangles are sorted into runs (runIndex, runOriginalID,
///runTransform) that correspond to different mesh inputs. Other 3D libraries
///may refer to these runs as primitives of a mesh (as in glTF) or draw calls,
///as they often represent different materials on different parts of the mesh.
///It is generally a good idea to maintain a map of OriginalIDs to materials to
///make it easy to reapply them after a set of Boolean operations. These runs
///can also be used as input, and thus also ensure a lossless roundtrip of data
///through MeshGL.
///
///As an example, with runIndex = [0, 6, 18, 21] and runOriginalID = [1, 3, 3],
///there are 7 triangles, where the first two are from the input mesh with ID 1,
///the next 4 are from an input mesh with ID 3, and the last triangle is from a
///different copy (instance) of the input mesh with ID 3. These two instances
///can be distinguished by their different runTransform matrices.
///
///You can reconstruct polygonal faces by assembling all the triangles that are
///from the same run and share the same faceID. These faces will be planar
///within the output tolerance.
///
///The halfedgeTangent vector is used to specify the weighted tangent vectors of
///each halfedge for the purpose of using the Refine methods to create a
///smoothly-interpolated surface. They can also be output when calculated
///automatically by the Smooth functions.
///
///MeshGL is an alias for the standard single-precision version. Use MeshGL64 to
///output the full double precision that Manifold uses internally.
#[derive(Debug, Clone)]
pub struct MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize>,
{
	/// Number of properties per vertex, always >= 3.
	pub num_prop: I,
	/// Flat, GL-style interleaved list of all vertex properties: propVal =
	/// vertProperties[vert * numProp + propIdx]. The first three properties are
	/// always the position x, y, z. The stride of the array is numProp.
	pub vert_properties: Vec<F>,
	/// The vertex indices of the three triangle corners in CCW (from the outside)
	/// order, for each triangle.
	pub tri_verts: Vec<I>,
	/// Optional: A list of only the vertex indicies that need to be merged to
	/// reconstruct the manifold.
	pub merge_from_vert: Vec<I>,
	/// Optional: The same length as mergeFromVert, and the corresponding value
	/// contains the vertex to merge with. It will have an identical position, but
	/// the other properties may differ.
	pub merge_to_vert: Vec<I>,
	/// Optional: Indicates runs of triangles that correspond to a particular
	/// input mesh instance. The runs encompass all of triVerts and are sorted
	/// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
	/// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
	/// runIndex will always be 1 longer than runOriginalID, but same length is
	/// also allowed as input: triVerts.size() will be automatically appended in
	/// this case.
	pub run_index: Vec<I>,
	/// Optional: The OriginalID of the mesh this triangle run came from. This ID
	/// is ideal for reapplying materials to the output mesh. Multiple runs may
	/// have the same ID, e.g. representing different copies of the same input
	/// mesh. If you create an input MeshGL that you want to be able to reference
	/// as one or more originals, be sure to set unique values from ReserveIDs().
	pub run_original_id: Vec<u32>,
	/// Optional: For each run, a 3x4 transform is stored representing how the
	/// corresponding original mesh was transformed to create this triangle run.
	/// This matrix is stored in column-major order and the length of the overall
	/// vector is 12 * runOriginalID.size().
	pub run_transform: Vec<F>,
	/// Optional: Length NumTri, contains the source face ID this triangle comes
	/// from. Simplification will maintain all edges between triangles with
	/// different faceIDs. Input faceIDs will be maintained to the outputs, but if
	/// none are given, they will be filled in with Manifold's coplanar face
	/// calculation based on mesh tolerance.
	pub face_id: Vec<I>,
	/// Tolerance for mesh simplification. When creating a Manifold, the tolerance
	/// used will be the maximum of this and a baseline tolerance from the size of
	/// the bounding box. Any edge shorter than tolerance may be collapsed.
	/// Tolerance may be enlarged when floating point error accumulates.
	pub tolerance: F,
}

impl<F, I> Default for MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize>,
{
	fn default() -> Self {
		Self {
			num_prop: I::lossy_from(3),
			tolerance: F::lossy_from(0.0),
			vert_properties: Vec::default(),
			tri_verts: Vec::default(),
			merge_from_vert: Vec::default(),
			merge_to_vert: Vec::default(),
			run_index: Vec::default(),
			run_original_id: Vec::default(),
			run_transform: Vec::default(),
			face_id: Vec::default(),
		}
	}
}

impl<F, I> MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
{
	pub fn num_vert(&self) -> usize {
		self.vert_properties.len() / usize::lossy_from(self.num_prop)
	}
}

impl<F, I> MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize>,
{
	pub fn num_tri(&self) -> usize {
		self.tri_verts.len() / 3
	}
}

pub type MeshGL32 = MeshGLP<f32, u32>;
pub type MeshGL64 = MeshGLP<f64, u64>;

#[derive(Default, Debug, Clone)]
pub struct MeshBool {
	meshbool_impl: MeshBoolImpl,
}

impl MeshBool {
	fn invalid() -> Self {
		let mut meshbool = Self::default();
		meshbool.meshbool_impl.status = ManifoldError::InvalidConstruction;
		meshbool
	}

	///Return a copy of the manifold simplified to the given tolerance, but with its
	///actual tolerance value unchanged. If the tolerance is not given or is less
	///than the current tolerance, the current tolerance is used for simplification.
	///The result will contain a subset of the original verts and all surfaces will
	///have moved by less than tolerance.
	pub fn simplify(&self, tolerance: Option<f64>) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		let old_tolerance = meshbool_impl.tolerance;
		let tolerance = tolerance.unwrap_or(old_tolerance);
		if tolerance > old_tolerance {
			meshbool_impl.tolerance = tolerance;
			meshbool_impl.mark_coplanar();
		}

		meshbool_impl.simplify_topology(0);
		meshbool_impl.finish();
		meshbool_impl.tolerance = tolerance;
		Self::from(meshbool_impl)
	}

	///The genus is a topological property of the manifold, representing the number
	///of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
	///mesh, so it is best to call Decompose() first.
	pub fn genus(&self) -> usize {
		let chi: i32 = self.num_vert() as i32 - self.num_edge() as i32 + self.num_tri() as i32;
		return (1 - chi / 2) as usize;
	}

	///Returns the surface area of the manifold.
	pub fn surface_area(&self) -> f64 {
		self.meshbool_impl
			.get_property(properties::Property::SurfaceArea)
	}

	///Returns the volume of the manifold.
	pub fn volume(&self) -> f64 {
		self.meshbool_impl
			.get_property(properties::Property::Volume)
	}

	///If this mesh is an original, this returns its meshID that can be referenced
	///by product manifolds' MeshRelation. If this manifold is a product, this
	///returns -1.
	pub fn original_id(&self) -> i32 {
		return self.meshbool_impl.mesh_relation.original_id;
	}

	///This removes all relations (originalID, faceID, transform) to ancestor meshes
	///and this new Manifold is marked an original. It also recreates faces
	///- these don't get joined at boundaries where originalID changes, so the
	///reset may allow triangles of flat faces to be further collapsed with
	///Simplify().
	pub fn as_original(&self) -> Self {
		let old_impl = &self.meshbool_impl;
		if old_impl.status != ManifoldError::NoError {
			let mut new_impl = MeshBoolImpl::default();
			new_impl.status = old_impl.status;
			return Self::from(new_impl);
		}

		let mut new_impl = self.meshbool_impl.clone();
		new_impl.initialize_original(false);
		new_impl.mark_coplanar();
		new_impl.initialize_original(true);
		Self::from(new_impl)
	}

	///Returns the first of n sequential new unique mesh IDs for marking sets of
	///triangles that can be looked up after further operations. Assign to
	///MeshGL.runOriginalID vector.
	pub fn reserve_ids(n: u32) -> u32 {
		return MeshBoolImpl::reserve_ids(n as usize) as u32;
	}

	///The triangle normal vectors are saved over the course of operations rather
	///than recalculated to avoid rounding error. This checks that triangles still
	///match their normal vectors within Precision().
	pub fn matches_tri_normals(&self) -> bool {
		self.meshbool_impl.matches_tri_normals()
	}

	///The number of triangles that are colinear within Precision(). This library
	///attempts to remove all of these, but it cannot always remove all of them
	///without changing the mesh by too much.
	pub fn num_degenerate_tris(&self) -> usize {
		self.meshbool_impl.num_degenerate_tris()
	}

	///Move this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to add to every vertex.
	pub fn translate(&self, v: Vector3<f64>) -> Self {
		let mut transform = Matrix3x4::<f64>::identity();
		*transform.column_mut(3) = *v;
		Self::from(self.meshbool_impl.transform(&transform))
	}

	///Scale this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to multiply every vertex by per component.
	pub fn scale(&self, v: Vector3<f64>) -> Self {
		let mut transform = Matrix3x4::<f64>::identity();
		for i in 0..3 {
			transform[(i, i)] = v[i];
		}

		Self::from(self.meshbool_impl.transform(&transform))
	}

	///Applies an Euler angle rotation to the manifold, first about the X axis, then
	///Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
	///and eliminate it completely for any multiples of 90 degrees. Additionally,
	///more efficient code paths are used to update the manifold when the transforms
	///only rotate by multiples of 90 degrees. This operation can be chained.
	///Transforms are combined and applied lazily.
	///
	///@param xDegrees First rotation, degrees about the X-axis.
	///@param yDegrees Second rotation, degrees about the Y-axis.
	///@param zDegrees Third rotation, degrees about the Z-axis.
	pub fn rotate(&self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> Self {
		let transform = UnitQuaternion::from_euler_angles(
			x_degrees.to_radians(),
			y_degrees.to_radians(),
			z_degrees.to_radians(),
		)
		.to_homogeneous()
		.fixed_view::<3, 4>(0, 0)
		.into_owned();

		Self::from(self.meshbool_impl.transform(&transform))
	}

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(&self, m: &Matrix3x4<f64>) -> Self {
		Self::from(self.meshbool_impl.transform(&m))
	}

	///Mirror this Manifold over the plane described by the unit form of the given
	///normal vector. If the length of the normal is zero, an empty Manifold is
	///returned. This operation can be chained. Transforms are combined and applied
	///lazily.
	///
	///@param normal The normal vector of the plane to be mirrored over
	pub fn mirror(&self, normal: Vector3<f64>) -> Self {
		if normal.norm() == 0.0 {
			return Self::default();
		}
		let n = normal.normalize();
		let m = Matrix3::identity() - (2.0 * (n * n.transpose()));
		let m = Matrix3x4::from_columns(&[
			m.column(0).into(),
			m.column(1).into(),
			m.column(2).into(),
			Vector3::default(),
		]);
		Self::from(self.meshbool_impl.transform(&m))
	}

	///This function does not change the topology, but allows the vertices to be
	///moved according to any arbitrary input function. It is easy to create a
	///function that warps a geometrically valid object into one which overlaps, but
	///that is not checked here, so it is up to the user to choose their function
	///with discretion.
	///
	///@param warpFunc A function that modifies a given vertex position.
	pub fn warp(&self, warp_func: impl Fn(&mut Point3<f64>)) -> Self {
		let old_impl = &self.meshbool_impl;
		if old_impl.status != ManifoldError::NoError {
			let mut meshbool_impl = MeshBoolImpl::default();
			meshbool_impl.status = old_impl.status;
			return Self::from(meshbool_impl);
		}
		let mut meshbool_impl = old_impl.clone();
		meshbool_impl.warp(warp_func);
		Self::from(meshbool_impl)
	}

	///Same as Manifold::Warp but calls warpFunc with with
	///a VecView which is roughly equivalent to std::span
	///pointing to all vec3 elements to be modified in-place
	///
	///@param warpFunc A function that modifies multiple vertex positions.
	pub fn warp_batch(&self, warp_func: impl Fn(&mut [Point3<f64>])) -> Self {
		let old_impl = &self.meshbool_impl;
		if old_impl.status != ManifoldError::NoError {
			let mut meshbool_impl = MeshBoolImpl::default();
			meshbool_impl.status = old_impl.status;
			return Self::from(meshbool_impl);
		}
		let mut meshbool_impl = old_impl.clone();
		meshbool_impl.warp_batch(warp_func);
		Self::from(meshbool_impl)
	}

	///Create a new copy of this manifold with updated vertex properties by
	///supplying a function that takes the existing position and properties as
	///input. You may specify any number of output properties, allowing creation and
	///removal of channels. Note: undefined behavior will result if you read past
	///the number of input properties or write past the number of output properties.
	///
	///If prop_func is a None, this function will just set the channel to zeroes.
	///
	///@param num_prop The new number of properties per vertex.
	///@param prop_func A function that modifies the properties of a given vertex.
	pub fn set_properties(
		&self,
		num_prop: i32,
		prop_func: Option<impl Fn(&mut [f64], Point3<f64>, &[f64])>,
	) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		let old_num_prop = self.num_prop();
		let old_properties = meshbool_impl.properties.clone();

		if num_prop == 0 {
			meshbool_impl.properties.clear();
		} else {
			meshbool_impl.properties = vec![0.0; num_prop as usize * self.num_prop_vert()];

			if let Some(prop_func) = prop_func {
				for tri in 0..self.num_tri() {
					for i in 0..3 {
						let edge = &meshbool_impl.halfedge[3 * tri + i];
						let vert = edge.start_vert;
						let prop_vert = edge.prop_vert;
						prop_func(
							&mut meshbool_impl.properties[(num_prop * prop_vert) as usize
								..(num_prop * (prop_vert + 1)) as usize],
							meshbool_impl.vert_pos[vert as usize],
							&old_properties[(old_num_prop * prop_vert as usize) as usize
								..(old_num_prop * (prop_vert as usize + 1)) as usize],
						);
					}
				}
			}
		}

		meshbool_impl.num_prop = num_prop;
		return Self::from(meshbool_impl);
	}

	///Curvature is the inverse of the radius of curvature, and signed such that
	///positive is convex and negative is concave. There are two orthogonal
	///principal curvatures at any point on a manifold, with one maximum and the
	///other minimum. Gaussian curvature is their product, while mean
	///curvature is their sum. This approximates them for every vertex and assigns
	///them as vertex properties on the given channels.
	///
	///@param gaussian_idx The property channel index in which to store the Gaussian
	///curvature. An index < 0 will be ignored (stores nothing). The property set
	///will be automatically expanded to include the channel index specified.
	///
	///@param mean_idx The property channel index in which to store the mean
	///curvature. An index < 0 will be ignored (stores nothing). The property set
	///will be automatically expanded to include the channel index specified.
	pub fn calculate_curvature(&self, gaussian_idx: i32, mean_idx: i32) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		meshbool_impl.calculate_curvature(gaussian_idx, mean_idx);
		Self::from(meshbool_impl)
	}

	///Fills in vertex properties for normal vectors, calculated from the mesh
	///geometry. Flat faces composed of three or more triangles will remain flat.
	///
	///@param normalIdx The property channel in which to store the X
	///values of the normals. The X, Y, and Z channels will be sequential. The
	///property set will be automatically expanded such that NumProp will be at
	///least normalIdx + 3.
	///
	///@param minSharpAngle Any edges with angles greater than this value will
	///remain sharp, getting different normal vector properties on each side of the
	///edge. By default, no edges are sharp and all normals are shared. With a value
	///of zero, the model is faceted and all normals match their triangle normals,
	///but in this case it would be better not to calculate normals at all.
	pub fn calculate_normals(&self, normal_idx: i32, min_sharp_angle: f64) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		meshbool_impl.set_normals(normal_idx, min_sharp_angle);
		return Self::from(meshbool_impl);
	}

	///Increase the density of the mesh by splitting every edge into n pieces. For
	///instance, with n = 2, each triangle will be split into 4 triangles. Quads
	///will ignore their interior triangle bisector. These will all be coplanar (and
	///will not be immediately collapsed) unless the Mesh/Manifold has
	///halfedgeTangents specified (e.g. from the Smooth() constructor), in which
	///case the new vertices will be moved to the interpolated surface according to
	///their barycentric coordinates.
	///
	///@param n The number of pieces to split every edge into. Must be > 1.
	pub fn refine(&self, n: i32) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		if n > 1 {
			meshbool_impl.refine(|_, _, _| n - 1, false);
		}
		Self::from(meshbool_impl)
	}

	///Increase the density of the mesh by splitting each edge into pieces of
	///roughly the input length. Interior verts are added to keep the rest of the
	///triangulation edges also of roughly the same length. If halfedgeTangents are
	///present (e.g. from the Smooth() constructor), the new vertices will be moved
	///to the interpolated surface according to their barycentric coordinates. Quads
	///will ignore their interior triangle bisector.
	///
	///@param length The length that edges will be broken down to.
	pub fn refine_to_length(&self, mut length: f64) -> Self {
		length = length.abs();
		let mut meshbool_impl = self.meshbool_impl.clone();
		meshbool_impl.refine(|edge, _, _| (edge.norm() / length) as i32, false);
		Self::from(meshbool_impl)
	}

	///Increase the density of the mesh by splitting each edge into pieces such that
	///any point on the resulting triangles is roughly within tolerance of the
	///smoothly curved surface defined by the tangent vectors. This means tightly
	///curving regions will be divided more finely than smoother regions. If
	///halfedgeTangents are not present, the result will simply be a copy of the
	///original. Quads will ignore their interior triangle bisector.
	///
	///@param tolerance The desired maximum distance between the faceted mesh
	///produced and the exact smoothly curving surface. All vertices are exactly on
	///the surface, within rounding error.
	pub fn refine_to_tolerance(&self, mut tolerance: f64) -> Self {
		tolerance = tolerance.abs();
		let mut meshbool_impl = self.meshbool_impl.clone();
		// if !pImpl.halfedge_tangent.empty() {
		if false {
			meshbool_impl.refine(
				|edge, tangent_start, tangent_end| {
					let edge_norm: Vector3<f64> = edge.normalize();
					// Weight heuristic
					let t_start: Vector3<f64> = tangent_start.xyz();
					let t_end: Vector3<f64> = tangent_end.xyz();
					// Perpendicular to edge
					let start: Vector3<f64> = t_start - edge_norm * edge_norm.dot(&t_start);
					let end: Vector3<f64> = t_end - edge_norm * edge_norm.dot(&t_end);
					// Circular arc result plus heuristic term for non-circular curves
					let d: f64 = 0.5 * (start.norm() + end.norm()) + (start - end).norm();
					(3.0 * d / (4.0 * tolerance)).sqrt() as i32
				},
				true,
			);
		}
		Self::from(meshbool_impl)
	}

	///	The central operation of this library: the Boolean combines two manifolds
	///	into another by calculating their intersections and removing the unused
	///	portions.
	///	[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid)
	///	inputs will produce &epsilon;-valid output. &epsilon;-invalid input may fail
	///	triangulation.
	///
	///	These operations are optimized to produce nearly-instant results if either
	///	input is empty or their bounding boxes do not overlap.
	///
	///	@param second The other Manifold.
	///	@param op The type of operation to perform.
	pub fn boolean(&self, other: &Self, op: OpType) -> Self {
		Self::from(Boolean3::new(&self.meshbool_impl, &other.meshbool_impl, op).result(op))
	}

	///Split cuts this manifold in two using the cutter manifold. The first result
	///is the intersection, second is the difference. This is more efficient than
	///doing them separately.
	///
	///@param cutter
	pub fn split(&self, cutter: &Self) -> (Self, Self) {
		let impl1 = &self.meshbool_impl;
		let impl2 = &cutter.meshbool_impl;

		let boolean = Boolean3::new(impl1, impl2, OpType::Subtract);
		let result1 = boolean.result(OpType::Intersect);
		let result2 = boolean.result(OpType::Subtract);
		(Self::from(result1), Self::from(result2))
	}

	///Convenient version of Split() for a half-space.
	///
	///@param normal This vector is normal to the cutting plane and its length does
	///not matter. The first result is in the direction of this vector, the second
	///result is on the opposite side.
	///@param originOffset The distance of the plane from the origin in the
	///direction of the normal vector.
	pub fn split_by_plane(&self, normal: Vector3<f64>, origin_offset: f64) -> (Self, Self) {
		self.split(&halfspace(self.bounding_box(), normal, origin_offset))
	}

	///Identical to SplitByPlane(), but calculating and returning only the first
	///result.
	///
	///@param normal This vector is normal to the cutting plane and its length does
	///not matter. The result is in the direction of this vector from the plane.
	///@param originOffset The distance of the plane from the origin in the
	///direction of the normal vector.
	pub fn trim_by_plane(&self, normal: Vector3<f64>, origin_offset: f64) -> Self {
		self ^ &halfspace(self.bounding_box(), normal, origin_offset)
	}

	///Returns the cross section of this object parallel to the X-Y plane at the
	///specified Z height, defaulting to zero. Using a height equal to the bottom of
	///the bounding box will return the bottom faces, while using a height equal to
	///the top of the bounding box will return empty.
	pub fn slice(&self, height: f64) -> Polygons {
		self.meshbool_impl.slice(height)
	}

	fn get_mesh_gl_impl<F, I>(meshbool_impl: &MeshBoolImpl, normal_idx: i32) -> MeshGLP<F, I>
	where
		F: LossyFrom<f64> + Copy,
		f64: From<F>,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
	{
		let num_prop = meshbool_impl.num_prop();
		let num_vert = meshbool_impl.num_prop_vert();
		let num_tri = meshbool_impl.num_tri();

		let is_original = meshbool_impl.mesh_relation.original_id >= 0;
		let update_normals = !is_original && normal_idx >= 0;

		let out_num_prop = 3 + num_prop;
		let tolerance = meshbool_impl
			.tolerance
			.max((f32::EPSILON as f64) * meshbool_impl.bbox.scale());

		let mut tri_verts: Vec<I> = vec![I::lossy_from(0); 3 * num_tri];

		// Sort the triangles into runs
		let mut face_id: Vec<I> = vec![I::lossy_from(0); num_tri];
		let mut tri_new2old: Vec<_> = (0..num_tri).map(|i| i as i32).collect();
		let tri_ref = &meshbool_impl.mesh_relation.tri_ref;
		// Don't sort originals - keep them in order
		if !is_original {
			tri_new2old
				.sort_by_key(|&i| (tri_ref[i as usize].original_id, tri_ref[i as usize].mesh_id));
		}

		let mut run_index: Vec<I> = Vec::new();
		let mut run_original_id: Vec<u32> = Vec::new();
		let mut run_transform: Vec<F> = Vec::new();

		let mut run_normal_transform: Vec<Matrix3<f64>> = Vec::new();
		let mut add_run = |tri, rel: Relation| {
			run_index.push(I::lossy_from(3 * tri));
			run_original_id.push(rel.original_id as u32);
			if update_normals {
				run_normal_transform.push(
					normal_transform(&rel.transform) * (if rel.back_side { -1.0 } else { 1.0 }),
				);
			}

			if !is_original {
				for col in 0..4 {
					for row in 0..3 {
						run_transform.push(F::lossy_from(rel.transform[(row, col)]))
					}
				}
			}
		};

		let mut mesh_id_transform = meshbool_impl.mesh_relation.mesh_id_transform.clone();
		let mut last_id = -1;
		for tri in 0..num_tri {
			let old_tri = tri_new2old[tri] as usize;
			let tri_ref = tri_ref[old_tri];
			let mesh_id = tri_ref.mesh_id;

			face_id[tri] = I::lossy_from(if tri_ref.face_id >= 0 {
				tri_ref.face_id as usize
			} else {
				tri_ref.coplanar_id as usize
			});
			for i in 0..3 {
				tri_verts[3 * tri + i] =
					I::lossy_from(meshbool_impl.halfedge[3 * old_tri + i].start_vert as usize);
			}

			if mesh_id != last_id {
				let it = mesh_id_transform.remove(&mesh_id);
				let rel = it.unwrap_or_default();
				add_run(tri, rel);
				last_id = mesh_id;
			}
		}

		// Add runs for originals that did not contribute any faces to the output
		for pair in mesh_id_transform {
			add_run(num_tri, pair.1);
		}

		run_index.push(I::lossy_from(3 * num_tri));

		// Early return for no props
		if num_prop == 0 {
			let mut vert_properties: Vec<F> = vec![F::lossy_from(0.0); 3 * num_vert];
			for i in 0..num_vert {
				let v = meshbool_impl.vert_pos[i];
				vert_properties[3 * i] = F::lossy_from(v.x);
				vert_properties[3 * i + 1] = F::lossy_from(v.y);
				vert_properties[3 * i + 2] = F::lossy_from(v.z);
			}

			return MeshGLP {
				num_prop: I::lossy_from(out_num_prop),
				vert_properties,
				tri_verts,
				merge_from_vert: Vec::default(),
				merge_to_vert: Vec::default(),
				run_index,
				run_original_id,
				run_transform,
				face_id,
				tolerance: F::lossy_from(tolerance),
			};
		}

		// Duplicate verts with different props
		let mut vert2idx: Vec<i32> = vec![-1; meshbool_impl.num_vert()];
		let mut vert_prop_pair: Vec<Vec<Vector2<i32>>> = vec![Vec::new(); meshbool_impl.num_vert()];
		let mut vert_properties: Vec<F> = Vec::with_capacity(num_vert * out_num_prop);

		let mut merge_from_vert: Vec<I> = Vec::new();
		let mut merge_to_vert: Vec<I> = Vec::new();

		for run in 0..run_original_id.len() {
			for tri in
				(usize::lossy_from(run_index[run]) / 3)..(usize::lossy_from(run_index[run + 1]) / 3)
			{
				for i in 0..3 {
					let prop =
						meshbool_impl.halfedge[3 * (tri_new2old[tri] as usize) + i].prop_vert;
					let vert = usize::lossy_from(tri_verts[3 * tri + i]);

					let bin = &mut vert_prop_pair[vert];
					let mut b_found = false;
					for b in bin.iter() {
						if b.x == prop {
							b_found = true;
							tri_verts[3 * tri + i] = I::lossy_from(b.y as usize);
							break;
						}
					}

					if b_found {
						continue;
					}
					let idx = vert_properties.len() / out_num_prop;
					tri_verts[3 * tri + i] = I::lossy_from(idx);
					bin.push(Vector2::new(prop, idx as i32));

					for p in 0..3 {
						vert_properties.push(F::lossy_from(meshbool_impl.vert_pos[vert][p]));
					}
					for p in 0..num_prop {
						vert_properties.push(F::lossy_from(
							meshbool_impl.properties[(prop as usize) * num_prop + p],
						));
					}

					if update_normals {
						let mut normal = Vector3::<f64>::default();
						let start = vert_properties.len() - out_num_prop;
						for i in 0..3 {
							normal[i] = f64::from(
								vert_properties[((start + 3 + i) as i32 + normal_idx) as usize],
							);
						}

						normal = (run_normal_transform[run] * normal).normalize();
						for i in 0..3 {
							vert_properties[((start + 3 + i) as i32 + normal_idx) as usize] =
								F::lossy_from(normal[i]);
						}
					}

					if vert2idx[vert] == -1 {
						vert2idx[vert] = idx as i32;
					} else {
						merge_from_vert.push(I::lossy_from(idx));
						merge_to_vert.push(I::lossy_from(vert2idx[vert] as usize));
					}
				}
			}
		}

		MeshGLP {
			num_prop: I::lossy_from(out_num_prop),
			vert_properties,
			tri_verts,
			merge_from_vert,
			merge_to_vert,
			run_index,
			run_original_id,
			run_transform,
			face_id,
			tolerance: F::lossy_from(tolerance),
		}
	}

	///The most complete output of this library, returning a MeshGL that is designed
	///to easily push into a renderer, including all interleaved vertex properties
	///that may have been input. It also includes relations to all the input meshes
	///that form a part of this result and the transforms applied to each.
	///
	///@param normalIdx If the original MeshGL inputs that formed this manifold had
	///properties corresponding to normal vectors, you can specify the first of the
	///three consecutive property channels forming the (x, y, z) normals, which will
	///cause this output MeshGL to automatically update these normals according to
	///the applied transforms and front/back side. normalIdx + 3 must be <=
	///numProp, and all original MeshGLs must use the same channels for their
	///normals.
	pub fn get_mesh_gl_32(&self, normal_idx: i32) -> MeshGL32 {
		Self::get_mesh_gl_impl(&self.meshbool_impl, normal_idx)
	}

	///The most complete output of this library, returning a MeshGL that is designed
	///to easily push into a renderer, including all interleaved vertex properties
	///that may have been input. It also includes relations to all the input meshes
	///that form a part of this result and the transforms applied to each.
	///
	///@param normalIdx If the original MeshGL inputs that formed this manifold had
	///properties corresponding to normal vectors, you can specify the first of the
	///three consecutive property channels forming the (x, y, z) normals, which will
	///cause this output MeshGL to automatically update these normals according to
	///the applied transforms and front/back side. normalIdx + 3 must be <=
	///numProp, and all original MeshGLs must use the same channels for their
	///normals.
	pub fn get_mesh_gl_64(&self, normal_idx: i32) -> MeshGL64 {
		Self::get_mesh_gl_impl(&self.meshbool_impl, normal_idx)
	}

	pub fn from_meshgl<F, I>(mesh_gl: &MeshGLP<F, I>) -> Self
	where
		F: LossyFrom<f64> + Copy,
		f64: From<F>,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
	{
		Self::from(MeshBoolImpl::from_meshgl(mesh_gl))
	}

	///Does the Manifold have any triangles?
	pub fn is_empty(&self) -> bool {
		self.meshbool_impl.is_empty()
	}

	///Returns the reason for an input Mesh producing an empty Manifold. This Status
	///will carry on through operations like NaN propogation, ensuring an errored
	///mesh doesn't get mysteriously lost. Empty meshes may still show
	///NoError, for instance the intersection of non-overlapping meshes.
	pub fn status(&self) -> ManifoldError {
		self.meshbool_impl.status
	}

	///The number of vertices in the Manifold.
	pub fn num_vert(&self) -> usize {
		self.meshbool_impl.num_vert()
	}

	///The number of edges in the Manifold.
	pub fn num_edge(&self) -> usize {
		self.meshbool_impl.num_edge()
	}

	///The number of triangles in the Manifold.
	pub fn num_tri(&self) -> usize {
		self.meshbool_impl.num_tri()
	}

	///The number of properties per vertex in the Manifold.
	pub fn num_prop(&self) -> usize {
		self.meshbool_impl.num_prop()
	}

	///The number of property vertices in the Manifold. This will always be >=
	///NumVert, as some physical vertices may be duplicated to account for different
	///properties on different neighboring triangles.
	pub fn num_prop_vert(&self) -> usize {
		self.meshbool_impl.num_prop_vert()
	}

	///Returns the axis-aligned bounding box of all the Manifold's vertices.
	pub fn bounding_box(&self) -> AABB {
		self.meshbool_impl.bbox
	}

	///Returns the epsilon value of this Manifold's vertices, which tracks the
	///approximate rounding error over all the transforms and operations that have
	///led to this state. This is the value of &epsilon; defining
	///[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
	pub fn get_epsilon(&self) -> f64 {
		self.meshbool_impl.epsilon
	}

	///Returns the tolerance value of this Manifold. Triangles that are coplanar
	///within tolerance tend to be merged and edges shorter than tolerance tend to
	///be collapsed.
	pub fn get_tolerance(&self) -> f64 {
		self.meshbool_impl.tolerance
	}

	///Return a copy of the manifold with the set tolerance value.
	///This performs mesh simplification when the tolerance value is increased.
	pub fn set_tolerance(&self, tolerance: f64) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		if tolerance > meshbool_impl.tolerance {
			meshbool_impl.tolerance = tolerance;
			meshbool_impl.mark_coplanar();
			meshbool_impl.simplify_topology(0);
			meshbool_impl.finish();
		} else {
			// for reducing tolerance, we need to make sure it is still at least
			// equal to epsilon.
			meshbool_impl.tolerance = meshbool_impl.epsilon.max(tolerance);
		}

		MeshBool { meshbool_impl }
	}

	///Returns polygons representing the projected outline of this object
	///onto the X-Y plane. These polygons will often self-intersect, so it is
	///recommended to run them through the positive fill rule of CrossSection to get
	///a sensible result before using them.
	pub fn project(&self) -> Polygons {
		self.meshbool_impl.project()
	}

	pub fn min_gap(&self, other: &Self, search_length: f64) -> f64 {
		let intersect = self ^ other;
		if !intersect.is_empty() {
			return 0.0;
		}

		self.meshbool_impl
			.min_gap(&other.meshbool_impl, search_length)
	}
}

impl From<MeshBoolImpl> for MeshBool {
	fn from(value: MeshBoolImpl) -> Self {
		Self {
			meshbool_impl: value,
		}
	}
}

impl Add for &MeshBool {
	type Output = MeshBool;
	fn add(self, rhs: Self) -> Self::Output {
		self.boolean(rhs, OpType::Add)
	}
}

impl AddAssign<&Self> for MeshBool {
	fn add_assign(&mut self, rhs: &Self) {
		*self = self.boolean(rhs, OpType::Add);
	}
}

impl Sub for &MeshBool {
	type Output = MeshBool;
	fn sub(self, rhs: Self) -> Self::Output {
		self.boolean(rhs, OpType::Subtract)
	}
}

impl SubAssign<&Self> for MeshBool {
	fn sub_assign(&mut self, rhs: &Self) {
		*self = self.boolean(rhs, OpType::Subtract);
	}
}

impl BitXor for &MeshBool {
	type Output = MeshBool;
	fn bitxor(self, rhs: Self) -> Self::Output {
		self.boolean(rhs, OpType::Intersect)
	}
}

impl BitXorAssign<&Self> for MeshBool {
	fn bitxor_assign(&mut self, rhs: &Self) {
		*self = self.boolean(rhs, OpType::Intersect);
	}
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ManifoldError {
	NoError,
	NonFiniteVertex,
	InvalidConstruction,
	ResultTooLarge,
	NotManifold,
	MissingPositionProperties,
	MergeVectorsDifferentLengths,
	TransformWrongLength,
	RunIndexWrongLength,
	FaceIDWrongLength,
	MergeIndexOutOfBounds,
	VertexOutOfBounds,
}
