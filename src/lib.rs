use crate::boolean3::Boolean3;
use crate::common::{LossyFrom, Polygons, cosd, sind};
use crate::mesh::MeshGLP;
use crate::meshboolimpl::MeshBoolImpl;
use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3};
use std::ops::{Add, AddAssign, BitXor, BitXorAssign, Sub, SubAssign};

pub use crate::common::{AABB, MeshGL32, MeshGL64, OpType};
pub use crate::polygon::triangulate;

mod boolean3;
mod boolean_result;
mod collider;
mod common;
mod constructors;
mod disjoint_sets;
mod edge_op;
mod face_op;
mod mesh;
mod mesh_fixes;
mod meshboolimpl;
mod multiset;
mod parallel;
mod polygon;
mod polygon_internal;
mod properties;
mod shared;
mod smoothing;
mod sort;
mod subdivision;
mod tree2d;
mod tri_dis;
mod utils;
mod vec;

#[cfg(feature = "test")]
mod test;

fn halfspace(b_box: AABB, mut normal: Vector3<f64>, origin_offset: f64) -> MeshBool {
	normal.normalize_mut();
	let mut cutter =
		MeshBool::cube(Vector3::repeat(2.0), true).translate(Vector3::new(1.0, 0.0, 0.0));
	let size: f64 = (b_box.center() - normal * origin_offset).norm() + 0.5 * b_box.size().norm();
	cutter = cutter
		.scale(Vector3::repeat(size))
		.translate(Vector3::new(origin_offset, 0.0, 0.0));
	let y_deg: f64 = (-libm::asin(normal.z)).to_degrees();
	let z_deg: f64 = libm::atan2(normal.y, normal.x).to_degrees();
	return cutter.rotate(0.0, y_deg, z_deg);
}

#[derive(Default, Debug, Clone)]
pub struct MeshBool {
	meshbool_impl: MeshBoolImpl,
}

impl MeshBool {
	fn invalid() -> Self {
		Self::propagate_status(MeshBoolError::InvalidConstruction)
	}

	fn propagate_status(status: MeshBoolError) -> Self {
		let mut meshbool = Self::default();
		meshbool.meshbool_impl.status = status;
		meshbool
	}

	///Returns a MeshGL that is designed
	///to easily push into a renderer, including all interleaved vertex properties
	///that may have been input. It also includes relations to all the input meshes
	///that form a part of this result and the transforms applied to each.
	///
	///@param normalIdx If this manifold has properties corresponding to normal
	///vectors, you can specify the first of the three consecutive property channels
	///forming the (x, y, z) normals, which will cause this output MeshGL to
	///automatically update these normals according to the applied transforms and
	///front/back side. normalIdx + 3 must be <= numProp, and all original meshes
	///must use the same channels for their normals. Default is -1: if
	///`CalculateNormals()` recorded normals at the standard slot, that slot is
	///used automatically; otherwise no normals are applied. If normals are
	///selected, the runTransform matrices will be removed from the output, to
	///avoid them being double-applied when round-tripped.
	///Passing a non-negative `normalIdx` is the legacy interface from before
	///`CalculateNormals` recorded the slot on the Manifold itself; prefer the
	///no-arg form after `CalculateNormals(0)`. The explicit-idx path will be
	///removed in a future release.
	pub fn get_mesh_gl_32(&self, mut normal_idx: i32) -> MeshGL32 {
		if normal_idx < 0 && self.meshbool_impl.all_have_normals() {
			normal_idx = 0;
		}
		self.meshbool_impl.get_mesh_gl_impl(normal_idx)
	}

	///Returns a MeshGL64 that is designed
	///to easily push into a renderer, including all interleaved vertex properties
	///that may have been input. It also includes relations to all the input meshes
	///that form a part of this result and the transforms applied to each.
	///
	///@param normalIdx If this manifold has properties corresponding to normal
	///vectors, you can specify the first of the three consecutive property channels
	///forming the (x, y, z) normals, which will cause this output MeshGL to
	///automatically update these normals according to the applied transforms and
	///front/back side. normalIdx + 3 must be <= numProp, and all original meshes
	///must use the same channels for their normals. Default is -1: if
	///`CalculateNormals()` recorded normals at the standard slot, that slot is
	///used automatically; otherwise no normals are applied. If normals are
	///selected, the runTransform matrices will be removed from the output, to
	///avoid them being double-applied when round-tripped.
	///Same deprecation note as `GetMeshGL`: prefer the no-arg form after
	///`CalculateNormals(0)`.
	pub fn get_mesh_gl_64(&self, mut normal_idx: i32) -> MeshGL64 {
		if normal_idx < 0 && self.meshbool_impl.all_have_normals() {
			normal_idx = 0;
		}
		self.meshbool_impl.get_mesh_gl_impl(normal_idx)
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
	pub fn status(&self) -> MeshBoolError {
		// Routes through any attached ExecutionContext (see WithContext). The
		// atomic_load temporary pins the Impl's lifetime for the duration of the full
		// expression -- through the lazy eval inside GetCsgLeafNode -- so a
		// concurrent op= reseating ctx_ on this Manifold can't free the Impl out
		// from under us.
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
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
		if tolerance > meshbool_impl.tolerance {
			meshbool_impl.tolerance = tolerance;
			meshbool_impl.set_normals_and_coplanar();
			meshbool_impl.simplify_topology(0);
			meshbool_impl.sort_geometry();
		} else {
			// for reducing tolerance, we need to make sure it is still at least
			// equal to epsilon.
			meshbool_impl.tolerance = meshbool_impl.epsilon.max(tolerance);
		}

		Self::from(meshbool_impl)
	}

	///Return a copy of the manifold simplified to the given tolerance, but with its
	///actual tolerance value unchanged. If the tolerance is not given or is less
	///than the current tolerance, the current tolerance is used for simplification.
	///The result will contain a subset of the original verts and all surfaces will
	///have moved by less than tolerance.
	pub fn simplify(&self, tolerance: Option<f64>) -> Self {
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
		let old_tolerance = meshbool_impl.tolerance;
		let tolerance = tolerance.unwrap_or(old_tolerance);
		if tolerance > old_tolerance {
			meshbool_impl.tolerance = tolerance;
			meshbool_impl.set_normals_and_coplanar();
		}

		meshbool_impl.simplify_topology(0);
		meshbool_impl.sort_geometry();
		meshbool_impl.tolerance = old_tolerance;
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
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut new_impl = self.meshbool_impl.clone();
		new_impl.initialize_original();
		new_impl.set_normals_and_coplanar();
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

	///Applies an Euler angle rotation to the manifold, This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///We use degrees so that we can minimize rounding error, and eliminate it
	///completely for any multiples of 90 degrees. Additionally, more efficient code
	///paths are used to update the manifold when the transforms only rotate by
	///multiples of 90 degrees.
	///
	///From the reference frame of the model being rotated, rotations are applied in
	///*z-y'-x"* order. That is yaw first, then pitch and finally roll.
	///
	///From the global reference frame, a model will be rotated in *x-y-z* order.
	///That is about the global X axis, then global Y axis, and finally global Z.
	///
	///@param xDegrees First rotation, degrees about the global X-axis.
	///@param yDegrees Second rotation, degrees about the global Y-axis.
	///@param zDegrees Third rotation, degrees about the global Z-axis.
	pub fn rotate(&self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> Self {
		let rx = Matrix3::from_column_slice(&[
			1.0,
			0.0,
			0.0,
			0.0,
			cosd(x_degrees),
			sind(x_degrees),
			0.0,
			-sind(x_degrees),
			cosd(x_degrees),
		]);
		let ry = Matrix3::from_column_slice(&[
			cosd(y_degrees),
			0.0,
			-sind(y_degrees),
			0.0,
			1.0,
			0.0,
			sind(y_degrees),
			0.0,
			cosd(y_degrees),
		]);
		let rz = Matrix3::from_column_slice(&[
			cosd(z_degrees),
			sind(z_degrees),
			0.0,
			-sind(z_degrees),
			cosd(z_degrees),
			0.0,
			0.0,
			0.0,
			1.0,
		]);

		let mut transform = Matrix3x4::default();
		transform
			.fixed_view_mut::<3, 3>(0, 0)
			.copy_from(&(rz * ry * rx));
		self.transform(&transform)
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
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}
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
	///Any normals recording set by `CalculateNormals()` is preserved across the
	///Warp, but the stored values reflect the pre-warp surface and may no longer
	///match the new geometry. Re-call `CalculateNormals()` if accurate normals
	///matter after a non-rigid warp.
	///
	///@param warpFunc A function that modifies a given vertex position.
	pub fn warp(&self, warp_func: impl FnMut(&mut Point3<f64>)) -> Self {
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
		meshbool_impl.warp(warp_func);
		Self::from(meshbool_impl)
	}

	///Same as Manifold::Warp but calls warpFunc with
	///a VecView which is roughly equivalent to std::span
	///pointing to all vec3 elements to be modified in-place. Like Warp, this
	///preserves any normals recording without updating the stored values;
	///re-call `CalculateNormals()` if accurate normals matter after a non-rigid
	///warp.
	///
	///@param warpFunc A function that modifies multiple vertex positions.
	pub fn warp_batch(&self, warp_func: impl FnMut(&mut [Point3<f64>])) -> Self {
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
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
	///Any normals recording set by `CalculateNormals()` is preserved. If the new
	///properties overwrite slot 0..2 with non-normal data, the recording becomes
	///stale; re-call `CalculateNormals()` (or use a numProp < 3 call followed by
	///CalculateNormals) to reset.
	///
	///@param num_prop The new number of properties per vertex.
	///@param prop_func A function that modifies the properties of a given vertex.
	pub fn set_properties(
		&self,
		num_prop: i32,
		prop_func: Option<impl FnMut(&mut [f64], Point3<f64>, &[f64])>,
	) -> Self {
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
		let old_num_prop = self.num_prop();
		let old_properties = meshbool_impl.properties.clone();

		if num_prop == 0 {
			meshbool_impl.properties = Vec::new();
		} else {
			meshbool_impl.properties = vec![0.0; num_prop as usize * self.num_prop_vert()];

			if let Some(mut prop_func) = prop_func {
				for tri in 0..self.num_tri() {
					for i in 0..3 {
						let edge = (3 * tri + i) as i32;
						let vert = meshbool_impl.halfedge.start(edge);
						let prop_vert = meshbool_impl.halfedge.prop(edge);
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
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
		meshbool_impl.calculate_curvature(gaussian_idx, mean_idx);
		Self::from(meshbool_impl)
	}

	///Fills in vertex properties for normal vectors, calculated from the mesh
	///geometry.
	///
	///@param normalIdx The property channel in which to store the X values of the
	///normals. The X, Y, and Z channels will be sequential. The property set will
	///be automatically expanded such that NumProp will be at least normalIdx + 3.
	///Default is 0, the standard slot (MeshGL channels 3, 4, 5); the Manifold
	///records the recording per-meshID in that case so subsequent `GetMeshGL()` /
	///`GetMeshGL64()` without an explicit normalIdx will return world-frame
	///normals and mark each output run via runFlags bit 1. Non-zero values are
	///retained for compatibility and will not be supported in a future release.
	///
	///@param minSharpAngle Any edges with angles greater than this value will
	///remain sharp, getting different normal vector properties on each side of the
	///edge. By default, no edges are sharp and all normals are shared. With a value
	///of zero, the model is faceted and all normals match their triangle normals,
	///but in this case it would be better not to calculate normals at all.
	pub fn calculate_normals(&self, normal_idx: i32, min_sharp_angle: f64) -> Self {
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return Self::propagate_status(self.meshbool_impl.status);
		}

		let mut meshbool_impl = self.meshbool_impl.clone();
		meshbool_impl.set_normals(normal_idx, min_sharp_angle);
		// Mark per-meshID hasNormals so GetMeshGL(-1) can auto-substitute slot 0 on
		// export. Restricted to the standard slot since a non-zero slot would be
		// ambiguous when round-tripping through MeshGL.
		if normal_idx == 0 {
			for rel in meshbool_impl.mesh_relation.mesh_id_transform.values_mut() {
				rel.has_normals = true;
			}
		}
		return Self::from(meshbool_impl);
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
		if self.meshbool_impl.status != MeshBoolError::NoError {
			return (
				Self::propagate_status(self.meshbool_impl.status),
				Self::propagate_status(self.meshbool_impl.status),
			);
		}
		if self.is_empty() {
			return (Self::default(), Self::default());
		}
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
	fn from(meshbool_impl: MeshBoolImpl) -> Self {
		Self { meshbool_impl }
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
pub enum MeshBoolError {
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
