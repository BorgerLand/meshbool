use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation, tri_has_normals};
use crate::ops::boolean::expression::CSGExpression;
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::math::{
	is_axis_aligned, mat3, mul_mat3x4, normal_transform, safe_normalize3, transform_normal,
};
use crate::{Box3D, MeshBool, Precision, Properties, Triangles};
use crate::{TrianglesPartial, postprocessing as pp};
use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3, Vector4};
use std::mem;
use std::rc::Rc;

impl MeshBool {
	///Move this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to add to every vertex.
	pub fn translate(self, v: Vector3<f64>) -> CSGExpression {
		CSGExpression::from(self).translate(v)
	}

	///Scale this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to multiply every vertex by per component.
	pub fn scale(self, v: Vector3<f64>) -> CSGExpression {
		CSGExpression::from(self).scale(v)
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
	pub fn rotate(self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> CSGExpression {
		CSGExpression::from(self).rotate(x_degrees, y_degrees, z_degrees)
	}

	///Mirror this Manifold over the plane described by the unit form of the given
	///normal vector. If the length of the normal is zero, an empty Manifold is
	///returned. This operation can be chained. Transforms are combined and applied
	///lazily.
	///
	///@param normal The normal vector of the plane to be mirrored over
	pub fn mirror(self, normal: Vector3<f64>) -> CSGExpression {
		CSGExpression::from(self).mirror(normal)
	}

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(self, transform: Matrix3x4<f64>) -> CSGExpression {
		CSGExpression::from(self).transform(transform)
	}

	pub(crate) fn apply_transform(self, transform: Matrix3x4<f64>) -> Self {
		if transform == Matrix3x4::identity() {
			return self.clone();
		}

		let num_prop_vert = self.num_prop_vert();

		let mut instance_rel = Rc::unwrap_or_clone(self.instance_relation);
		for rel in instance_rel.iter_mut() {
			rel.transform = mul_mat3x4(transform, rel.transform);
		}
		let instance_rel = Rc::new(instance_rel);

		if !transform.iter().fold(true, |acc, e| acc && e.is_finite()) {
			return Self::decimated(None, instance_rel, self.properties.stride, self.precision);
		}

		let mut vert_pos = Rc::unwrap_or_clone(self.vert_pos);
		for v in vert_pos.iter_mut() {
			*v = (transform * Vector4::new(v.x, v.y, v.z, 1.0)).into();
		}

		let normal_transform = normal_transform(transform);
		let mut tri_normal = Rc::unwrap_or_clone(self.tri.normal);
		for n in tri_normal.iter_mut() {
			*n = transform_normal(normal_transform, *n);
		}

		let mut properties = Rc::unwrap_or_clone(self.properties);
		if properties.stride >= 3 {
			eager_transform_prop_normals(
				&mut properties,
				&self.tri.halfedge,
				&instance_rel,
				&self.tri.relation,
				normal_transform,
				num_prop_vert,
				0,
			);
		}

		let mut halfedge = Rc::unwrap_or_clone(self.tri.halfedge);
		let invert = mat3(transform).determinant() < 0.0;
		if invert {
			for tri in 0..halfedge.num_tri() {
				FlipTris {
					halfedge: &mut halfedge,
				}
				.call(tri);
			}
		}

		let epsilon_spectral_norm =
			self.precision.epsilon * mat3(transform).svd(false, false).singular_values[0];
		let mut precision = Precision::new(
			Box3D::from_cloud(&vert_pos),
			self.precision.tolerance,
			false,
		);
		precision.epsilon = precision.epsilon.max(epsilon_spectral_norm);
		precision.tolerance = precision.tolerance.max(epsilon_spectral_norm);

		let collider = Rc::new(if halfedge.num_tri() == 0 {
			BVHCollider::default()
		} else {
			let mut collider = Rc::unwrap_or_clone(self.collider);
			if is_axis_aligned(transform) {
				collider.transform_axis_aligned(transform);
			} else {
				let (tri_box, _) = get_tri_box_morton(&halfedge, &vert_pos, None);
				collider.update_boxes(&tri_box);
			}

			collider
		});

		Self {
			original_id: None,
			precision,
			vert_pos: Rc::new(vert_pos),
			properties: Rc::new(properties),
			tri: Triangles {
				halfedge: Rc::new(halfedge),
				normal: Rc::new(tri_normal),
				relation: self.tri.relation,
			},
			instance_relation: instance_rel,
			collider,
		}
	}
}

fn eager_transform_prop_normals(
	properties: &mut Properties,
	halfedge: &Halfedges,
	instance_rel: &[InstanceRelation],
	tri_rel: &[TriRelation],
	normal_transform: Matrix3<f64>,
	num_prop_vert: usize,
	offset: usize,
) {
	// Short-circuit when no meshID carries normals. OR semantics (any has
	// it), unlike AllHaveNormals() - mixed inputs still need the per-meshID
	// iteration below to rotate the with-normals subset.
	let mut any_has_normals = false;
	for m in instance_rel {
		if m.has_normals {
			any_has_normals = true;
			break;
		}
	}

	if !any_has_normals {
		return;
	}
	let mut prop_visited = vec![false; num_prop_vert];
	for e in 0..halfedge.len() {
		if !tri_has_normals(instance_rel, tri_rel[e / 3]) {
			continue;
		}
		let prop = halfedge.prop[e];
		if prop < 0 || prop_visited[prop as usize] {
			continue;
		}
		let prop = prop as usize;
		prop_visited[prop] = true;
		let mut n = Vector3::default();
		for i in 0..3 {
			n[i] = properties.data[(offset + prop) * properties.stride + i];
		}
		// Re-normalize as we transform: non-orthogonal transforms (scale) and
		// barycentric interpolation upstream both leave non-unit values that
		// would otherwise compound and break downstream lighting / smoothing.
		n = safe_normalize3(normal_transform * n);
		for i in 0..3 {
			properties.data[(offset + prop) * properties.stride + i] = n[i];
		}
	}
}

pub struct FlipTris<'a> {
	pub halfedge: &'a mut Halfedges,
}

impl<'a> FlipTris<'a> {
	pub fn call(&mut self, tri: usize) {
		let mut face = [
			self.halfedge.get(3 * tri + 2),
			self.halfedge.get(3 * tri + 1),
			self.halfedge.get(3 * tri),
		];
		for i in 0..3 {
			mem::swap(&mut face[i].start_vert, &mut face[i].end_vert);
			face[i].paired_halfedge = flip_halfedge(face[i].paired_halfedge);
		}
		for i in 0..3 {
			self.halfedge.start[3 * tri + i] = face[i].start_vert;
			self.halfedge.pair[3 * tri + i] = face[i].paired_halfedge;
			self.halfedge.prop[3 * tri + i] = face[i].prop_vert;
		}
	}
}

#[inline(always)]
fn flip_halfedge(halfedge: i32) -> i32 {
	let tri = halfedge / 3;
	let vert = 2 - (halfedge - 3 * tri);
	3 * tri + vert
}

impl MeshBool {
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
	pub fn warp(self, mut warp_func: impl FnMut(&mut Point3<f64>)) -> Self {
		self.warp_batch(|vecs| {
			vecs.iter_mut().for_each(|v| warp_func(v));
		})
	}

	///Same as Manifold::Warp but calls warpFunc with
	///a VecView which is roughly equivalent to std::span
	///pointing to all vec3 elements to be modified in-place. Like Warp, this
	///preserves any normals recording without updating the stored values;
	///re-call `CalculateNormals()` if accurate normals matter after a non-rigid
	///warp.
	///
	///@param warpFunc A function that modifies multiple vertex positions.
	pub fn warp_batch(self, mut warp_func: impl FnMut(&mut [Point3<f64>])) -> Self {
		drop(self.collider);
		drop(self.tri.normal);

		let mut vert_pos = Rc::unwrap_or_clone(self.vert_pos);
		warp_func(&mut vert_pos);

		let bbox = Box3D::from_cloud(&vert_pos);
		if !bbox.is_finite() {
			return Self::decimated(
				None,
				self.instance_relation,
				self.properties.stride,
				self.precision,
			);
		}

		let precision = Precision::new(bbox, self.precision.tolerance, false);
		let mut properties = Rc::unwrap_or_clone(self.properties);
		let mut halfedge = Rc::unwrap_or_clone(self.tri.halfedge);
		let mut tri_rel = Rc::unwrap_or_clone(self.tri.relation);
		let collider = pp::sort_and_compact_geometry(
			&mut vert_pos,
			&mut properties,
			TrianglesPartial {
				halfedge: &mut halfedge,
				normal: None,
				relation: Some(&mut tri_rel),
			},
			bbox,
		)
		.unwrap();
		let tri_normal = pp::set_normals_and_coplanar(
			&mut tri_rel,
			&self.instance_relation,
			&halfedge,
			&vert_pos,
			precision.tolerance,
		);

		Self {
			original_id: None,
			precision,
			vert_pos: Rc::new(vert_pos),
			properties: Rc::new(properties),
			tri: Triangles {
				halfedge: Rc::new(halfedge),
				normal: Rc::new(tri_normal),
				relation: Rc::new(tri_rel),
			},
			instance_relation: self.instance_relation,
			collider,
		}
	}
}
