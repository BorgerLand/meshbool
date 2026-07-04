use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation, tri_has_normals};
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::hash_table::DeterministicMap;
use crate::util::math::{
	cosd, is_axis_aligned, mat3, mat4, normal_transform, safe_normalize3, sind, transform_normal,
};
use crate::{Box3D, MeshBool, Precision, Properties, Triangles};
use crate::{TrianglesPartial, postprocessing as pp};
use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3, Vector4};
use std::mem;

impl MeshBool {
	///Move this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to add to every vertex.
	pub fn translate(&self, v: Vector3<f64>) -> Self {
		let mut transform = Matrix3x4::<f64>::identity();
		*transform.column_mut(3) = *v;
		self.transform(transform)
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

		self.transform(transform)
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
		self.transform(transform)
	}

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(&self, transform: Matrix3x4<f64>) -> Self {
		if transform == Matrix3x4::identity() {
			return self.clone();
		}

		let instance_relation = self
			.instance_relation
			.iter()
			.map(|(&instance_id, &rel)| {
				let mut rel = rel;
				rel.transform = transform * mat4(rel.transform);
				(instance_id, rel)
			})
			.collect();

		if !transform.iter().fold(true, |acc, e| acc && e.is_finite()) {
			return Self::decimated(
				None,
				instance_relation,
				self.properties.stride,
				self.precision,
			);
		}

		let vert_pos: Vec<_> = self
			.vert_pos
			.iter()
			.map(|v| (transform * Vector4::new(v.x, v.y, v.z, 1.0)).into())
			.collect();

		let normal_transform = normal_transform(transform);
		let tri_normal = self
			.tri
			.normal
			.iter()
			.map(|&n| transform_normal(normal_transform, n))
			.collect();

		let mut properties = self.properties.clone();
		if properties.stride >= 3 {
			eager_transform_prop_normals(
				&mut properties,
				&self.tri.halfedge,
				&self.instance_relation,
				&self.tri.relation,
				normal_transform,
				self.num_prop_vert(),
				0,
			);
		}

		let mut halfedge = self.tri.halfedge.clone();
		let invert = mat3(transform).determinant() < 0.0;
		if invert {
			for tri in 0..self.num_tri() {
				FlipTris {
					halfedge: &mut halfedge,
				}
				.call(tri as i32);
			}
		}

		let epsilon = self.precision.epsilon * mat3(transform).svd(false, false).singular_values[0];
		let precision = Precision {
			epsilon: self.precision.epsilon.max(epsilon),
			tolerance: self.precision.tolerance.max(epsilon),
		};

		let collider = if self.is_empty() {
			BVHCollider::default()
		} else {
			let mut collider = self.collider.clone();
			if is_axis_aligned(transform) {
				collider.transform(transform);
			} else {
				let (tri_box, _) = get_tri_box_morton(&halfedge, &vert_pos, None);
				collider.update_boxes(&tri_box);
			}

			collider
		};

		Self {
			original_id: None,
			precision,
			vert_pos,
			properties,
			tri: Triangles {
				halfedge,
				normal: tri_normal,
				relation: self.tri.relation.clone(),
			},
			instance_relation,
			collider,
		}
	}
}

fn eager_transform_prop_normals(
	properties: &mut Properties,
	halfedge: &Halfedges,
	instance_rel: &DeterministicMap<u32, InstanceRelation>,
	tri_rel: &[TriRelation],
	normal_transform: Matrix3<f64>,
	num_prop_vert: usize,
	offset: usize,
) {
	// Short-circuit when no meshID carries normals. OR semantics (any has
	// it), unlike AllHaveNormals() - mixed inputs still need the per-meshID
	// iteration below to rotate the with-normals subset.
	let mut any_has_normals = false;
	for m in instance_rel.values() {
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
		let prop = halfedge.prop(e as i32);
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
	pub fn call(&mut self, tri: i32) {
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
			self.halfedge
				.set_start(3 * tri + i, face[i as usize].start_vert);
			self.halfedge
				.set_pair(3 * tri + i, face[i as usize].paired_halfedge);
			self.halfedge
				.set_prop(3 * tri + i, face[i as usize].prop_vert);
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
	///Mirror this Manifold over the plane described by the unit form of the given
	///normal vector. If the length of the normal is zero, an empty Manifold is
	///returned. This operation can be chained. Transforms are combined and applied
	///lazily.
	///
	///@param normal The normal vector of the plane to be mirrored over
	pub fn mirror(&self, normal: Vector3<f64>) -> Self {
		if normal.magnitude_squared() == 0.0 {
			return Self::decimated(
				None,
				self.instance_relation.clone(),
				self.properties.stride,
				self.precision,
			);
		}
		let n = normal.normalize();
		let m = Matrix3::identity() - (2.0 * (n * n.transpose()));
		let m = Matrix3x4::from_columns(&[
			m.column(0).into(),
			m.column(1).into(),
			m.column(2).into(),
			Vector3::default(),
		]);
		self.transform(m)
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
	pub fn warp(&self, mut warp_func: impl FnMut(&mut Point3<f64>)) -> Self {
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
	pub fn warp_batch(&self, mut warp_func: impl FnMut(&mut [Point3<f64>])) -> Self {
		let mut vert_pos = self.vert_pos.clone();
		warp_func(&mut vert_pos);

		let bbox = Box3D::from_cloud(&vert_pos);
		if !bbox.is_finite() {
			return Self::decimated(
				None,
				self.instance_relation.clone(),
				self.properties.stride,
				self.precision,
			);
		}

		let precision = Precision::new(bbox, self.precision.tolerance, false);
		let mut properties = self.properties.clone();
		let mut halfedge = self.tri.halfedge.clone();
		let mut tri_rel = self.tri.relation.clone();
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
		let tri_normal =
			pp::set_normals_and_coplanar(&mut tri_rel, &halfedge, &vert_pos, precision.tolerance);

		Self {
			original_id: None,
			precision,
			vert_pos,
			properties,
			tri: Triangles {
				halfedge,
				normal: tri_normal,
				relation: self.tri.relation.clone(),
			},
			instance_relation: self.instance_relation.clone(),
			collider,
		}
	}
}
