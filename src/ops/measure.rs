use crate::MeshBool;
use crate::ops::boolean::{OpType, boolean};
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::aabb::Box3D;
use crate::util::math::{ccw, get_axis_aligned_projection};
use crate::util::tri_dst::distance_triangle_triangle_squared;
use nalgebra::{Point2, Point3, Vector3};

impl MeshBool {
	///The genus is a topological property of the manifold, representing the number
	///of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
	///mesh, so it is best to call Decompose() first.
	pub fn genus(&self) -> i32 {
		let chi = (self.num_vert() as i32) - (self.num_edge() as i32) + (self.num_tri() as i32);
		1 - chi / 2
	}

	///The number of triangles that are colinear within tolerance. This library
	///attempts to remove all of these, but it cannot always remove all of them
	///without changing the mesh by too much.
	pub fn num_degenerate_tris(&self) -> usize {
		if self.is_empty() {
			return 0;
		}
		(0..self.num_tri())
			.filter(|&tri| {
				if self.tri.halfedge.pair[3 * tri] < 0 {
					return false;
				}

				let projection = get_axis_aligned_projection(self.tri.normal[tri]);
				let mut v = [Point2::default(); 3];
				for i in 0..3 {
					v[i] =
						projection * self.vert_pos[self.tri.halfedge.start[3 * tri + i] as usize];
				}

				let ccw = ccw(v[0], v[1], v[2], self.precision.tolerance / 2.0);
				ccw == 0
			})
			.count()
	}

	///Returns the surface area of the manifold.
	pub fn surface_area(&self) -> f64 {
		self.get_property(|tri| {
			let v = self.vert_pos[self.tri.halfedge.start[3 * tri] as usize].coords;
			(self.vert_pos[self.tri.halfedge.start[3 * tri + 1] as usize] - v)
				.coords
				.cross(&(self.vert_pos[self.tri.halfedge.start[3 * tri + 2] as usize] - v).coords)
				.norm() / 2.0
		})
	}

	///Returns the volume of the manifold.
	pub fn volume(&self) -> f64 {
		self.get_property(|tri| {
			let v = self.vert_pos[self.tri.halfedge.start[3 * tri] as usize].coords;
			let cross_p = (self.vert_pos[self.tri.halfedge.start[3 * tri + 1] as usize] - v)
				.coords
				.cross(&(self.vert_pos[self.tri.halfedge.start[3 * tri + 2] as usize] - v).coords);
			cross_p.dot(&v) / 6.0
		})
	}

	fn get_property(&self, mut term: impl FnMut(usize) -> f64) -> f64 {
		if self.is_empty() {
			return 0.0;
		}

		// Kahan summation
		let mut value = 0.0;
		let mut value_compensation = 0.0;
		for i in 0..self.num_tri() {
			let value1 = term(i);
			let t = value + value1;
			value_compensation += (value - t) + value1;
			value = t;
		}
		value + value_compensation
	}

	///Returns the minimum gap between two manifolds. Returns a double between
	///0 and searchLength.
	///
	///@param other The other manifold to compute the minimum gap to.
	///@param searchLength The maximum distance to search for a minimum gap.
	pub fn min_gap(&self, other: &Self, search_length: f64) -> f64 {
		if let Ok(intersect) = boolean(self.clone(), OpType::Intersection, other.clone())
			&& !intersect.is_empty()
		{
			return 0.0;
		}

		let (mut face_box_other, _) =
			get_tri_box_morton(&other.tri.halfedge, &other.vert_pos, None);

		for aabb in face_box_other.iter_mut() {
			*aabb = Box3D::new(
				(aabb.min.coords - Vector3::repeat(search_length)).into(),
				(aabb.max.coords + Vector3::repeat(search_length)).into(),
			);
		}

		let mut min_distance = f64::INFINITY;
		self.collider.collisions_from_slice::<false, _>(
			|tri_other, tri| {
				let mut p: [Point3<f64>; 3] = Default::default();
				let mut q: [Point3<f64>; 3] = Default::default();

				for j in 0..3 {
					p[j] = self.vert_pos[self.tri.halfedge.start[3 * tri + j] as usize];
					q[j] = other.vert_pos[other.tri.halfedge.start[3 * tri_other + j] as usize];
				}
				min_distance = min_distance.min(distance_triangle_triangle_squared(&p, &q));
			},
			&face_box_other,
			false,
		);
		min_distance.min(search_length * search_length).sqrt()
	}

	///Returns true if properties are shared everywhere except across mesh
	///boundaries. This is not true in general, but only because an input mesh may
	///have property discontinuities. For simple input meshes where properties are
	///1:1 with verts, this HasSimpleProps condition should still be true after any
	///combination of boolean operations and simplifications. CalculateNormals()
	///will cause this to be false anytime the mesh contains a sharp edge.
	pub fn has_simple_props(&self) -> bool {
		if self.tri.halfedge.len() == 0 || self.properties.stride == 0 {
			return true;
		}
		(0..self.tri.halfedge.len()).all(|edge| {
			let pair = self.tri.halfedge.pair[edge];
			if pair < 0 || !self.tri.halfedge.is_forward(edge) {
				return true;
			}

			let pair = pair as usize;
			let props_match = self.tri.halfedge.prop[edge] == self.tri.halfedge.prop_end(pair)
				&& self.tri.halfedge.prop[pair] == self.tri.halfedge.prop_end(edge);
			let meshes_match = self.tri.relation[self.tri.halfedge.tri(edge)].instance_id
				== self.tri.relation[self.tri.halfedge.tri(pair)].instance_id;
			meshes_match == props_match
		})
	}

	///The triangle normal vectors are saved over the course of operations rather
	///than recalculated to avoid rounding error. This checks that triangles still
	///match their normal vectors within Precision(), and if all triangles are CCW
	///relative to their triNormals_
	#[cfg(feature = "test")]
	pub fn matches_tri_normals(&self) -> bool {
		if self.tri.halfedge.len() == 0 || self.tri.normal.len() != self.num_tri() {
			return true;
		}
		return (0..self.num_tri()).all(|face| {
			if self.tri.halfedge.pair[3 * face] < 0 {
				return true;
			}

			let projection = get_axis_aligned_projection(self.tri.normal[face]);
			let mut v = [Point2::default(); 3];
			let mut max = -f64::INFINITY;
			let mut min = f64::INFINITY;
			for i in 0..3 {
				let p = self.vert_pos[self.tri.halfedge.start[3 * face + i] as usize];
				v[i] = projection * p;
				let d = p.coords.dot(&self.tri.normal[face]);
				if !d.is_finite() {
					return true;
				}
				max = max.max(d);
				min = min.min(d);
			}
			if max - min > 2.0 * self.precision.tolerance {
				return false;
			}

			let ccw = ccw(v[0], v[1], v[2], self.precision.epsilon * 2.0);
			return ccw >= 0;
		});
	}
}
