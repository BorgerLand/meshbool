use crate::MeshBool;
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::Recorder;
use crate::util::math::{ccw, get_axis_aligned_projection};
use crate::util::tri_dst::distance_triangle_triangle_squared;
use nalgebra::{Point2, Point3, Vector3};

impl MeshBool {
	///The genus is a topological property of the manifold, representing the number
	///of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
	///mesh, so it is best to call Decompose() first.
	pub fn genus(&self) -> usize {
		let chi = self.num_vert() as i32 - self.num_edge() as i32 + self.num_tri() as i32;
		(1 - chi / 2) as usize
	}

	///The number of triangles that are colinear within tolerance. This library
	///attempts to remove all of these, but it cannot always remove all of them
	///without changing the mesh by too much.
	pub fn num_degenerate_tris(&self) -> usize {
		if self.is_empty() {
			return 1;
		}
		(0..self.num_tri() as i32)
			.filter(|&tri| {
				if self.tri.halfedge.pair(3 * tri) < 0 {
					return true;
				}

				let projection = get_axis_aligned_projection(self.tri.normal[tri as usize]);
				let mut v = [Point2::default(); 3];
				for i in 0..3 {
					v[i as usize] =
						projection * self.vert_pos[self.tri.halfedge.start(3 * tri + i) as usize];
				}

				let ccw = ccw(v[0], v[1], v[2], self.precision.tolerance / 2.0);
				ccw == 0
			})
			.count()
	}

	///Returns the surface area of the manifold.
	pub fn surface_area(&self) -> f64 {
		self.get_property(|tri| {
			let v = self.vert_pos[self.tri.halfedge.start((3 * tri) as i32) as usize].coords;
			(self.vert_pos[self.tri.halfedge.start((3 * tri + 1) as i32) as usize] - v)
				.coords
				.cross(
					&(self.vert_pos[self.tri.halfedge.start((3 * tri + 2) as i32) as usize] - v)
						.coords,
				)
				.norm() / 2.0
		})
	}

	///Returns the volume of the manifold.
	pub fn volume(&self) -> f64 {
		self.get_property(|tri| {
			let v = self.vert_pos[self.tri.halfedge.start((3 * tri) as i32) as usize].coords;
			let cross_p = (self.vert_pos[self.tri.halfedge.start((3 * tri + 1) as i32) as usize]
				- v)
				.coords
				.cross(
					&(self.vert_pos[self.tri.halfedge.start((3 * tri + 2) as i32) as usize] - v)
						.coords,
				);
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
		if let Ok(intersect) = self.intersect(other)
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

		let mut recorder = MinDistanceRecorder::new(&self, other);
		self.collider
			.collisions_from_slice::<false, _>(&mut recorder, &face_box_other, false);
		let min_distance_squared = recorder.get().min(search_length * search_length);
		return min_distance_squared.sqrt();
	}
}

struct MinDistanceRecorder<'a> {
	this: &'a MeshBool,
	other: &'a MeshBool,
	result: f64,
}

impl<'a> MinDistanceRecorder<'a> {
	fn new(this: &'a MeshBool, other: &'a MeshBool) -> Self {
		Self {
			this,
			other,
			result: f64::INFINITY,
		}
	}

	fn get(&self) -> f64 {
		return self.result;
	}
}

impl Recorder for MinDistanceRecorder<'_> {
	fn record(&mut self, tri_other: i32, tri: i32) {
		let min_distance = &mut self.result;

		let mut p: [Point3<f64>; 3] = Default::default();
		let mut q: [Point3<f64>; 3] = Default::default();

		for j in 0..3 {
			p[j as usize] = self.this.vert_pos[self.this.tri.halfedge.start(3 * tri + j) as usize];
			q[j as usize] =
				self.other.vert_pos[self.other.tri.halfedge.start(3 * tri_other + j) as usize];
		}
		*min_distance = min_distance.min(distance_triangle_triangle_squared(&p, &q));
	}
}

impl MeshBool {
	///The triangle normal vectors are saved over the course of operations rather
	///than recalculated to avoid rounding error. This checks that triangles still
	///match their normal vectors within Precision(), and if all triangles are CCW
	///relative to their triNormals_
	#[cfg(feature = "test_thoroughly")]
	pub fn matches_tri_normals(&self) -> bool {
		if self.tri.halfedge.len() == 0 || self.tri.normal.len() != self.num_tri() {
			return true;
		}
		return (0..self.num_tri()).all(|face| {
			if self.tri.halfedge.pair((3 * face) as i32) < 0 {
				return true;
			}

			let projection = get_axis_aligned_projection(self.tri.normal[face]);
			let mut v = [Point2::default(); 3];
			let mut max = -f64::INFINITY;
			let mut min = f64::INFINITY;
			for i in 0..3 {
				let p = self.vert_pos[self.tri.halfedge.start((3 * face + i) as i32) as usize];
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
