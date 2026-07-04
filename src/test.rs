use crate::MeshBool;
use crate::meshgl::MeshGLError;
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::bvh_collider::SimpleRecorder;
use crate::util::tri_dst::distance_triangle_triangle_squared;
use nalgebra::{Point2, Point3, Vector3};
use std::sync::atomic::{AtomicBool, Ordering};

include!(concat!(env!("OUT_DIR"), "/test.rs"));

/// Perform extra sanity checks and assertions on the intermediate data
/// structures.
static INTERMEDIATE_CHECKS: AtomicBool = AtomicBool::new(false);
/// Perform 3D mesh self-intersection test on intermediate boolean results to
/// test for ϵ-validity. For debug purposes only.
static SELF_INTERSECTION_CHECKS: AtomicBool = AtomicBool::new(false);
/// If processOverlaps is false, a geometric check will be performed to assert
/// all triangles are CCW.
static PROCESS_OVERLAPS: AtomicBool = AtomicBool::new(true);

pub fn get_intermediate_checks() -> bool {
	INTERMEDIATE_CHECKS.load(Ordering::Relaxed)
}

pub fn set_intermediate_checks(value: bool) {
	INTERMEDIATE_CHECKS.store(value, Ordering::Relaxed);
}

pub fn get_self_intersection_checks() -> bool {
	SELF_INTERSECTION_CHECKS.load(Ordering::Relaxed)
}

pub fn set_self_intersection_checks(value: bool) {
	SELF_INTERSECTION_CHECKS.store(value, Ordering::Relaxed);
}

pub fn get_process_overlaps() -> bool {
	PROCESS_OVERLAPS.load(Ordering::Relaxed)
}

pub fn set_process_overlaps(value: bool) {
	PROCESS_OVERLAPS.store(value, Ordering::Relaxed);
}

impl MeshGLError {
	pub fn is_non_finite_vertex(&self) -> bool {
		*self == MeshGLError::NonFiniteVertex
	}
	pub fn is_invalid_construction(&self) -> bool {
		*self == MeshGLError::InvalidConstruction
	}
	pub fn is_result_too_large(&self) -> bool {
		*self == MeshGLError::ResultTooLarge
	}
	pub fn is_not_manifold(&self) -> bool {
		*self == MeshGLError::NotManifold
	}
	pub fn is_missing_position_properties(&self) -> bool {
		*self == MeshGLError::MissingPositionProperties
	}
	pub fn is_merge_vectors_different_lengths(&self) -> bool {
		*self == MeshGLError::MergeVectorsDifferentLengths
	}
	pub fn is_transform_wrong_length(&self) -> bool {
		*self == MeshGLError::TransformWrongLength
	}
	pub fn is_run_index_wrong_length(&self) -> bool {
		*self == MeshGLError::RunIndexWrongLength
	}
	pub fn is_face_id_wrong_length(&self) -> bool {
		*self == MeshGLError::FaceIDWrongLength
	}
	pub fn is_merge_index_out_of_bounds(&self) -> bool {
		*self == MeshGLError::MergeIndexOutOfBounds
	}
	pub fn is_vertex_out_of_bounds(&self) -> bool {
		*self == MeshGLError::VertexOutOfBounds
	}
}

pub trait Vector3i32Coords {
	fn get_x(&self) -> i32;
	fn get_y(&self) -> i32;
	fn get_z(&self) -> i32;
}

impl Vector3i32Coords for Vector3<i32> {
	fn get_x(&self) -> i32 {
		self.x
	}
	fn get_y(&self) -> i32 {
		self.y
	}
	fn get_z(&self) -> i32 {
		self.z
	}
}

pub trait Point2f64Coords {
	fn get_x(&self) -> f64;
	fn get_y(&self) -> f64;
}

impl Point2f64Coords for Point2<f64> {
	fn get_x(&self) -> f64 {
		self.x
	}
	fn get_y(&self) -> f64 {
		self.y
	}
}

pub trait Point3f64Coords {
	fn get_x(&self) -> f64;
	fn get_y(&self) -> f64;
	fn get_z(&self) -> f64;
}

impl Point3f64Coords for Point3<f64> {
	fn get_x(&self) -> f64 {
		self.x
	}
	fn get_y(&self) -> f64 {
		self.y
	}
	fn get_z(&self) -> f64 {
		self.z
	}
}

impl MeshBool {
	///Returns true if this manifold is self-intersecting.
	///Note that this is not checking for epsilon-validity.
	pub fn is_self_intersecting(&self) -> bool {
		let ep = 2.0 * self.precision.epsilon;
		let epsilon_sq = ep * ep;
		let (tri_box, _) = get_tri_box_morton(&self.tri.halfedge, &self.vert_pos, None);

		let intersecting = AtomicBool::new(false);

		let mut f = |tri0: i32, tri1: i32| {
			let mut tri_verts0: [Point3<f64>; 3] = [Point3::default(); 3];
			let mut tri_verts1: [Point3<f64>; 3] = [Point3::default(); 3];
			for i in 0..3 {
				tri_verts0[i as usize] =
					self.vert_pos[self.tri.halfedge.start(3 * tri0 + i) as usize];
				tri_verts1[i as usize] =
					self.vert_pos[self.tri.halfedge.start(3 * tri1 + i) as usize];
			}
			// if triangles tri0 and tri1 share a vertex, return true to skip the
			// check. we relax the sharing criteria a bit to allow for at most
			// distance epsilon squared
			for i in 0..3 {
				for j in 0..3 {
					if (tri_verts1[j] - tri_verts0[i]).magnitude_squared() <= epsilon_sq {
						return;
					}
				}
			}

			if distance_triangle_triangle_squared(&tri_verts0, &tri_verts1) == 0.0 {
				// try to move the triangles around the normal of the other face
				let mut tmp0: [Point3<f64>; 3] = [Point3::default(); 3];
				let mut tmp1: [Point3<f64>; 3] = [Point3::default(); 3];
				for i in 0..3 {
					tmp0[i] = tri_verts0[i] + ep * self.tri.normal[tri1 as usize];
				}
				if distance_triangle_triangle_squared(&tmp0, &tri_verts1) > 0.0 {
					return;
				}
				for i in 0..3 {
					tmp0[i] = tri_verts0[i] - ep * self.tri.normal[tri1 as usize];
				}
				if distance_triangle_triangle_squared(&tmp0, &tri_verts1) > 0.0 {
					return;
				}
				for i in 0..3 {
					tmp1[i] = tri_verts1[i] + ep * self.tri.normal[tri0 as usize];
				}
				if distance_triangle_triangle_squared(&tri_verts0, &tmp1) > 0.0 {
					return;
				}
				for i in 0..3 {
					tmp1[i] = tri_verts1[i] - ep * self.tri.normal[tri0 as usize];
				}
				if distance_triangle_triangle_squared(&tri_verts0, &tmp1) > 0.0 {
					return;
				}

				#[cfg(feature = "test_thoroughly")]
				intersecting.store(true, Ordering::SeqCst);
			}
		};

		let mut recorder = SimpleRecorder::new(&mut f);
		self.collider
			.collisions_from_slice::<true, _>(&mut recorder, &tri_box, true);

		intersecting.load(Ordering::SeqCst)
	}
}
