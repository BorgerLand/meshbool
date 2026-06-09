//anything that can't be cleanly expressed in test.zng can go here

use crate::MeshBoolError;
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

impl MeshBoolError {
	pub fn is_no_error(&self) -> bool {
		*self == MeshBoolError::NoError
	}
	pub fn is_non_finite_vertex(&self) -> bool {
		*self == MeshBoolError::NonFiniteVertex
	}
	pub fn is_invalid_construction(&self) -> bool {
		*self == MeshBoolError::InvalidConstruction
	}
	pub fn is_result_too_large(&self) -> bool {
		*self == MeshBoolError::ResultTooLarge
	}
	pub fn is_not_manifold(&self) -> bool {
		*self == MeshBoolError::NotManifold
	}
	pub fn is_missing_position_properties(&self) -> bool {
		*self == MeshBoolError::MissingPositionProperties
	}
	pub fn is_merge_vectors_different_lengths(&self) -> bool {
		*self == MeshBoolError::MergeVectorsDifferentLengths
	}
	pub fn is_transform_wrong_length(&self) -> bool {
		*self == MeshBoolError::TransformWrongLength
	}
	pub fn is_run_index_wrong_length(&self) -> bool {
		*self == MeshBoolError::RunIndexWrongLength
	}
	pub fn is_face_id_wrong_length(&self) -> bool {
		*self == MeshBoolError::FaceIDWrongLength
	}
	pub fn is_merge_index_out_of_bounds(&self) -> bool {
		*self == MeshBoolError::MergeIndexOutOfBounds
	}
	pub fn is_vertex_out_of_bounds(&self) -> bool {
		*self == MeshBoolError::VertexOutOfBounds
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
