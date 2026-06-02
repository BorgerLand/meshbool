//anything that can't be cleanly expressed in test.zng can go here

use crate::MeshBoolError;
use nalgebra::{Point2, Point3, Vector3};

include!(concat!(env!("OUT_DIR"), "/test.rs"));

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
