use crate::MeshBool;
use crate::ops::boolean::pipeline::boolean;

mod construct;
pub mod face2tri;
mod intersect;
mod pipeline;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BooleanError {
	ResultTooLarge,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum OpType {
	Union,
	Difference,
	Intersect,
}

impl MeshBool {
	pub fn union(&self, other: &Self) -> Result<Self, BooleanError> {
		boolean(self, OpType::Union, other)
	}

	pub fn difference(&self, other: &Self) -> Result<Self, BooleanError> {
		boolean(self, OpType::Difference, other)
	}

	pub fn intersect(&self, other: &Self) -> Result<Self, BooleanError> {
		boolean(self, OpType::Intersect, other)
	}
}
