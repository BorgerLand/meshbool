use crate::{Box3D, MeshBool, Precision};
use nalgebra::Matrix3x4;

pub mod expr_build;
mod expr_eval;

#[derive(Clone, Debug)]
#[allow(private_interfaces)]
pub enum CSGExpression {
	Leaf(CSGLeaf),
	Difference(Box<CSGDifference>),
	Commutative(CSGCommutative),
}

#[derive(Clone, Debug)]
struct CSGLeaf {
	leaf: MeshBool,
	pending_transform: Matrix3x4<f64>,
}

#[derive(Clone, Debug)]
struct CSGDifference {
	lhs: CSGExpression,
	rhs: CSGExpression,
	approximate_bbox: Option<Box3D>,
}

#[derive(Clone, Debug)]
struct CSGCommutative {
	children: Vec<CSGExpression>,
	op: CommutativeOpType,
	approximate_bbox: Option<Box3D>,
}

impl From<MeshBool> for CSGExpression {
	fn from(value: MeshBool) -> Self {
		Self::Leaf(CSGLeaf {
			leaf: value,
			pending_transform: Matrix3x4::identity(),
		})
	}
}

///Optimizations to avoid running the full boolean are
///applied when an expression consists of multiple
///consecutive, commutative, matching ops. For example
///expressions A + B + C + D, or A ^ B ^ C ^ D can
///potentially be collapsed into something simpler:
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum CommutativeOpType {
	///A vec of unions can be optimized by searching
	///for non-overlapping meshes that can run through
	///boolean_disjoint_union instead of boolean.
	Union,
	///A vec of intersections can be optimized by first
	///checking that every mesh is overlapping. If not,
	///the entire vec collapses to nothing.
	Intersection,
}
