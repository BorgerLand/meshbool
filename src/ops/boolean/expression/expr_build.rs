use crate::ops::boolean::OpType;
use crate::ops::boolean::expression::*;
use crate::util::math::{cosd, mul_mat3x4, sind};
use nalgebra::{Matrix3, Vector3};
use reblessive::{Stack, Stk};
use std::mem;
use std::ops::{Add, AddAssign, BitXor, BitXorAssign, Sub, SubAssign};

impl From<CommutativeOpType> for OpType {
	fn from(value: CommutativeOpType) -> Self {
		match value {
			CommutativeOpType::Union => Self::Union,
			CommutativeOpType::Intersection => Self::Intersection,
		}
	}
}

macro_rules! csg_op {
	($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:expr) => {
		// expr op expr (the base case everything else forwards to)
		impl $trait for CSGExpression {
			type Output = Self;
			fn $method(self, rhs: Self) -> Self::Output {
				boolean_lazy(self, $op, rhs)
			}
		}

		// expr op mesh
		impl $trait<MeshBool> for CSGExpression {
			type Output = Self;
			fn $method(self, rhs: MeshBool) -> Self::Output {
				self.$method(Self::from(rhs))
			}
		}

		// mesh op expr
		impl $trait<CSGExpression> for MeshBool {
			type Output = CSGExpression;
			fn $method(self, rhs: CSGExpression) -> Self::Output {
				CSGExpression::from(self).$method(rhs)
			}
		}

		// mesh op mesh
		impl $trait for MeshBool {
			type Output = CSGExpression;
			fn $method(self, rhs: Self) -> Self::Output {
				CSGExpression::from(self).$method(CSGExpression::from(rhs))
			}
		}

		// expr op= expr
		impl $assign_trait for CSGExpression {
			fn $assign_method(&mut self, rhs: Self) {
				*self = mem::replace(self, Self::temporary_dud()).$method(rhs);
			}
		}

		// expr op= mesh
		impl $assign_trait<MeshBool> for CSGExpression {
			fn $assign_method(&mut self, rhs: MeshBool) {
				self.$assign_method(CSGExpression::from(rhs));
			}
		}
	};
}

csg_op!(Add, add, AddAssign, add_assign, OpType::Union);
csg_op!(Sub, sub, SubAssign, sub_assign, OpType::Difference);
csg_op!(
	BitXor,
	bitxor,
	BitXorAssign,
	bitxor_assign,
	OpType::Intersection
);

fn boolean_lazy(mut in_p: CSGExpression, op: OpType, mut in_q: CSGExpression) -> CSGExpression {
	//goal is to balance the tree in a way that maximizes the
	//commutative children vec length and minimizes depth using
	//algebra of sets identities
	//https://en.wikipedia.org/wiki/Algebra_of_sets

	let op = match op {
		OpType::Union => CommutativeOpType::Union,
		OpType::Intersection => CommutativeOpType::Intersection,
		OpType::Difference => {
			return CSGExpression::Difference(if let CSGExpression::Difference(mut in_p) = in_p {
				//case 1: if in_p is also a difference, union its rhs + in_q.
				//(use case: repeatedly subtracting from something)
				//(A - B) - C = A - (B + C)
				//2 children, tree depth 2
				//(A - (B + C)) - D = A - (B + C + D)
				//3 children, tree depth still 2
				in_p.rhs += in_q;
				in_p
			} else {
				//case 2: failed to rebalance any further
				Box::new(CSGDifference {
					lhs: in_p,
					rhs: in_q,
					approximate_bbox: None,
				})
			});
		}
	};

	//de morgan allows factoring subtrahends (rhs) out of
	//one or both operands of an intersection, which unlocks
	//more case 1 rewrites
	//https://en.wikipedia.org/wiki/De_Morgan%27s_laws
	if op == CommutativeOpType::Intersection {
		let factored_p_subtrahend = match in_p {
			CSGExpression::Difference(expr) => {
				in_p = expr.lhs;
				Some(expr.rhs)
			}
			expr => {
				in_p = expr; //no-op
				None
			}
		};

		let factored_q_subtrahend = match in_q {
			CSGExpression::Difference(expr) => {
				in_q = expr.lhs;
				Some(expr.rhs)
			}
			expr => {
				in_q = expr; //no-op
				None
			}
		};

		match (factored_p_subtrahend, factored_q_subtrahend) {
			//case 3: factored from both operands
			//((A - B) ^ (C - D)) - (E + F) =
			//((A ^ C) - (B + D)) - (E + F) =
			// (A ^ C) - (B + D + E + F)
			//children 2 depth 3 -> depth 2 children 4
			(Some(factored_p_subtrahend), Some(factored_q_subtrahend)) => {
				return (in_p ^ in_q) - (factored_p_subtrahend + factored_q_subtrahend);
			}

			//case 4: factored from 1 operand
			//((A - B) ^ C) - (E + F) =
			//((A ^ C) - B) - (E + F) =
			// (A ^ C) - (B + E + F)
			//children 2 depth 3 -> children 3 depth 2
			(Some(sub), None) | (None, Some(sub)) => {
				return (in_p ^ in_q) - sub;
			}

			//de morgan is n/a; continue to commutative children
			_ => {}
		};
	}

	let into_matching_children = |expr| match expr {
		CSGExpression::Commutative(expr) if expr.op == op => Ok(expr.children),
		other => Err(other),
	};

	//tricks to increase len of commutative children vec
	CSGExpression::Commutative(CSGCommutative {
		op,
		approximate_bbox: None,
		children: match (into_matching_children(in_p), into_matching_children(in_q)) {
			//case 5: in_p, in_q, and op are all the same commutative op.
			//merge everything into a single flat concatenated vec
			//(A + B) + (C + D) = A + B + C + D
			//4 children, tree depth 1
			(Ok(mut in_p_children), Ok(in_q_children)) => {
				in_p_children.extend(in_q_children);
				in_p_children
			}

			//case 6: one of in_p or in_q is commutative and has the same
			//op as the op argument. append the non-matching operand to the
			//other operand that does match
			//(A + B) + (C - D) = A + B + (C - D)
			//3 children, tree depth 2
			(Ok(mut children), Err(child)) | (Err(child), Ok(mut children)) => {
				children.push(child);
				children
			}

			//case 7: unable to merge p/q; build new commutative expression
			//(A - B) + (C - D) = ((A - B) + (C - D))
			//2 children, tree depth 2
			(Err(p), Err(q)) => vec![p, q],
		},
	})
}

impl CSGExpression {
	///Move this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to add to every vertex.
	pub fn translate(self, v: Vector3<f64>) -> Self {
		let mut transform = Matrix3x4::<f64>::identity();
		*transform.column_mut(3) = *v;
		self.transform(transform)
	}

	///Scale this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to multiply every vertex by per component.
	pub fn scale(self, v: Vector3<f64>) -> Self {
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
	pub fn rotate(self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> CSGExpression {
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

	///Mirror this Manifold over the plane described by the unit form of the given
	///normal vector. If the length of the normal is zero, an empty Manifold is
	///returned. This operation can be chained. Transforms are combined and applied
	///lazily.
	///
	///@param normal The normal vector of the plane to be mirrored over
	pub fn mirror(self, normal: Vector3<f64>) -> Self {
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

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(mut self, transform: Matrix3x4<f64>) -> Self {
		//recursively multiples down through to each leaf's
		//pending_transform, but does not actually apply it via
		//transform_impl. a pending_transform on every tree node
		//is not worth the hassle of maintaining a scene graph,
		//and gets in the way of boolean of sets algebra. a few
		//wasted matrix multiplications are cheap compared to
		//full boolean

		//reblessive allows writing recursive algorithms without
		//running stack overflow risk. necessitates async syntax
		Stack::new()
			.enter(|ctx| self.transform_impl(transform, ctx))
			.finish();

		self
	}

	async fn transform_impl(&mut self, transform: Matrix3x4<f64>, ctx: &mut Stk) {
		match self {
			CSGExpression::Leaf(expr) => {
				expr.pending_transform = mul_mat3x4(transform, expr.pending_transform);
			}
			CSGExpression::Difference(expr) => {
				ctx.run(|ctx| expr.lhs.transform_impl(transform, ctx)).await;
				ctx.run(|ctx| expr.rhs.transform_impl(transform, ctx)).await;
			}
			CSGExpression::Commutative(expr) => {
				for child in expr.children.iter_mut() {
					ctx.run(|ctx| child.transform_impl(transform, ctx)).await;
				}
			}
		};
	}
}
