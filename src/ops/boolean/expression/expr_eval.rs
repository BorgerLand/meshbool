use crate::BooleanError;
use crate::mesh_relations::InstanceRelation;
use crate::ops::boolean::expression::disjoint_union::boolean_disjoint_union;
use crate::ops::boolean::expression::*;
use crate::ops::boolean::{OpType, boolean};
use crate::spatial::aabb::Overlap;
use crate::util::math::mul_mat3x4;
use reblessive::{Stack, Stk};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::iter;
use std::rc::Rc;

impl CSGExpression {
	pub fn eval(self) -> Result<MeshBool, BooleanError> {
		//reblessive allows writing recursive algorithms without
		//running stack overflow risk. necessitates async syntax
		Stack::new().enter(|ctx| self.eval_impl(ctx)).finish()
	}

	async fn eval_impl(mut self, ctx: &mut Stk) -> Result<MeshBool, BooleanError> {
		if self.approximate_bbox(ctx).await.is_empty() {
			return Ok(self.decimated(ctx).await.into());
		}

		match self {
			CSGExpression::Leaf(expr) => Ok(expr.eval()),
			CSGExpression::Difference(expr) => ctx.run(|ctx| expr.eval(ctx)).await,
			CSGExpression::Commutative(expr) => {
				//boolean_lazy should never group fewer than 2
				debug_assert!(expr.children.len() >= 2);

				match expr.op {
					CommutativeOpType::Union => ctx.run(|ctx| expr.eval_union(ctx)).await,
					CommutativeOpType::Intersection => {
						ctx.run(|ctx| expr.eval_intersection(ctx)).await
					}
				}
			}
		}
	}

	async fn approximate_bbox(&mut self, ctx: &mut Stk) -> Box3D {
		match self {
			CSGExpression::Leaf(expr) => expr.approximate_bbox(),
			CSGExpression::Difference(expr) => ctx.run(|ctx| expr.approximate_bbox(ctx)).await,
			CSGExpression::Commutative(expr) => ctx.run(|ctx| expr.approximate_bbox(ctx)).await,
		}
	}

	//collapses entire tree through MeshBool::decimated
	async fn decimated(self, ctx: &mut Stk) -> Decimated {
		match self {
			CSGExpression::Leaf(expr) => expr.decimated(),
			CSGExpression::Difference(expr) => ctx.run(|ctx| expr.decimated(ctx)).await,
			CSGExpression::Commutative(expr) => ctx.run(|ctx| expr.decimated(ctx)).await,
		}
	}
}

impl CSGLeaf {
	fn eval(self) -> MeshBool {
		self.leaf.apply_transform(self.pending_transform)
	}

	pub fn approximate_bbox(&self) -> Box3D {
		self.leaf.bounding_box().transform(self.pending_transform)
	}

	fn decimated(self) -> Decimated {
		Decimated {
			instance_relation: if self.pending_transform == Matrix3x4::identity() {
				self.leaf.instance_relation
			} else {
				let mut instance_rel = Rc::unwrap_or_clone(self.leaf.instance_relation);
				for rel in instance_rel.iter_mut() {
					rel.transform = mul_mat3x4(self.pending_transform, rel.transform)
				}

				Rc::new(instance_rel)
			},
			prop_stride: self.leaf.properties.stride,
			precision: self.leaf.precision,
		}
	}
}

impl CSGDifference {
	async fn eval(mut self, ctx: &mut Stk) -> Result<MeshBool, BooleanError> {
		if self.lhs.approximate_bbox(ctx).await.is_empty() {
			return Ok(self.decimated(ctx).await.into());
		}
		if self.rhs.approximate_bbox(ctx).await.is_empty() {
			return self.lhs.eval_impl(ctx).await;
		}

		let lhs = self.lhs.eval_impl(ctx).await?;
		if lhs.is_empty() {
			//avoid evaling rhs
			Ok(CSGDifference {
				lhs: lhs.into(),
				rhs: self.rhs,
				approximate_bbox: None,
			}
			.decimated(ctx)
			.await
			.into())
		} else {
			let rhs = self.rhs.eval_impl(ctx).await?;
			boolean(lhs, OpType::Difference, rhs)
		}
	}

	async fn approximate_bbox(&mut self, ctx: &mut Stk) -> Box3D {
		if self.approximate_bbox.is_none() {
			//because rhs is an oversized approxmiation, attempting to
			//subtract it would potentially shrink the result too much
			self.approximate_bbox = Some(self.lhs.approximate_bbox(ctx).await);
		}

		self.approximate_bbox.unwrap()
	}

	async fn decimated(self, ctx: &mut Stk) -> Decimated {
		let lhs = self.lhs.decimated(ctx).await;
		let rhs = self.rhs.decimated(ctx).await;

		let mut instance_relation = lhs.instance_relation.to_vec();
		instance_relation.extend(rhs.instance_relation.iter().map(|rel| {
			let mut rel = *rel;
			rel.back_side = !rel.back_side;
			rel
		}));

		Decimated {
			instance_relation: Rc::new(instance_relation),
			prop_stride: lhs.prop_stride.max(rhs.prop_stride),
			precision: Precision {
				epsilon: lhs.precision.epsilon.max(rhs.precision.epsilon),
				tolerance: lhs.precision.tolerance.max(rhs.precision.tolerance),
			},
		}
	}
}

impl CSGCommutative {
	async fn eval_union(self, ctx: &mut Stk) -> Result<MeshBool, BooleanError> {
		let mut children = Vec::with_capacity(self.children.len());
		let mut decimated = Vec::with_capacity(self.children.len());
		for mut child in self.children {
			if child.approximate_bbox(ctx).await.is_empty() {
				decimated.push(child);
				continue;
			}

			let child = match child {
				CSGExpression::Leaf(expr) => expr,
				expr => CSGLeaf {
					leaf: expr.eval_impl(ctx).await?,
					pending_transform: Matrix3x4::identity(),
				},
			};

			if child.leaf.is_empty() {
				decimated.push(CSGExpression::Leaf(child));
			} else {
				children.push(child);
			}
		}

		//this will be unioned into the result at the end,
		//purely for correctness in instance_relation
		let decimated = CSGCommutative {
			children: decimated,
			op: CommutativeOpType::Union,
			approximate_bbox: None,
		}
		.decimated(ctx)
		.await
		.into();

		if children.len() == 0 {
			return Ok(decimated);
		}

		Ok(boolean(
			if children.len() == 1 {
				children.into_iter().next().unwrap().eval()
			} else if children.len() == 2 {
				let mut children = children.into_iter();
				let lhs = children.next().unwrap().eval();
				let rhs = children.next().unwrap().eval();
				boolean(lhs, OpType::Union, rhs)?
			} else {
				//in an ideal world this is a boxicity-3 mwis-like problem:
				//what is the maximum number of triangles in the children
				//array that can be pushed through boolean_disjoint_union
				//(potentially multiple calls to it)? unfortunately it is
				//ludicrously expensive to compute for more than a handful of
				//boxes, so a naive first fit O(n^2) heuristic is used instead
				//https://en.wikipedia.org/wiki/Boxicity#Algorithmic_results

				//sort most to least complex meshes, hoping that looping in this
				//order pulls out the most complex meshes first, approximating mwis
				children.sort_unstable_by_key(|child| Reverse(child.leaf.num_tri()));

				//bounding_box() is a lookup into the bvh collider's internal
				//array. to avoid cache miss per comparison, copy it into a tuple
				let mut children = VecDeque::from_iter(children.into_iter().map(|child| {
					let bbox = child.approximate_bbox();
					(child, bbox)
				}));

				let mut cur_disjoint_union_mesh = Vec::with_capacity(children.len());
				let mut cur_disjoint_union_bbox = Vec::with_capacity(children.len());
				let mut disjoint_unions = BinaryHeap::with_capacity(children.len());

				while children.len() > 0 {
					let cur_disjoint_union_first = children.pop_front().unwrap();
					cur_disjoint_union_mesh.push(cur_disjoint_union_first.0);
					cur_disjoint_union_bbox.push(cur_disjoint_union_first.1);

					let mut test_candidate_i = 0;
					while test_candidate_i < children.len() {
						let test_candidate = children[test_candidate_i].1;
						if cur_disjoint_union_bbox
							.iter()
							.all(|&test_member| !test_candidate.overlaps(test_member))
						{
							let cur_disjoint_union_next =
								children.remove(test_candidate_i).unwrap();
							cur_disjoint_union_mesh.push(cur_disjoint_union_next.0);
							cur_disjoint_union_bbox.push(cur_disjoint_union_next.1);
						} else {
							test_candidate_i += 1;
						}
					}

					cur_disjoint_union_bbox.clear();
					disjoint_unions.push(SortByNumTri(if cur_disjoint_union_mesh.len() == 1 {
						cur_disjoint_union_mesh.pop().unwrap().eval()
					} else {
						boolean_disjoint_union(cur_disjoint_union_mesh.drain(..))?
					}));
				}

				drop(cur_disjoint_union_mesh);
				drop(cur_disjoint_union_bbox);
				drop(children);

				// apply boolean operations starting from smaller meshes
				// the assumption is that boolean operations on smaller meshes is faster,
				// due to less data being copied and processed
				while disjoint_unions.len() >= 2 {
					let lhs = disjoint_unions.pop().unwrap().0;
					let rhs = disjoint_unions.pop().unwrap().0;
					disjoint_unions.push(SortByNumTri(boolean(lhs, OpType::Union, rhs)?));
				}

				disjoint_unions.into_iter().next().unwrap().0
			},
			OpType::Union,
			decimated,
		)
		.unwrap()) //this should not fail. decimated has 0 vertices
	}

	async fn eval_intersection(mut self, ctx: &mut Stk) -> Result<MeshBool, BooleanError> {
		let mut bbox = self.approximate_bbox(ctx).await;
		let mut sorted_children = BinaryHeap::<SortByNumTri>::with_capacity(self.children.len());
		let mut children = self.children.into_iter();

		while let Some(child) = children.next() {
			let child = child.eval_impl(ctx).await?;

			bbox = bbox.intersection_box3(child.bounding_box());
			if bbox.is_empty() {
				return Ok(CSGCommutative {
					children: iter::once(child.into())
						.chain(children)
						.chain(sorted_children.into_iter().map(|child| child.0.into()))
						.collect(),
					op: CommutativeOpType::Intersection,
					approximate_bbox: None,
				}
				.decimated(ctx)
				.await
				.into());
			}

			sorted_children.push(SortByNumTri(child));
		}

		drop(children);

		// apply boolean operations starting from smaller meshes
		// the assumption is that boolean operations on smaller meshes is faster,
		// due to less data being copied and processed
		while sorted_children.len() >= 2 {
			let lhs = sorted_children.pop().unwrap().0;
			let rhs = sorted_children.pop().unwrap().0;
			let out_r = boolean(lhs, OpType::Intersection, rhs)?;

			bbox = bbox.intersection_box3(out_r.bounding_box());
			if bbox.is_empty() {
				return Ok(CSGCommutative {
					children: iter::once(out_r.into())
						.chain(sorted_children.into_iter().map(|child| child.0.into()))
						.collect(),
					op: CommutativeOpType::Intersection,
					approximate_bbox: None,
				}
				.decimated(ctx)
				.await
				.into());
			}

			sorted_children.push(SortByNumTri(out_r));
		}

		Ok(sorted_children.into_iter().next().unwrap().0)
	}

	async fn approximate_bbox(&mut self, ctx: &mut Stk) -> Box3D {
		if self.approximate_bbox.is_none() {
			let mut bbox = match self.op {
				CommutativeOpType::Union => Box3D::empty(),
				CommutativeOpType::Intersection => Box3D::infinite(),
			};

			for child in self.children.iter_mut() {
				let child_bbox = child.approximate_bbox(ctx).await;
				match self.op {
					CommutativeOpType::Union => bbox = bbox.union_box3(child_bbox),
					CommutativeOpType::Intersection => {
						bbox = bbox.intersection_box3(child_bbox);
						if bbox.is_empty() {
							break;
						}
					}
				}
			}

			self.approximate_bbox = Some(bbox);
		}

		self.approximate_bbox.unwrap()
	}

	async fn decimated(self, ctx: &mut Stk) -> Decimated {
		let mut instance_rel = Vec::new();
		let mut prop_stride = 0;
		let mut precision = Precision {
			epsilon: -1.0,
			tolerance: -1.0,
		};

		for child in self.children.into_iter() {
			let child = child.decimated(ctx).await;
			instance_rel.extend(child.instance_relation.iter());
			prop_stride = prop_stride.max(child.prop_stride);
			precision.epsilon = precision.epsilon.max(child.precision.epsilon);
			precision.tolerance = precision.tolerance.max(child.precision.tolerance);
		}

		Decimated {
			instance_relation: Rc::new(instance_rel),
			prop_stride,
			precision,
		}
	}
}

struct Decimated {
	instance_relation: Rc<Vec<InstanceRelation>>,
	prop_stride: usize,
	precision: Precision,
}

impl From<Decimated> for MeshBool {
	fn from(value: Decimated) -> Self {
		MeshBool::decimated(
			None,
			value.instance_relation,
			value.prop_stride,
			value.precision,
		)
	}
}

struct SortByNumTri(MeshBool);

impl Ord for SortByNumTri {
	fn cmp(&self, other: &Self) -> Ordering {
		//heap defaults to reverse/descending order so reverse it again
		self.0.num_tri().cmp(&other.0.num_tri()).reverse()
	}
}

impl PartialOrd for SortByNumTri {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl PartialEq for SortByNumTri {
	fn eq(&self, other: &Self) -> bool {
		self.0.num_tri() == other.0.num_tri()
	}
}

impl Eq for SortByNumTri {}
