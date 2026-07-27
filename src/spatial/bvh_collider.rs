use crate::spatial::aabb::{Box3D, Overlap};
use crate::util::math::{atomic_add, is_axis_aligned};
use crate::util::vec_ext;
use nalgebra::{Matrix3x4, Point3, Vector3};
use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Debug;

// Adjustable parameters
const K_INITIAL_LENGTH: isize = 128;
const K_LENGTH_MULTIPLE: isize = 4;
// Fundamental constants
const K_ROOT: usize = 1;

#[derive(Clone, Default, Debug)]
pub struct BVHCollider {
	node_bbox: Vec<Box3D>,
	node_parent: Vec<i32>,
	// even nodes are leaves, odd nodes are internal, root is 1
	internal_children: Vec<(i32, i32)>,
}

impl BVHCollider {
	pub fn new(leaf_bb: &[Box3D], leaf_morton: &[u32]) -> Self {
		debug_assert!(
			leaf_bb.len() == leaf_morton.len(),
			"vectors must be the same length"
		);
		if leaf_bb.len() == 0 {
			return Self::default();
		}
		let num_nodes = 2 * leaf_bb.len() - 1;

		// assign and allocate members
		let mut collider = Self {
			node_bbox: unsafe { vec_ext::uninit(num_nodes) },
			node_parent: vec![-1; num_nodes],
			internal_children: vec![(-1, -1); leaf_bb.len() - 1],
		};

		for internal in 0..collider.num_internal() {
			CreateRadixTree {
				node_parent: &mut collider.node_parent,
				internal_children: &mut collider.internal_children,
				leaf_morton,
			}
			.call(internal);
		}

		collider.update_boxes(leaf_bb);
		collider
	}

	pub fn update_boxes(&mut self, leaf_bb: &[Box3D]) {
		debug_assert!(
			leaf_bb.len() == self.num_leaves(),
			"must have the same number of updated boxes as original"
		);

		// copy in leaf node Boxes
		for i in 0..leaf_bb.len() {
			self.node_bbox[i * 2] = leaf_bb[i];
		}

		// create global counters
		let mut counter = vec![0; self.num_internal()];
		// kernel over leaves to save internal Boxes
		for leaf in 0..self.num_leaves() {
			BuildInternalBoxes {
				node_bbox: &mut self.node_bbox,
				counter: &mut counter,
				node_parent: &self.node_parent,
				internal_children: &self.internal_children,
			}
			.call(leaf)
		}
	}

	pub fn transform(&mut self, transform: Matrix3x4<f64>) {
		debug_assert!(is_axis_aligned(transform), "transform must be axis-aligned");

		for aabb in &mut self.node_bbox {
			*aabb = aabb.transform(transform)
		}
	}

	pub fn get_bounding_box(&self) -> Box3D {
		if self.node_bbox.is_empty() {
			return Box3D::default();
		}
		self.node_bbox[internal2node(0)]
	}

	pub fn collisions_from_fn<const SELF_COLLISION: bool, OverlapT>(
		&self,
		recorder: impl FnMut(/*query_idx*/ usize, /*leaf_idx*/ usize),
		f: impl Fn(usize) -> OverlapT,
		n: usize,
		_parallel: bool,
	) where
		OverlapT: Debug + Copy + 'static,
		Box3D: Overlap<OverlapT>,
	{
		if self.internal_children.is_empty() {
			return;
		}
		let mut find_collision = FindCollision::<SELF_COLLISION, _, _, _> {
			f: &f,
			node_bbox: &self.node_bbox,
			internal_children: &self.internal_children,
			recorder,
		};
		for query_idx in 0..n {
			find_collision.call(query_idx);
		}
	}

	///This function iterates over queriesIn and calls recorder.record(queryIdx,
	///leafIdx, local) for each collision it found.
	///If selfCollisionl is true, it will skip the case where queryIdx == leafIdx.
	///The recorder should provide a local() method that returns a Recorder::Local
	///type, representing thread local storage. By default, recorder.record can
	///run in parallel and the thread local storage can be combined at the end.
	///If parallel is false, the function will run in sequential mode.
	///
	///If thread local storage is not needed, use SimpleRecorder.
	pub fn collisions_from_slice<const SELF_COLLISION: bool, OverlapT>(
		&self,
		recorder: impl FnMut(/*query_idx*/ usize, /*leaf_idx*/ usize),
		queries_in: &[OverlapT],
		parallel: bool,
	) where
		OverlapT: Debug + Copy + 'static,
		Box3D: Overlap<OverlapT>,
	{
		let f = |i| queries_in[i];
		self.collisions_from_fn::<SELF_COLLISION, _>(recorder, f, queries_in.len(), parallel);
	}

	pub fn morton_code(position: Point3<f64>, bbox: Box3D) -> u32 {
		let mut xyz = (position - bbox.min).component_div(&(bbox.max - bbox.min));
		xyz = Vector3::from_element(1023.0).inf(&Vector3::from_element(0.0).sup(&(1024.0 * xyz)));
		let x = spread_bits3(xyz.x as u32);
		let y = spread_bits3(xyz.y as u32);
		let z = spread_bits3(xyz.z as u32);
		x.wrapping_mul(4)
			.wrapping_add(y.wrapping_mul(2))
			.wrapping_add(z)
	}

	fn num_internal(&self) -> usize {
		self.internal_children.len()
	}

	fn num_leaves(&self) -> usize {
		if self.internal_children.is_empty() {
			0
		} else {
			self.num_internal() + 1
		}
	}
}

struct CreateRadixTree<'a> {
	node_parent: &'a mut [i32],
	// even nodes are leaves, odd nodes are internal, root is 1
	internal_children: &'a mut [(i32, i32)],
	leaf_morton: &'a [u32],
}

impl<'a> CreateRadixTree<'a> {
	fn prefix_length(&self, a: usize, b: usize) -> usize {
		(a ^ b).leading_zeros() as usize
	}

	fn prefix_length_checked(&self, i: usize, j: Option<usize>) -> Option<usize> {
		match j {
			Some(j) if j < self.leaf_morton.len() => {
				Some(if self.leaf_morton[i] == self.leaf_morton[j] {
					// use index to disambiguate
					32 + self.prefix_length(i, j)
				} else {
					self.prefix_length(self.leaf_morton[i] as usize, self.leaf_morton[j] as usize)
				})
			}
			_ => None,
		}
	}

	fn range_end(&self, i: usize) -> usize {
		// Determine direction of range (+1 or -1)
		let forward = self.prefix_length_checked(i, i.checked_add_signed(1));
		let backward = self.prefix_length_checked(i, i.checked_add_signed(-1));
		let dir = match forward.cmp(&backward) {
			Ordering::Greater => 1,
			Ordering::Less => -1,
			Ordering::Equal => 0,
		};
		// Compute conservative range length with exponential increase
		let common_prefix = self.prefix_length_checked(i, i.checked_add_signed(-dir));
		let mut max_length = K_INITIAL_LENGTH;
		while self.prefix_length_checked(i, i.checked_add_signed(dir * max_length)) > common_prefix
		{
			max_length *= K_LENGTH_MULTIPLE;
		}

		// Compute precise range length with binary search
		let mut length = 0;
		let mut step = max_length / 2;
		loop {
			if step <= 0 {
				break;
			}

			if self.prefix_length_checked(i, i.checked_add_signed(dir * (length + step)))
				> common_prefix
			{
				length += step;
			}

			step /= 2;
		}

		i.checked_add_signed(dir * length).unwrap()
	}

	fn find_split(&self, first: usize, last: usize) -> usize {
		let common_prefix = self.prefix_length_checked(first, Some(last));
		// Find the furthest object that shares more than commonPrefix bits with the
		// first one, using binary search.
		let mut split = first;
		let mut step = last - first;
		loop {
			step = (step + 1) >> 1; // divide by 2, rounding up
			let new_split = split + step;
			if new_split < last {
				let split_prefix = self.prefix_length_checked(first, Some(new_split));
				if split_prefix > common_prefix {
					split = new_split;
				}
			}

			if step <= 1 {
				break;
			}
		}

		split
	}

	fn call(&mut self, internal: usize) {
		// Find the range of objects with a common prefix
		let end = self.range_end(internal);
		let (first, last) = if internal > end {
			(end, internal)
		} else {
			(internal, end)
		};
		// Determine where the next-highest difference occurs
		let mut split = self.find_split(first, last);
		let child1 = if split == first {
			leaf2node(split)
		} else {
			internal2node(split)
		};
		split += 1;
		let child2 = if split == last {
			leaf2node(split)
		} else {
			internal2node(split)
		};
		// Record parent_child relationships.
		self.internal_children[internal] = (child1 as i32, child2 as i32);
		let node = internal2node(internal);
		self.node_parent[child1] = node as i32;
		self.node_parent[child2] = node as i32;
	}
}

struct FindCollision<'a, const SELF_COLLISION: bool, F, OverlapT, RecorderT>
where
	F: Fn(usize) -> OverlapT,
	RecorderT: FnMut(/*query_idx*/ usize, /*leaf_idx*/ usize),
{
	f: &'a F,
	node_bbox: &'a [Box3D],
	internal_children: &'a [(i32, i32)],
	recorder: RecorderT,
}

impl<'a, const SELF_COLLISION: bool, F, OverlapT, RecorderT>
	FindCollision<'a, SELF_COLLISION, F, OverlapT, RecorderT>
where
	F: Fn(usize) -> OverlapT,
	OverlapT: Copy + Debug + 'static,
	RecorderT: FnMut(/*query_idx*/ usize, /*leaf_idx*/ usize),
	Box3D: Overlap<OverlapT>,
{
	#[inline(always)]
	fn record_collision(&mut self, query: OverlapT, node: usize, query_idx: usize) -> bool {
		let bbox = self.node_bbox[node];
		let overlaps = bbox.does_overlap(query);
		if overlaps && is_leaf(node) {
			let leaf_idx = node2leaf(node);
			if !SELF_COLLISION || leaf_idx != query_idx {
				(self.recorder)(query_idx, leaf_idx);
			}
		}

		overlaps && is_internal(node) //should traverse into node
	}

	fn call(&mut self, query_idx: usize) {
		let query = (self.f)(query_idx);

		// early exit for empty boxes
		if let Some(query) = (&query as &dyn Any).downcast_ref::<Box3D>() {
			if query.min.x == f64::INFINITY {
				return;
			}
		}

		// stack cannot overflow because radix tree has max depth 30 (Morton code) +
		// 32 (index).
		let mut stack = [0; 64];
		let mut top = 0;
		// Depth-first search
		let mut node = K_ROOT;
		loop {
			let internal = node2internal(node);
			let child1 = self.internal_children[internal].0 as usize;
			let child2 = self.internal_children[internal].1 as usize;

			let traverse1 = self.record_collision(query, child1, query_idx);
			let traverse2 = self.record_collision(query, child2, query_idx);

			if !traverse1 && !traverse2 {
				if top == 0 {
					break;
				} //done
				top -= 1;
				node = stack[top]; //get a saved node
			} else {
				node = if traverse1 { child1 } else { child2 }; //go here next
				if traverse1 && traverse2 {
					stack[top] = child2; //save the other for later
					top += 1;
				}
			}
		}
	}
}

struct BuildInternalBoxes<'a> {
	node_bbox: &'a mut [Box3D],
	counter: &'a mut [i32],
	node_parent: &'a [i32],
	internal_children: &'a [(i32, i32)],
}

impl<'a> BuildInternalBoxes<'a> {
	fn call(&mut self, leaf: usize) {
		let mut node = leaf2node(leaf);
		loop {
			node = self.node_parent[node] as usize;
			let internal = node2internal(node);
			if atomic_add(&mut self.counter[internal], 1) == 0 {
				return;
			}
			self.node_bbox[node] = self.node_bbox[self.internal_children[internal].0 as usize]
				.union_box3(self.node_bbox[self.internal_children[internal].1 as usize]);

			if node == K_ROOT {
				break;
			}
		}
	}
}

#[inline(always)]
const fn spread_bits3(mut v: u32) -> u32 {
	v = 0xFF0000FF & (v.wrapping_mul(0x00010001));
	v = 0x0F00F00F & (v.wrapping_mul(0x00000101));
	v = 0xC30C30C3 & (v.wrapping_mul(0x00000011));
	v = 0x49249249 & (v.wrapping_mul(0x00000005));
	return v;
}

#[inline(always)]
const fn is_leaf(node: usize) -> bool {
	node % 2 == 0
}
#[inline(always)]
const fn is_internal(node: usize) -> bool {
	node % 2 == 1
}
#[inline(always)]
const fn node2internal(node: usize) -> usize {
	(node - 1) / 2
}
#[inline(always)]
const fn internal2node(internal: usize) -> usize {
	internal * 2 + 1
}
#[inline(always)]
const fn node2leaf(node: usize) -> usize {
	node / 2
}
#[inline(always)]
const fn leaf2node(leaf: usize) -> usize {
	leaf * 2
}
