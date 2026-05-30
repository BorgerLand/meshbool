use crate::common::{AABB, AABBOverlap};
use crate::utils::atomic_add_i32;
use crate::vec::vec_uninit;
use nalgebra::{Matrix3x4, Point3, Vector3};
use std::any::TypeId;
use std::fmt::Debug;
use std::mem;

// Adjustable parameters
const K_INITIAL_LENGTH: i32 = 128;
const K_LENGTH_MULTIPLE: i32 = 4;
// Fundamental constants
const K_ROOT: i32 = 1;

#[inline]
const fn is_leaf(node: i32) -> bool {
	node % 2 == 0
}
#[inline]
const fn is_internal(node: i32) -> bool {
	node % 2 == 1
}
#[inline]
const fn node2internal(node: i32) -> i32 {
	(node - 1) / 2
}
#[inline]
const fn internal2node(internal: i32) -> i32 {
	internal * 2 + 1
}
#[inline]
const fn node2leaf(node: i32) -> i32 {
	node / 2
}
#[inline]
const fn leaf2node(leaf: i32) -> i32 {
	leaf * 2
}

struct CreateRadixTree<'a> {
	node_parent: &'a mut [i32],
	// even nodes are leaves, odd nodes are internal, root is 1
	internal_children: &'a mut [(i32, i32)],
	leaf_morton: &'a [u32],
}

impl<'a> CreateRadixTree<'a> {
	fn prefix_length_unsigned(&self, a: u32, b: u32) -> i32 {
		(a ^ b).leading_zeros() as i32
	}

	fn prefix_length_signed(&self, i: i32, j: i32) -> i32 {
		if j < 0 || j >= self.leaf_morton.len() as i32 {
			-1
		} else {
			if self.leaf_morton[i as usize] == self.leaf_morton[j as usize] {
				// use index to disambiguate
				32 + self.prefix_length_unsigned(i as u32, j as u32)
			} else {
				self.prefix_length_unsigned(
					self.leaf_morton[i as usize],
					self.leaf_morton[j as usize],
				)
			}
		}
	}

	fn range_end(&self, i: i32) -> i32 {
		// Determine direction of range (+1 or -1)
		let mut dir = self.prefix_length_signed(i, i + 1) - self.prefix_length_signed(i, i - 1);
		dir = ((dir > 0) as i32) - ((dir < 0) as i32);
		// Compute conservative range length with exponential increase
		let common_prefix = self.prefix_length_signed(i, i - dir);
		let mut max_length = K_INITIAL_LENGTH;
		while self.prefix_length_signed(i, i + dir * max_length) > common_prefix {
			max_length *= K_LENGTH_MULTIPLE;
		}

		// Compute precise range length with binary search
		let mut length = 0;
		let mut step = max_length / 2;
		loop {
			if step <= 0 {
				break;
			}

			if self.prefix_length_signed(i, i + dir * (length + step)) > common_prefix {
				length += step;
			}

			step /= 2;
		}

		i + dir * length
	}

	fn find_split(&self, first: i32, last: i32) -> i32 {
		let common_prefix = self.prefix_length_signed(first, last);
		// Find the furthest object that shares more than commonPrefix bits with the
		// first one, using binary search.
		let mut split = first;
		let mut step = last - first;
		loop {
			step = (step + 1) >> 1; // divide by 2, rounding up
			let new_split = split + step;
			if new_split < last {
				let split_prefix = self.prefix_length_signed(first, new_split);
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

	fn call(&mut self, internal: i32) {
		let mut first = internal;
		// Find the range of objects with a common prefix
		let mut last = self.range_end(first);
		if first > last {
			mem::swap(&mut first, &mut last);
		}
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
		self.internal_children[internal as usize] = (child1, child2);
		let node = internal2node(internal);
		self.node_parent[child1 as usize] = node;
		self.node_parent[child2 as usize] = node;
	}
}

struct FindCollision<'a, const SELF_COLLISION: bool, F, AABBOverlapT, RecorderT>
where
	F: Fn(i32) -> AABBOverlapT,
	RecorderT: Recorder,
{
	f: &'a F,
	node_bbox: &'a [AABB],
	internal_children: &'a [(i32, i32)],
	recorder: &'a mut RecorderT,
}

impl<'a, const SELF_COLLISION: bool, F, AABBOverlapT, RecorderT>
	FindCollision<'a, SELF_COLLISION, F, AABBOverlapT, RecorderT>
where
	F: Fn(i32) -> AABBOverlapT,
	AABBOverlapT: Debug + 'static,
	RecorderT: Recorder,
	AABB: AABBOverlap<AABBOverlapT>,
{
	#[inline]
	fn record_collision(&mut self, query: &AABBOverlapT, node: i32, query_idx: i32) -> bool {
		let bbox = self.node_bbox[node as usize];
		let overlaps = bbox.does_overlap(query);
		if overlaps && is_leaf(node) {
			let leaf_idx = node2leaf(node);
			if !SELF_COLLISION || leaf_idx != query_idx {
				self.recorder.record(query_idx, leaf_idx);
			}
		}

		overlaps && is_internal(node) //should traverse into node
	}

	fn call(&mut self, query_idx: i32) {
		let query = (self.f)(query_idx);

		// early exit for empty boxes
		if TypeId::of::<AABBOverlapT>() == TypeId::of::<AABB>() {
			let query: &AABB = unsafe { std::mem::transmute(&query) };
			if query.min.x == f64::INFINITY {
				return;
			}
		}

		// stack cannot overflow because radix tree has max depth 30 (Morton code) +
		// 32 (index).
		let mut stack = [0; 64];
		let mut top = -1;
		// Depth-first search
		let mut node = K_ROOT;
		loop {
			let internal = node2internal(node);
			let child1 = self.internal_children[internal as usize].0;
			let child2 = self.internal_children[internal as usize].1;

			let traverse1 = self.record_collision(&query, child1, query_idx);
			let traverse2 = self.record_collision(&query, child2, query_idx);

			if !traverse1 && !traverse2 {
				if top < 0 {
					break;
				} //done
				node = stack[top as usize];
				top -= 1; //get a saved node
			} else {
				node = if traverse1 { child1 } else { child2 }; //go here next
				if traverse1 && traverse2 {
					top += 1;
					stack[top as usize] = child2; //save the other for later
				}
			}
		}
	}
}

struct BuildInternalBoxes<'a> {
	node_bbox: &'a mut [AABB],
	counter: &'a mut [i32],
	node_parent: &'a [i32],
	internal_children: &'a [(i32, i32)],
}

impl<'a> BuildInternalBoxes<'a> {
	fn call(&mut self, leaf: i32) {
		let mut node = leaf2node(leaf);
		loop {
			node = self.node_parent[node as usize];
			let internal = node2internal(node);
			if unsafe { atomic_add_i32(&mut self.counter[internal as usize], 1) } == 0 {
				return;
			}
			self.node_bbox[node as usize] = self.node_bbox
				[self.internal_children[internal as usize].0 as usize]
				.union_aabb(&self.node_bbox[self.internal_children[internal as usize].1 as usize]);

			if node == K_ROOT {
				break;
			}
		}
	}
}

#[inline]
const fn spread_bits3(mut v: u32) -> u32 {
	v = 0xFF0000FF & (v.wrapping_mul(0x00010001));
	v = 0x0F00F00F & (v.wrapping_mul(0x00000101));
	v = 0xC30C30C3 & (v.wrapping_mul(0x00000011));
	v = 0x49249249 & (v.wrapping_mul(0x00000005));
	return v;
}

pub trait Recorder {
	fn record(&mut self, query_idx: i32, leaf_idx: i32);
}

pub struct SimpleRecorder<'a> {
	pub f: &'a mut dyn FnMut(i32, i32),
}

impl<'a> SimpleRecorder<'a> {
	pub fn new(f: &'a mut dyn FnMut(i32, i32)) -> Self {
		Self { f }
	}
}

impl<'a> Recorder for SimpleRecorder<'a> {
	fn record(&mut self, query_idx: i32, leaf_idx: i32) {
		(self.f)(query_idx, leaf_idx);
	}
}

#[derive(Clone, Default, Debug)]
pub struct Collider {
	node_bbox: Vec<AABB>,
	node_parent: Vec<i32>,
	// even nodes are leaves, odd nodes are internal, root is 1
	internal_children: Vec<(i32, i32)>,
}

impl Collider {
	pub fn new(leaf_bb: &[AABB], leaf_morton: &[u32]) -> Self {
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
			node_bbox: unsafe { vec_uninit(num_nodes) },
			node_parent: vec![-1; num_nodes],
			internal_children: vec![(-1, -1); leaf_bb.len() - 1],
		};

		for internal in 0..collider.num_internal() {
			CreateRadixTree {
				node_parent: &mut collider.node_parent,
				internal_children: &mut collider.internal_children,
				leaf_morton,
			}
			.call(internal as i32);
		}

		collider.update_boxes(leaf_bb);
		collider
	}

	pub fn get_bounding_box(&self) -> AABB {
		if self.node_bbox.is_empty() {
			return AABB::default();
		}
		self.node_bbox[internal2node(0) as usize]
	}

	pub fn update_boxes(&mut self, leaf_bb: &[AABB]) {
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
			.call(leaf as i32)
		}
	}

	pub fn transform(&mut self, transform: Matrix3x4<f64>) {
		debug_assert!(
			Collider::is_axis_aligned(&transform),
			"transform must be axis-aligned"
		);

		for aabb in &mut self.node_bbox {
			*aabb = aabb.transform(transform)
		}
	}

	pub fn collisions_from_fn<const SELF_COLLISION: bool, AABBOverlapT>(
		&self,
		recorder: &mut impl Recorder,
		f: impl Fn(i32) -> AABBOverlapT,
		n: usize,
		_parallel: bool,
	) where
		AABBOverlapT: Debug + 'static,
		AABB: AABBOverlap<AABBOverlapT>,
	{
		if self.internal_children.is_empty() {
			return;
		}
		for query_idx in 0..n {
			FindCollision::<SELF_COLLISION, _, _, _> {
				f: &f,
				node_bbox: &self.node_bbox,
				internal_children: &self.internal_children,
				recorder,
			}
			.call(query_idx as i32);
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
	pub fn collisions_from_slice<const SELF_COLLISION: bool, AABBOverlapT>(
		&self,
		recorder: &mut impl Recorder,
		queries_in: &[AABBOverlapT],
		parallel: bool,
	) where
		AABBOverlapT: Debug + Clone + 'static,
		AABB: AABBOverlap<AABBOverlapT>,
	{
		let f = |i: i32| -> AABBOverlapT { queries_in[i as usize].clone() };
		self.collisions_from_fn::<SELF_COLLISION, _>(recorder, f, queries_in.len(), parallel);
	}

	pub fn morton_code(position: Point3<f64>, bbox: AABB) -> u32 {
		let mut xyz = (position - bbox.min).component_div(&(bbox.max - bbox.min));
		xyz = Vector3::from_element(1023.0).inf(&Vector3::from_element(0.0).sup(&(1024.0 * xyz)));
		let x = spread_bits3(xyz.x as u32);
		let y = spread_bits3(xyz.y as u32);
		let z = spread_bits3(xyz.z as u32);
		x.wrapping_mul(4)
			.wrapping_add(y.wrapping_mul(2))
			.wrapping_add(z)
	}

	fn is_axis_aligned(transform: &Matrix3x4<f64>) -> bool {
		for row in 0..3 {
			let mut count = 0;
			for col in 0..3 {
				if transform[(row, col)] == 0.0 {
					count += 1;
				}
			}

			if count != 2 {
				return false;
			}
		}

		true
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
