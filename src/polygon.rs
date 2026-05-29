use crate::common::Polygons;
use crate::common::{OrderedF64, Rect};
use crate::polygon_internal::HalfedgeTriangulation;
use crate::tree2d::{build_2d_tree, query_2d_tree};
use crate::utils::{K_PRECISION, ccw};
use crate::vec::InsertSorted;
use nalgebra::{Matrix2, Point2, Vector2, Vector3};
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::ops::Range;
use std::{collections::BTreeMap, ptr};

const K_BEST: f64 = f64::NEG_INFINITY;

///Polygon vertex.
#[derive(Debug)]
pub struct PolyVert {
	/// X-Y position
	pub pos: Point2<f64>,
	/// ID or index into another vertex vector
	pub idx: i32,
}

#[allow(unused)]
impl PolyVert {
	pub fn new(pos: Point2<f64>, idx: i32) -> Self {
		Self { pos, idx }
	}
}

pub type SimplePolygonIdx = Vec<PolyVert>;
pub type PolygonsIdx = Vec<SimplePolygonIdx>;

///Tests if the input polygons are convex by searching for any reflex vertices.
///Exactly colinear edges and zero-length edges are treated conservatively as
///reflex. Does not check for overlaps.
fn is_convex(polys: &PolygonsIdx, epsilon: f64) -> bool {
	for poly in polys {
		let first_edge = poly[0].pos - poly.last().unwrap().pos;
		// Zero-length edges comes out NaN, which won't trip the early return, but
		// it's okay because that zero-length edge will also get tested
		// non-normalized and will trip det == 0.
		let mut last_edge = first_edge.normalize();
		for v in 0..poly.len() {
			let edge = if v + 1 < poly.len() {
				poly[v + 1].pos - poly[v].pos
			} else {
				first_edge
			};

			let det = last_edge.perp(&edge);
			if det <= 0.0 || (det.abs() < epsilon && last_edge.dot(&edge) < 0.0) {
				return false;
			}

			last_edge = edge.normalize();
		}
	}

	true
}

///Triangulates a set of convex polygons by alternating instead of a fan, to
///avoid creating high-degree vertices.
fn triangulate_convex(polys: &PolygonsIdx) -> HalfedgeTriangulation {
	let num_tri = polys.iter().fold(0, |acc, poly| acc + poly.len() - 2);
	let mut result = HalfedgeTriangulation::default();
	result.add_contours(polys);
	result.reserve_triangles(num_tri);
	for poly in polys {
		let mut i = 0;
		let mut k = poly.len() - 1;
		let mut right = true;
		while i + 1 < k {
			let j = if right { i + 1 } else { k - 1 };
			result.add_triangle(poly[i].idx, poly[j].idx, poly[k].idx);
			if right {
				i = j;
			} else {
				k = j;
			}

			right = !right;
		}
	}

	result
}

///Ear-clipping triangulator based on David Eberly's approach from Geometric
///Tools, but adjusted to handle epsilon-valid polygons, and including a
///fallback that ensures a manifold triangulation even for overlapping polygons.
///This is reduced from an O(n^2) algorithm by means of our BVH Collider.
///
///The main adjustments for robustness involve clipping the sharpest ears first
///(a known technique to get higher triangle quality), and doing an exhaustive
///search to determine ear convexity exactly if the first geometric result is
///within epsilon.
struct EarClip {
	///The flat list where all the Verts are stored. Not used much for traversal.
	polygon: Vec<Vert>,
	//Pointers to first and last verts within self.polygon
	polygon_range: Range<usize>,
	///The set of right-most starting points, one for each negative-area contour.
	//originally a c++ multiset
	holes: Vec<usize>,
	///The set of starting points, one for each positive-area contour.
	outers: Vec<usize>,
	///The set of starting points, one for each simple polygon.
	simples: Vec<usize>,
	///Maps each hole (by way of starting point) to its bounding box.
	hole2bbox: BTreeMap<usize, Rect>,
	///The output triangulation, represented directly as halfedges.
	result: HalfedgeTriangulation,
	///Bounding box of the entire set of polygons
	bbox: Rect,
	///Working epsilon: max of float error and input value.
	epsilon: f64,
}

impl EarClip {
	fn new(polys: &PolygonsIdx, epsilon: f64) -> EarClip {
		let mut num_vert = 0;
		for poly in polys {
			num_vert += poly.len();
		}

		let polygon: Vec<Vert> = Vec::with_capacity(num_vert + 2 * polys.len()); //must never reallocate or else all vert.left and vert.right break
		let polygon_first = polygon.as_ptr() as usize;
		let polygon_end = unsafe { polygon.as_ptr().add(polygon.capacity()) } as usize;
		let polygon_range = polygon_first..polygon_end;

		let mut ret = EarClip {
			polygon,
			polygon_range,
			holes: Vec::new(),
			outers: Vec::new(),
			simples: Vec::new(),
			hole2bbox: BTreeMap::new(),
			result: HalfedgeTriangulation::default(),
			bbox: Rect::default(),
			epsilon,
		};

		let starts = ret.initialize(polys);

		for v in 0..ret.polygon.len() {
			Self::clip_if_degenerate(
				v,
				&mut ret.polygon,
				&ret.polygon_range,
				&mut ret.result,
				ret.epsilon,
			);
		}

		for first in starts {
			ret.find_start(first);
		}

		ret
	}

	///@brief Triangulates a set of &epsilon;-valid polygons. If the input is not
	///&epsilon;-valid, the triangulation may overlap, but will always return a
	///manifold result that matches the input edge directions.
	///
	///@param polygons The set of polygons, wound CCW and representing multiple
	///polygons and/or holes.
	///@param epsilon The value of &epsilon;, bounding the uncertainty of the
	///input.
	///@param allowConvex If true (default), the triangulator will use a fast
	///triangulation if the input is convex, falling back to ear-clipping if not.
	///The triangle quality may be lower, so set to false to disable this
	///optimization.
	///@return std::vector<ivec3> The triangles, referencing the original
	///polygon points in order.
	fn triangulate(&mut self) -> HalfedgeTriangulation {
		for &start in &self.holes {
			Self::cut_keyhole(
				start,
				&mut self.simples,
				&mut self.result,
				&mut self.polygon,
				&self.polygon_range,
				&self.outers,
				&self.hole2bbox,
				self.epsilon,
			);
		}

		for &start in &self.simples {
			Self::triangulate_poly(
				start,
				&mut self.polygon,
				&self.polygon_range,
				&mut self.result,
				self.epsilon,
			);
		}

		self.result.clone()
	}

	fn get_precision(&self) -> f64 {
		self.epsilon
	}

	fn safe_normalize(v: Vector2<f64>) -> Vector2<f64> {
		let n = v.normalize();
		if n.x.is_finite() {
			n
		} else {
			Vector2::new(0.0, 0.0)
		}
	}

	///This function and JoinPolygons are the only functions that affect the
	///circular list data structure. This helps ensure it remains circular.
	fn link(left: usize, right: usize, polygon: &mut Vec<Vert>, polygon_range: &Range<usize>) {
		polygon[left].right = Vert::index2ptr(right, polygon_range);
		polygon[right].left = Vert::index2ptr(left, polygon_range);
		polygon[left].right_dir = Self::safe_normalize(polygon[right].pos - polygon[left].pos);
	}

	///When an ear vert is clipped, its neighbors get linked, so they get unlinked
	///from it, but it is still linked to them.
	fn clipped(v: &Vert) -> bool {
		!ptr::eq(v.right().left, v)
	}

	fn loop_verts(
		mut first: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		mut func: impl FnMut(usize, &mut Vec<Vert>),
	) -> Option<usize> {
		let mut v = first;
		loop {
			let mut ref_v = &polygon[v];
			if Self::clipped(ref_v) {
				// Update first to an un-clipped vert so we will return to it instead
				// of infinite-looping.
				let new_first = ref_v.right().left();
				first = new_first.ptr2index(polygon_range);
				if !Self::clipped(new_first) {
					v = first;
					ref_v = &polygon[v];
					if ref_v.right == ref_v.left {
						return None;
					}

					func(v, polygon);
				}
			} else {
				if ref_v.right == ref_v.left {
					return None;
				}

				func(v, polygon);
			}

			v = polygon[v].right().ptr2index(polygon_range);
			if v == first {
				break;
			}
		}

		Some(v)
	}

	fn clip_ear(
		ear: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		result: &mut HalfedgeTriangulation,
	) {
		let ear_ref = &polygon[ear];
		let left_i = ear_ref.left().ptr2index(polygon_range);
		let right_i = ear_ref.right().ptr2index(polygon_range);
		Self::link(left_i, right_i, polygon, polygon_range);

		let ear_ref = &polygon[ear];
		let self_mesh = ear_ref.mesh_idx;
		let left_mesh = ear_ref.left().mesh_idx;
		let right_mesh = ear_ref.right().mesh_idx;

		if left_mesh != self_mesh && self_mesh != right_mesh && right_mesh != left_mesh {
			// Filter out topological degenerates, which can form in bad
			// triangulations of polygons with holes, due to vert duplication.
			result.add_triangle(left_mesh, self_mesh, right_mesh);
		}
		//else Topological degenerate!
	}

	///If an ear will make a degenerate triangle, clip it early to avoid
	///difficulty in key-holing. This function is recursive, as the process of
	///clipping may cause the neighbors to degenerate.
	fn clip_if_degenerate(
		ear: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		result: &mut HalfedgeTriangulation,
		epsilon: f64,
	) {
		let ear_ref = &polygon[ear];
		if Self::clipped(ear_ref) {
			return;
		}

		if ear_ref.left == ear_ref.right {
			return;
		}

		if ear_ref.is_short(epsilon)
			|| (ccw(
				ear_ref.left().pos,
				ear_ref.pos,
				ear_ref.right().pos,
				epsilon,
			) == 0 && (ear_ref.left().pos - ear_ref.pos)
				.dot(&(ear_ref.right().pos - ear_ref.pos))
				> 0.0)
		{
			Self::clip_ear(ear, polygon, polygon_range, result);
			let ear_ref = &polygon[ear];
			let left = ear_ref.left().ptr2index(polygon_range);
			let right = ear_ref.right().ptr2index(polygon_range);
			Self::clip_if_degenerate(left, polygon, polygon_range, result, epsilon);
			Self::clip_if_degenerate(right, polygon, polygon_range, result, epsilon);
		}
	}

	///Build the circular list polygon structures.
	fn initialize(&mut self, polys: &PolygonsIdx) -> Vec<usize> {
		let mut starts = Vec::new();
		for poly in polys {
			let vert = &poly[0];
			self.polygon.push(Vert {
				mesh_idx: vert.idx,
				cost: 0.0,
				ear: false,
				pos: vert.pos,
				right_dir: Vector2::new(0.0, 0.0),
				left: ptr::null_mut::<Vert>(),
				right: ptr::null_mut::<Vert>(),
			});

			let first = self.polygon.last().unwrap();

			self.bbox.union(first.pos);
			let first = self.polygon.len() - 1;
			let mut last = first;
			// This is not the real rightmost start, but just an arbitrary vert for
			// now to identify each polygon.
			starts.push(first);

			for vert in &poly[1..] {
				self.bbox.union(vert.pos);

				self.polygon.push(Vert {
					mesh_idx: vert.idx,
					cost: 0.0,
					ear: false,
					pos: vert.pos,
					right_dir: Vector2::new(0.0, 0.0),
					left: ptr::null_mut::<Vert>(),
					right: ptr::null_mut::<Vert>(),
				});

				let next = self.polygon.len() - 1;
				Self::link(last, next, &mut self.polygon, &self.polygon_range);
				last = next
			}

			Self::link(last, first, &mut self.polygon, &self.polygon_range);
		}

		if self.epsilon < 0.0 {
			self.epsilon = self.bbox.scale() * K_PRECISION;
		}

		self.result.add_contours(polys);
		// Slightly more than enough, since each hole can cause two extra triangles.
		self.result
			.reserve_triangles(self.polygon.len() + 2 * starts.len());

		starts
	}

	///Find the actual rightmost starts after degenerate removal. Also calculate
	///the polygon bounding boxes. Only holes need rightmost starts, and the CCW
	///check is to handle degenerate cases for them - the other contour starts are
	///arbitrary.
	fn find_start(&mut self, first: usize) {
		let origin = self.polygon[first].pos;

		let mut start = first;
		let mut max_x = f64::NEG_INFINITY;
		let mut bbox = Rect::default();
		// Kahan summation
		let mut area = 0.0;
		let mut area_compensation = 0.0;

		let add_point = |v: usize, polygon: &mut Vec<Vert>| {
			let v = &polygon[v];
			bbox.union(v.pos);
			let area1 =
				Matrix2::from_columns(&[v.pos - origin, v.right().pos - origin]).determinant();
			let t1 = area + area1;
			area_compensation += (area - t1) + area1;
			area = t1;

			if v.pos.x > max_x && v.is_reflex(self.epsilon) {
				max_x = v.pos.x;
				start = v.ptr2index(&self.polygon_range);
			}
		};

		if Self::loop_verts(first, &mut self.polygon, &self.polygon_range, add_point).is_none() {
			// No polygon left if all ears were degenerate and already clipped.
			return;
		}

		area += area_compensation;
		let size = bbox.size();
		let min_area = self.epsilon * size.x.max(size.y);

		if max_x.is_finite() && area < -min_area {
			self.holes
				.insert_sorted_by_key(start, |&hole| Reverse(OrderedF64(self.polygon[hole].pos.x))); //descending pos.x
			self.hole2bbox.entry(start).or_insert(bbox);
		} else {
			self.simples.push(start);
			if area > min_area {
				self.outers.push(start);
			}
		}
	}

	///All holes must be key-holed (attached to an outer polygon) before ear
	///clipping can commence. Instead of relying on sorting, which may be
	///incorrect due to epsilon, we check for polygon edges both ahead and
	///behind to ensure all valid options are found.
	fn cut_keyhole(
		start: usize,
		simples: &mut Vec<usize>,
		result: &mut HalfedgeTriangulation,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		outers: &Vec<usize>,
		hole2bbox: &BTreeMap<usize, Rect>,
		epsilon: f64,
	) {
		let bbox = hole2bbox.get(&start).unwrap();
		let start_ref = &polygon[start];
		let on_top = if start_ref.pos.y >= bbox.max.y - epsilon {
			1
		} else if start_ref.pos.y <= bbox.min.y + epsilon {
			-1
		} else {
			0
		};

		let mut connector: Option<usize> = None;

		let mut check_edge = |edge: usize, polygon: &mut Vec<Vert>| {
			let edge = &polygon[edge];
			let start_ref = &polygon[start];
			let x = edge.interp_y2x(start_ref.pos, on_top, epsilon);
			if x.is_finite()
				&& start_ref.inside_edge(edge, epsilon, true)
				&& (connector.as_ref().map_or(true, |&connector| {
					ccw(
						Point2::new(x, start_ref.pos.y),
						polygon[connector].pos,
						polygon[connector].right().pos,
						epsilon,
					) == 1 || (if polygon[connector].pos.y < edge.pos.y {
						edge.inside_edge(&polygon[connector], epsilon, false)
					} else {
						!polygon[connector].inside_edge(edge, epsilon, false)
					})
				})) {
				connector = Some(edge.ptr2index(polygon_range));
			}
		};

		for &first in outers {
			Self::loop_verts(first, polygon, polygon_range, &mut check_edge);
		}

		if connector.is_none() {
			//hole did not find an outer contour!
			simples.push(start);
			return;
		}

		let connector = Self::find_closer_bridge(
			start,
			connector.unwrap(),
			polygon,
			polygon_range,
			outers,
			epsilon,
		);

		Self::join_polygons(start, connector, polygon, polygon_range, result, epsilon);
	}

	///This converts the initial guess for the keyhole location into the final one
	///and returns it. It does so by finding any reflex verts inside the triangle
	///containing the best connection and the initial horizontal line.
	fn find_closer_bridge(
		start: usize,
		edge: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		outers: &Vec<usize>,
		epsilon: f64,
	) -> usize {
		let start_ref = &polygon[start];
		let edge_ref = &polygon[edge];
		let connector_ref = if edge_ref.pos.x < start_ref.pos.x {
			edge_ref.right()
		} else if edge_ref.right().pos.x < start_ref.pos.x {
			edge_ref
		} else if edge_ref.right().pos.y - start_ref.pos.y > start_ref.pos.y - edge_ref.pos.y {
			edge_ref
		} else {
			edge_ref.right()
		};

		let mut connector = connector_ref.ptr2index(polygon_range);
		if (connector_ref.pos.y - start_ref.pos.y).abs() <= epsilon {
			return connector;
		}

		let above_i32 = if connector_ref.pos.y > start_ref.pos.y {
			1
		} else {
			-1
		};
		let above_f64 = above_i32 as f64;

		let mut check_vert = |vert: usize, polygon: &mut Vec<Vert>| {
			let vert = &polygon[vert];
			let start_ref = &polygon[start];
			let edge_ref = &polygon[edge];
			let connector_ref = &polygon[connector];

			let inside = above_i32 * ccw(start_ref.pos, vert.pos, connector_ref.pos, epsilon);
			if vert.pos.x > start_ref.pos.x - epsilon
				&& vert.pos.y * above_f64 > start_ref.pos.y * above_f64 - epsilon
				&& (inside > 0
					|| (inside == 0
						&& vert.pos.x < connector_ref.pos.x
						&& vert.pos.y * above_f64 < connector_ref.pos.y * above_f64))
				&& vert.inside_edge(edge_ref, epsilon, true)
				&& vert.is_reflex(epsilon)
			{
				connector = vert.ptr2index(polygon_range);
			}
		};

		for &first in outers {
			Self::loop_verts(first, polygon, polygon_range, &mut check_vert);
		}

		connector
	}

	///Creates a keyhole between the start vert of a hole and the connector vert
	///of an outer polygon. To do this, both verts are duplicated and reattached.
	///This process may create degenerate ears, so these are clipped if necessary
	///to keep from confusing subsequent key-holing operations.
	fn join_polygons(
		start: usize,
		connector: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		result: &mut HalfedgeTriangulation,
		epsilon: f64,
	) {
		let new_start = polygon.len();
		polygon.push(polygon[start].clone());
		let new_connector = polygon.len();
		polygon.push(polygon[connector].clone());

		polygon[start].right_mut().left = Vert::index2ptr(new_start, polygon_range);
		polygon[connector].left_mut().right = Vert::index2ptr(new_connector, polygon_range);
		Self::link(start, connector, polygon, polygon_range);
		Self::link(new_connector, new_start, polygon, polygon_range);

		Self::clip_if_degenerate(start, polygon, polygon_range, result, epsilon);
		Self::clip_if_degenerate(new_start, polygon, polygon_range, result, epsilon);
		Self::clip_if_degenerate(connector, polygon, polygon_range, result, epsilon);
		Self::clip_if_degenerate(new_connector, polygon, polygon_range, result, epsilon);
	}

	fn process_ear(
		v: usize,
		collider: &IdxCollider,
		ears_queue: &mut VecDeque<usize>,
		polygon: &mut Vec<Vert>,
		epsilon: f64,
	) {
		if polygon[v].ear {
			ears_queue.remove(ears_queue.iter().position(|&ball| ball == v).unwrap());
			polygon[v].ear = false;
		}

		if polygon[v].is_short(epsilon) {
			polygon[v].cost = K_BEST;
			polygon[v].ear = true;
			ears_queue.insert_sorted_by_key(v, |&ear| OrderedF64(polygon[ear].cost)); //ascending cost
		} else if polygon[v].is_convex(2.0 * epsilon) {
			polygon[v].cost = polygon[v].ear_cost(epsilon, collider, polygon);
			polygon[v].ear = true;
			ears_queue.insert_sorted_by_key(v, |&ear| OrderedF64(polygon[ear].cost)); //ascending cost
		} else {
			polygon[v].cost = 1.0; // not used, but marks reflex verts for debug
		}
	}

	///Create a collider of all vertices in this polygon, each expanded by
	///epsilon_. Each ear uses this BVH to quickly find a subset of vertices to
	///check for cost.
	fn vert_collider(
		start: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
	) -> IdxCollider {
		let mut itr = Vec::new();
		let mut points = Vec::new();
		Self::loop_verts(start, polygon, polygon_range, |v, polygon| {
			points.push(PolyVert {
				pos: polygon[v].pos,
				idx: itr.len() as i32,
			});

			itr.push(v);
		});

		build_2d_tree(&mut points);
		IdxCollider { points, itr }
	}

	///The main ear-clipping loop. This is called once for each simple polygon -
	///all holes have already been key-holed and joined to an outer polygon.
	fn triangulate_poly(
		start: usize,
		polygon: &mut Vec<Vert>,
		polygon_range: &Range<usize>,
		result: &mut HalfedgeTriangulation,
		epsilon: f64,
	) {
		let vert_collider = Self::vert_collider(start, polygon, polygon_range);

		if vert_collider.itr.is_empty() {
			//empty poly
			return;
		}

		// A simple polygon always creates two fewer triangles than it has verts.
		let mut num_tri = -2;

		// A priority queue of valid ears - the multiset allows them to be updated.
		//c++ uses multiset here, whose big o complexity is probably more desirable here
		let mut ears_queue = VecDeque::new();

		let queue_vert = |v, polygon: &mut Vec<Vert>| {
			Self::process_ear(v, &vert_collider, &mut ears_queue, polygon, epsilon);
			num_tri += 1;
		};

		let v = Self::loop_verts(start, polygon, polygon_range, queue_vert);
		if v.is_none() {
			return;
		}
		let mut v = v.unwrap();

		while num_tri > 0 {
			let ear = ears_queue.front();
			if let Some(&ear) = ear {
				v = ear;
				// Cost should always be negative, generally < -epsilon.
				ears_queue.pop_front();
			} else {
				//no ear found!
			}

			Self::clip_ear(v, polygon, polygon_range, result);
			num_tri -= 1;

			let ear_left = polygon[v].left().ptr2index(polygon_range);
			Self::process_ear(ear_left, &vert_collider, &mut ears_queue, polygon, epsilon);
			let ear_right = polygon[v].right().ptr2index(polygon_range);
			Self::process_ear(ear_right, &vert_collider, &mut ears_queue, polygon, epsilon);
			// This is a backup vert that is used if the queue is empty (geometrically
			// invalid polygon), to ensure manifoldness.
			v = ear_right;
		}

		debug_assert!(polygon[v].right == polygon[v].left, "Triangulator error!");
		//finished poly
	}
}

struct IdxCollider {
	points: Vec<PolyVert>,
	itr: Vec<usize>,
}

/// A circularly-linked list representing the polygon(s) that still need to be
/// triangulated. This gets smaller as ears are clipped until it degenerates to
/// two points and terminates.
#[derive(Clone)]
struct Vert {
	mesh_idx: i32,
	cost: f64,
	ear: bool,
	pos: Point2<f64>,
	right_dir: Vector2<f64>,
	left: *mut Vert,
	right: *mut Vert,
	//note left and right could point to the same vert,
	//as evidenced by c++: if (v->right == v->left)
}

impl Vert {
	//safety (for all 4): the verts are stored basically in a vec being used as an arena.
	//the vec never reallocates and all verts are dropped together at the end
	fn left(&self) -> &Self {
		unsafe { &*self.left }
	}

	fn right(&self) -> &Self {
		unsafe { &*self.right }
	}

	fn left_mut(&mut self) -> &mut Self {
		unsafe { &mut *self.left }
	}

	fn right_mut(&mut self) -> &mut Self {
		unsafe { &mut *self.right }
	}

	//safety: this is only meant for assigning left+right fields, which should
	//only be dereferenced through the accessor methods in order to let the
	//borrow checker do its job
	fn index2ptr(index: usize, polygon_range: &Range<usize>) -> *mut Vert {
		assert!(index < polygon_range.len() / size_of::<Vert>());
		(polygon_range.start + index * size_of::<Vert>()) as *mut Vert
	}

	fn ptr2index(&self, polygon_range: &Range<usize>) -> usize {
		let address = self as *const Vert as usize;
		assert!(address >= polygon_range.start && address < polygon_range.end);

		(address - polygon_range.start) / size_of::<Vert>()
	}

	fn is_short(&self, epsilon: f64) -> bool {
		let edge = self.right().pos - self.pos;
		edge.magnitude_squared() * 4.0 < epsilon.powi(2)
	}

	///Returns true if Vert is on the inside of the edge that goes from tail to
	///tail->right. This will walk the edges if necessary until a clear answer
	///is found (beyond epsilon). If toLeft is true, this Vert will walk its
	///edges to the left. This should be chosen so that the edges walk in the
	///same general direction - tail always walks to the right.
	fn inside_edge(&self, tail: &Vert, epsilon: f64, to_left: bool) -> bool {
		let p2 = epsilon.powi(2);
		let mut next_l = self.left().right();
		let mut next_r = tail.right();
		let mut center = tail;
		let mut last = center;

		while !ptr::eq(next_l, next_r)
			&& !ptr::eq(tail, next_r)
			&& !ptr::eq(next_l, if to_left { self.right } else { self.left })
		{
			let edge_l = next_l.pos - center.pos;
			let l2 = edge_l.magnitude_squared();
			if l2 <= p2 {
				next_l = if to_left {
					next_l.left()
				} else {
					next_l.right()
				};
				continue;
			}

			let edge_r = next_r.pos - center.pos;
			let r2 = edge_r.magnitude_squared();
			if r2 <= p2 {
				next_r = next_r.right();
				continue;
			}

			let vec_lr = next_r.pos - next_l.pos;
			let lr2 = vec_lr.magnitude_squared();
			if lr2 <= p2 {
				last = center;
				center = next_l;
				next_l = if to_left {
					next_l.left()
				} else {
					next_l.right()
				};
				if ptr::eq(next_l, next_r) {
					break;
				}
				next_r = next_r.right();
				continue;
			}

			let mut convexity = ccw(next_l.pos, center.pos, next_r.pos, epsilon);
			if !ptr::eq(center, last) {
				convexity += ccw(last.pos, center.pos, next_l.pos, epsilon)
					+ ccw(next_r.pos, center.pos, last.pos, epsilon);
			}

			if convexity != 0 {
				return convexity > 0;
			}

			if l2 < r2 {
				center = next_l;
				next_l = if to_left {
					next_l.left()
				} else {
					next_l.right()
				};
			} else {
				center = next_r;
				next_r = next_r.right();
			}

			last = center;
		}

		// The whole polygon is degenerate - consider this to be convex.
		true
	}

	///Returns true for convex or colinear ears.
	fn is_convex(&self, epsilon: f64) -> bool {
		ccw(self.left().pos, self.pos, self.right().pos, epsilon) >= 0
	}

	///Subtly different from !IsConvex because IsConvex will return true for
	///colinear non-folded verts, while IsReflex will always check until actual
	///certainty is determined.
	fn is_reflex(&self, epsilon: f64) -> bool {
		let left = self.left();
		!left.inside_edge(left.right(), epsilon, true)
	}

	fn interp_y2x(&self, start: Point2<f64>, on_top: i32, epsilon: f64) -> f64 {
		let right_pos_y = self.right().pos.y;
		if (self.pos.y - start.y).abs() <= epsilon {
			if right_pos_y <= start.y + epsilon || on_top == 1 {
				f64::NAN
			} else {
				self.pos.x
			}
		} else if self.pos.y < start.y - epsilon {
			if right_pos_y > start.y + epsilon {
				self.pos.x
					+ (start.y - self.pos.y) * (self.right().pos.x - self.pos.x)
						/ (right_pos_y - self.pos.y)
			} else if right_pos_y < start.y - epsilon || on_top == -1 {
				f64::NAN
			} else {
				self.right().pos.x
			}
		} else {
			f64::NAN
		}
	}

	///This finds the cost of this vert relative to one of the two closed sides
	///of the ear. Points are valid even when they touch, so long as their edge
	///goes to the outside. No need to check the other side, since all verts are
	///processed in the EarCost loop.
	fn signed_dist(&self, v: &Vert, unit: Vector2<f64>, epsilon: f64) -> f64 {
		let d = Matrix2::from_columns(&[unit, v.pos - self.pos]).determinant();
		if d.abs() < epsilon {
			let d_r = Matrix2::from_columns(&[unit, v.right().pos - self.pos]).determinant();
			if d_r.abs() > epsilon {
				return d_r;
			}
			let d_l = Matrix2::from_columns(&[unit, v.left().pos - self.pos]).determinant();
			if d_l.abs() > epsilon {
				return d_l;
			}
		}

		d
	}

	///Find the cost of Vert v within this ear, where openSide is the unit
	///vector from Verts right to left - passed in for reuse.
	fn cost(&self, v: &Vert, open_side: Vector2<f64>, epsilon: f64) -> f64 {
		let cost = self
			.signed_dist(v, self.right_dir, epsilon)
			.min(self.signed_dist(v, self.left().right_dir, epsilon));

		let open_cost = Matrix2::from_columns(&[open_side, v.pos - self.right().pos]).determinant();
		cost.min(open_cost)
	}

	///For verts outside the ear, apply a cost based on the Delaunay condition
	///to aid in prioritization and produce cleaner triangulations. This doesn't
	///affect robustness, but may be adjusted to improve output.
	fn delaunay_cost(diff: Vector2<f64>, scale: f64, epsilon: f64) -> f64 {
		-epsilon - scale * diff.magnitude_squared()
	}

	///This is the expensive part of the algorithm, checking this ear against
	///every Vert to ensure none are inside. The Collider brings the total
	///triangulator cost down from O(n^2) to O(nlogn) for most large polygons.
	///
	///Think of a cost as vaguely a distance metric - 0 is right on the edge of
	///being invalid. cost > epsilon is definitely invalid. Cost < -epsilon
	///is definitely valid, so all improvement costs are designed to always give
	///values < -epsilon so they will never affect validity. The first
	///totalCost is designed to give priority to sharper angles. Any cost < (-1
	///- epsilon) has satisfied the Delaunay condition.
	fn ear_cost(&self, epsilon: f64, collider: &IdxCollider, polygon: &Vec<Vert>) -> f64 {
		let left_pos = self.left().pos;
		let right_pos = self.right().pos;

		let mut open_side = left_pos - right_pos;
		let center = nalgebra::center(&left_pos, &right_pos);
		let scale = 4.0 / open_side.magnitude_squared();
		let radius = open_side.magnitude() / 2.0;
		open_side = open_side.normalize();

		let mut total_cost = self.left().right_dir.dot(&self.right_dir) - 1.0 - epsilon;
		if ccw(self.pos, left_pos, right_pos, epsilon) == 0 {
			// Clip folded ears first
			return total_cost;
		}

		let mut ear_box = Rect::new(
			center.coords.add_scalar(-radius).into(),
			center.coords.add_scalar(radius).into(),
		);
		ear_box.union(self.pos);
		ear_box.min.coords.add_scalar_mut(-epsilon);
		ear_box.max.coords.add_scalar_mut(epsilon);

		let lid = self.left().mesh_idx;
		let rid = self.right().mesh_idx;
		query_2d_tree(&collider.points, ear_box, |point| {
			let test = &polygon[collider.itr[point.idx as usize]];
			if !EarClip::clipped(test)
				&& test.mesh_idx != self.mesh_idx
				&& test.mesh_idx != lid
				&& test.mesh_idx != rid
			{
				// Skip duplicated verts
				let mut cost = self.cost(test, open_side, epsilon);
				if cost < -epsilon {
					cost = Self::delaunay_cost(test.pos - center, scale, epsilon);
				}

				if cost > total_cost {
					total_cost = cost;
				}
			}
		});

		total_cost
	}
}

///@brief Triangulates a set of &epsilon;-valid polygons. If the input is not
///&epsilon;-valid, the triangulation may overlap, but will always return a
///manifold result that matches the input edge directions.
///
///@param polys The set of polygons, wound CCW and representing multiple
///polygons and/or holes. These have 2D-projected positions as well as
///references back to the original vertices.
///@param epsilon The value of &epsilon;, bounding the uncertainty of the
///input.
///@param allowConvex If true (default), the triangulator will use a fast
///triangulation if the input is convex, falling back to ear-clipping if not.
///The triangle quality may be lower, so set to false to disable this
///optimization.
///@return HalfedgeTriangulation The contour and triangle halfedges,
///referencing the original vertex indicies.
pub fn triangulate_idx_halfedges(
	polys: &PolygonsIdx,
	epsilon: f64,
	allow_convex: bool,
) -> HalfedgeTriangulation {
	let mut result;
	let mut updated_epsilon = epsilon;
	if allow_convex && is_convex(polys, epsilon)
	//fast path
	{
		result = triangulate_convex(polys)
	} else {
		let mut triangulator = EarClip::new(polys, epsilon);
		result = triangulator.triangulate();
		updated_epsilon = triangulator.get_precision();
	};
	result.epsilon = updated_epsilon;
	result.finalize();
	result
}

pub fn triangulate_idx(polys: &PolygonsIdx, epsilon: f64, allow_convex: bool) -> Vec<Vector3<i32>> {
	let result = triangulate_idx_halfedges(polys, epsilon, allow_convex);
	result.triangles()
}

///@brief Triangulates a set of &epsilon;-valid polygons. If the input is not
///&epsilon;-valid, the triangulation may overlap, but will always return a
///manifold result that matches the input edge directions.
///
///@param polygons The set of polygons, wound CCW and representing multiple
///polygons and/or holes.
///@param epsilon The value of &epsilon;, bounding the uncertainty of the
///input.
///@param allowConvex If true (default), the triangulator will use a fast
///triangulation if the input is convex, falling back to ear-clipping if not.
///The triangle quality may be lower, so set to false to disable this
///optimization.
///@return std::vector<ivec3> The triangles, referencing the original
///polygon points in order.
pub fn triangulate(polygons: &Polygons, epsilon: f64, allow_convex: bool) -> Vec<Vector3<i32>> {
	let mut idx: i32 = 0;
	let mut polygons_indexed = PolygonsIdx::default();
	for poly in polygons.iter() {
		let mut simple_indexed = SimplePolygonIdx::default();
		for poly_vert in poly.iter() {
			simple_indexed.push(PolyVert {
				pos: poly_vert.clone(),
				idx,
			});
			idx += 1;
		}
		polygons_indexed.push(simple_indexed);
	}
	triangulate_idx(&polygons_indexed, epsilon, allow_convex)
}
