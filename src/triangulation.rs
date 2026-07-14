use crate::halfedge::Halfedge;
use crate::triangulation::ear_clip::triangulate_ear_clip;
use crate::util::hash_table::DeterministicMap;
use nalgebra::{Point2, Vector3};
use std::collections::hash_map::Entry;

#[cfg(feature = "test_thoroughly")]
use {
	crate::test::{get_intermediate_checks, get_process_overlaps},
	crate::util::math::ccw,
	std::mem,
};

mod ear_clip;

///Polygon vertex.
#[derive(Debug)]
pub struct PolyVert {
	/// X-Y position
	pub pos: Point2<f64>,
	/// ID or index into another vertex vector
	pub idx: i32,
}

pub type SimplePolygon = Vec<Point2<f64>>;
pub type Polygons = Vec<SimplePolygon>;
pub type SimplePolygonIdx = Vec<PolyVert>;
pub type PolygonsIdx = Vec<SimplePolygonIdx>;

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

pub fn triangulate_idx(polys: &PolygonsIdx, epsilon: f64, allow_convex: bool) -> Vec<Vector3<i32>> {
	triangulate_idx_halfedges(polys, epsilon, allow_convex).triangles()
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
	mut epsilon: f64,
	allow_convex: bool,
) -> HalfedgeTriangulation {
	let mut result;
	if allow_convex && is_convex(polys, epsilon) {
		//fast path
		result = triangulate_convex(polys)
	} else {
		result = triangulate_ear_clip(polys, &mut epsilon);
	};

	#[cfg(feature = "test_thoroughly")]
	{
		if get_intermediate_checks() {
			check_topology(&halfedges2edges(&result));
			if !get_process_overlaps() {
				check_geometry(&result.triangles(), polys, 2.0 * epsilon);
			}
		}

		debug_assert!(
			result.edge2halfedge.is_empty(),
			"triangulation has unpaired halfedges"
		);
		for i in 0..result.halfedges.len() {
			let pair = result.halfedges[i].paired_halfedge;
			debug_assert!(
				pair >= 0 && pair < result.halfedges.len() as i32,
				"invalid paired halfedge"
			);
			debug_assert!(
				result.halfedges[pair as usize].paired_halfedge == i as i32,
				"halfedge pair is not reciprocal"
			);
			debug_assert!(
				result.halfedges[i].start_vert == result.halfedges[pair as usize].end_vert
					&& result.halfedges[i].end_vert == result.halfedges[pair as usize].start_vert,
				"halfedge pair endpoints do not match"
			);
		}
	}

	result.finalize();
	result
}

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
	let mut result = HalfedgeTriangulation::new();
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

#[derive(Clone, Debug)]
pub struct HalfedgeTriangulation {
	pub halfedges: Vec<Halfedge>,
	pub contour_end: usize,
	edge2halfedge: DeterministicMap<u64, Vec<i32>>,
}

impl HalfedgeTriangulation {
	fn new() -> Self {
		Self {
			halfedges: Vec::default(),
			contour_end: 0,
			edge2halfedge: DeterministicMap::default(),
		}
	}

	pub fn num_tri(&self) -> usize {
		(self.halfedges.len() - self.contour_end) / 3
	}

	fn add_contours(&mut self, polys: &PolygonsIdx) {
		let mut num_contour_edges = 0;
		for poly in polys {
			num_contour_edges += poly.len();
		}
		self.halfedges.reserve(num_contour_edges);
		self.edge2halfedge.reserve(num_contour_edges);
		for poly in polys {
			for i in 0..poly.len() {
				let start = poly[i].idx;
				let end = poly[if i + 1 < poly.len() { i + 1 } else { 0 }].idx;
				// Store the exterior contour halfedge, opposite the filled contour.
				self.add_halfedge(end, start);
			}
		}
		self.contour_end = self.halfedges.len();
	}

	fn reserve_triangles(&mut self, num_tri: usize) {
		self.halfedges
			.reserve((self.contour_end + 3 * num_tri).saturating_sub(self.halfedges.len()));
		self.edge2halfedge.reserve(num_tri);
	}

	fn add_triangle(&mut self, first: i32, second: i32, third: i32) {
		self.add_halfedge(first, second);
		self.add_halfedge(second, third);
		self.add_halfedge(third, first);
	}

	fn finalize(&mut self) {
		self.edge2halfedge = DeterministicMap::new();
	}

	fn triangles(&self) -> Vec<Vector3<i32>> {
		let mut triangles = Vec::with_capacity(self.num_tri());
		let mut edge = self.contour_end;
		while edge < self.halfedges.len() {
			triangles.push(Vector3::new(
				self.halfedges[edge].start_vert,
				self.halfedges[edge + 1].start_vert,
				self.halfedges[edge + 2].start_vert,
			));
			edge += 3;
		}
		triangles
	}

	fn edge_key(start: i32, end: i32) -> u64 {
		((start as u32 as u64) << 32) | (end as u32 as u64)
	}

	fn add_halfedge(&mut self, start: i32, end: i32) {
		let halfedge = self.halfedges.len();
		let mut data = Halfedge {
			start_vert: start,
			end_vert: end,
			paired_halfedge: -1,
			prop_vert: -1,
		};
		if let Entry::Occupied(mut reverse_entry) =
			self.edge2halfedge.entry(Self::edge_key(end, start))
			&& !reverse_entry.get().is_empty()
		{
			let reverse = reverse_entry.get_mut();
			data.paired_halfedge = *reverse.last().unwrap();
			self.halfedges[data.paired_halfedge as usize].paired_halfedge = halfedge as i32;
			reverse.pop().unwrap();
			if reverse.is_empty() {
				reverse_entry.remove();
			}
		} else {
			self.edge2halfedge
				.entry(Self::edge_key(start, end))
				.or_default()
				.push(halfedge as i32);
		}
		self.halfedges.push(data);
	}
}

#[cfg(feature = "test_thoroughly")]
#[derive(Clone)]
struct PolyEdge {
	start_vert: i32,
	end_vert: i32,
}

#[cfg(feature = "test_thoroughly")]
fn halfedges2edges(result: &HalfedgeTriangulation) -> Vec<PolyEdge> {
	let mut halfedges = Vec::with_capacity(result.halfedges.len());
	for edge in &result.halfedges {
		halfedges.push(PolyEdge {
			start_vert: edge.start_vert,
			end_vert: edge.end_vert,
		});
	}
	halfedges
}

#[cfg(feature = "test_thoroughly")]
fn check_topology(halfedges: &[PolyEdge]) {
	debug_assert!(halfedges.len() % 2 == 0, "Odd number of halfedges.");
	let n_edges = halfedges.len() / 2;
	let mut forward = Vec::with_capacity(n_edges);
	let mut backward = Vec::with_capacity(n_edges);

	forward.extend(
		halfedges
			.iter()
			.cloned()
			.filter(|e| e.end_vert > e.start_vert),
	);
	debug_assert!(
		forward.len() == n_edges,
		"Half of halfedges should be forward."
	);

	backward.extend(
		halfedges
			.iter()
			.cloned()
			.filter(|e| e.end_vert < e.start_vert),
	);
	debug_assert!(
		backward.len() == n_edges,
		"Half of halfedges should be backward."
	);

	for e in backward.iter_mut() {
		mem::swap(&mut e.start_vert, &mut e.end_vert);
	}
	forward.sort_by_key(|edge| (edge.start_vert, edge.end_vert));
	backward.sort_by_key(|edge| (edge.start_vert, edge.end_vert));
	for i in 0..n_edges {
		debug_assert!(
			forward[i].start_vert == backward[i].start_vert
				&& forward[i].end_vert == backward[i].end_vert,
			"Not manifold."
		);
	}
}

#[cfg(feature = "test_thoroughly")]
fn check_geometry(triangles: &[Vector3<i32>], polys: &PolygonsIdx, epsilon: f64) {
	let mut vert_pos: DeterministicMap<i32, Point2<f64>> = DeterministicMap::new();
	for poly in polys {
		for i in 0..poly.len() {
			vert_pos.insert(poly[i].idx, poly[i].pos);
		}
	}
	debug_assert!(
		triangles.iter().all(|tri| ccw(
			vert_pos[&tri[0]],
			vert_pos[&tri[1]],
			vert_pos[&tri[2]],
			epsilon
		) >= 0),
		"triangulation is not entirely CCW!"
	);
}
