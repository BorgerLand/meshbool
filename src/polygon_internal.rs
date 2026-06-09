use crate::common::DeterministicMap;
use crate::polygon::PolygonsIdx;
use crate::shared::Halfedge;
use nalgebra::Vector3;
use std::collections::hash_map::Entry;

#[derive(Clone)]
pub struct HalfedgeTriangulation {
	pub halfedges: Vec<Halfedge>,
	pub contour_end: usize,
	pub epsilon: f64,
	edge2halfedge: DeterministicMap<u64, Vec<i32>>,
}

impl Default for HalfedgeTriangulation {
	fn default() -> Self {
		Self {
			halfedges: Vec::default(),
			contour_end: 0,
			epsilon: -1.0,
			edge2halfedge: DeterministicMap::default(),
		}
	}
}

impl HalfedgeTriangulation {
	pub fn add_contours(&mut self, polys: &PolygonsIdx) {
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

	pub fn reserve_triangles(&mut self, num_tri: usize) {
		self.halfedges
			.reserve((self.contour_end + 3 * num_tri).saturating_sub(self.halfedges.len()));
		self.edge2halfedge.reserve(num_tri);
	}

	pub fn add_triangle(&mut self, first: i32, second: i32, third: i32) {
		self.add_halfedge(first, second);
		self.add_halfedge(second, third);
		self.add_halfedge(third, first);
	}

	pub fn num_tri(&self) -> usize {
		(self.halfedges.len() - self.contour_end) / 3
	}

	pub fn triangles(&self) -> Vec<Vector3<i32>> {
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

	pub fn finalize(&mut self) {
		#[cfg(feature = "test")]
		{
			debug_assert!(
				self.edge2halfedge.is_empty(),
				"triangulation has unpaired halfedges"
			);
			for i in 0..self.halfedges.len() {
				let pair = self.halfedges[i].paired_halfedge;
				debug_assert!(
					pair >= 0 && pair < self.halfedges.len() as i32,
					"invalid paired halfedge"
				);
				debug_assert!(
					self.halfedges[pair as usize].paired_halfedge == i as i32,
					"halfedge pair is not reciprocal"
				);
				debug_assert!(
					self.halfedges[i].start_vert == self.halfedges[pair as usize].end_vert
						&& self.halfedges[i].end_vert == self.halfedges[pair as usize].start_vert,
					"halfedge pair endpoints do not match"
				);
			}
		}
		self.edge2halfedge = DeterministicMap::new();
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
