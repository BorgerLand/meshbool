use crate::halfedge::Halfedge;
use crate::triangulation::{PolyVert, PolygonsIdx, SimplePolygonIdx};
use nalgebra::{Matrix2x3, Point3};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::collections::hash_map::Entry;

///Add the vertex position projection to the indexed polygons.
pub fn project_polygons(
	polys: &[Vec<i32>],
	halfedge: &[Halfedge],
	vert_pos: &[Point3<f64>],
	projection: Matrix2x3<f64>,
) -> PolygonsIdx {
	let mut polygons = PolygonsIdx::new();
	for poly in polys {
		let mut polygon = SimplePolygonIdx::new();
		for &edge in poly {
			polygon.push(PolyVert {
				pos: (projection * vert_pos[halfedge[edge as usize].start_vert as usize]),
				idx: edge,
			});
		} //for vert

		polygons.push(polygon);
	} //for poly

	polygons
}

///Returns an assembled set of vertex index loops of the input list of
///Halfedges, where each vert must be referenced the same number of times as a
///startVert and endVert. If startHalfedgeIdx is given, instead of putting
///vertex indices into the returned polygons structure, it will use the halfedge
///indices instead.
pub fn assemble_halfedges(edges: &[Halfedge], start_halfedge_idx: i32) -> Vec<Vec<i32>> {
	let mut vert_edge: FxHashMap<i32, VecDeque<i32>> = FxHashMap::default(); //originally a c++ multimap
	for (i, edge) in edges.iter().enumerate() {
		vert_edge
			.entry(edge.start_vert)
			.or_default()
			.push_back(i as i32);
	}

	let mut polys = Vec::new();
	let mut start_edge = 0;
	let mut this_edge = start_edge;
	loop {
		if this_edge == start_edge {
			if vert_edge.is_empty() {
				break;
			}
			start_edge = vert_edge.values().next().unwrap()[0];
			this_edge = start_edge;
			polys.push(Vec::new());
		}

		polys
			.last_mut()
			.unwrap()
			.push(start_halfedge_idx + this_edge);
		let Entry::Occupied(mut result) = vert_edge.entry(edges[this_edge as usize].end_vert)
		else {
			panic!("non-manifold edge");
		};
		this_edge = result.get_mut().pop_front().expect("non-manifold edge");
		if result.get().is_empty() {
			result.remove();
		}
	}

	polys
}
