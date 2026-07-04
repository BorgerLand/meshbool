pub use crate::postprocessing::sort::sort_and_compact_geometry;

use crate::halfedge::{Halfedges, next_halfedge};
use crate::mesh_relations::TriRelation;
use crate::util::disjoint_sets::DisjointSets;
use crate::util::hash_table::DeterministicMap;
use crate::util::math::{ccw, get_axis_aligned_projection};
use crate::util::num_convert::OrderedF64;
use crate::util::vec_ext;
use crate::{Precision, Properties, Triangles};
use nalgebra::{Point2, Point3, Vector3};
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::f64;

mod edge;
pub mod sort;

///In the case of a very bad triangulation, it is possible to create pinched
///verts. They must be removed before edge collapse.
///
///The idea here is to identify cycles of halfedges that can be iterated
///through using ForVert. Pinched verts are vertices where there are
///multiple cycles associated with the vertex. Each cycle is identified with
///the smallest halfedge index within the cycle, and when there are multiple
///cycles associated with the same starting vertex but with different ids,
///it means we have a pinched vertex. This check is done by using a single
///atomic cas operation, the expected case is either invalid id (the vertex
///was not processed) or with the same id.
pub fn split_pinched_verts(halfedge: &mut Halfedges, vert_pos: &mut Vec<Point3<f64>>) {
	debug_assert!(halfedge.is_manifold(), "polygon mesh is not manifold!");

	let nb_edges = halfedge.len();

	let mut vert_processed = vec![false; vert_pos.len()];
	let mut halfedge_processed = vec![false; nb_edges];
	for i in 0..nb_edges as i32 {
		if halfedge_processed[i as usize] {
			continue;
		}
		let mut vert = halfedge.start(i);
		if vert == -1 {
			continue;
		}
		if vert_processed[vert as usize] {
			vert_pos.push(vert_pos[vert as usize]);
			vert = (vert_pos.len() - 1) as i32;
			halfedge.for_vert_mut(i, |halfedge, current| {
				halfedge_processed[current as usize] = true;
				halfedge.set_start(current, vert);
				halfedge.set_end(halfedge.pair(current), vert);
			});
		} else {
			vert_processed[vert as usize] = true;
			halfedge.for_vert(i, |current| {
				halfedge_processed[current as usize] = true;
			});
		}
	}
}

///Dereference duplicate property vertices if they are exactly floating-point
///equal. These unreferenced properties are then removed by CompactProps.
pub fn dedupe_prop_verts(
	halfedge: &mut Halfedges,
	tri_rel: &[TriRelation],
	properties: &Properties,
) {
	let prop_stride = properties.stride;
	if prop_stride == 0 {
		return;
	}

	let mut vert2vert: Vec<(i32, i32)> = vec![(-1, -1); halfedge.len()];
	for edge_idx in 0..halfedge.len() {
		let pair = halfedge.pair(edge_idx as i32);
		if pair < 0 {
			continue;
		}
		let edge_face = edge_idx / 3;
		let pair_face = pair / 3;

		if tri_rel[edge_face].instance_id != tri_rel[pair_face as usize].instance_id {
			continue;
		}

		let prop0 = halfedge.prop(edge_idx as i32);
		let prop1 = halfedge.prop(next_halfedge(pair));
		let mut prop_equal = true;
		for p in 0..prop_stride {
			if properties.data[prop_stride * prop0 as usize + p]
				!= properties.data[prop_stride * prop1 as usize + p]
			{
				prop_equal = false;
				break;
			}
		}
		if prop_equal {
			vert2vert[edge_idx] = (prop0, prop1);
		}
	}

	let num_prop_vert = properties.data.len() / prop_stride;

	let uf = DisjointSets::new(num_prop_vert);
	for edge in vert2vert {
		if edge.0 == -1 || edge.1 == -1 {
			continue;
		}
		uf.unite(edge.0 as usize, edge.1 as usize);
	}

	let (vert_labels, num_labels) = uf.connected_components();

	let mut label2vert: Vec<i32> = vec![0; num_labels];
	for v in 0..num_prop_vert {
		label2vert[vert_labels[v] as usize] = v as i32;
	}
	for edge in 0..halfedge.len() as i32 {
		halfedge.set_prop(
			edge,
			label2vert[vert_labels[halfedge.prop(edge) as usize] as usize],
		);
	}
}

pub fn set_normals_and_coplanar(
	tri_rel: &mut [TriRelation],
	halfedge: &Halfedges,
	vert_pos: &[Point3<f64>],
	tolerance: f64,
) -> Vec<Vector3<f64>> {
	let num_tri = halfedge.num_tri();
	let mut tri_normal = unsafe { vec_ext::uninit(num_tri) };
	struct TriPriority {
		area2: f64,
		tri: i32,
	}
	let mut tri_priority = unsafe { vec_ext::uninit(num_tri) };
	for tri in 0..num_tri {
		if halfedge.start((3 * tri) as i32) < 0 {
			tri_priority[tri] = TriPriority {
				area2: 0.0,
				tri: tri as i32,
			};
			continue;
		}

		let v = vert_pos[halfedge.start((3 * tri) as i32) as usize];
		let n = (vert_pos[halfedge.end(3 * (tri as i32)) as usize] - v)
			.cross(&(vert_pos[halfedge.end((3 * tri + 1) as i32) as usize] - v));
		tri_normal[tri] = n.normalize();
		if tri_normal[tri].x.is_nan() {
			tri_normal[tri] = Vector3::new(0.0, 0.0, 1.0);
		}
		tri_priority[tri] = TriPriority {
			area2: n.magnitude_squared(),
			tri: tri as i32,
		};
	}

	tri_priority.sort_by_key(|t| Reverse(OrderedF64(t.area2)));

	let mut interior_halfedges: Vec<i32> = Vec::default();
	for tp in &tri_priority {
		if tri_rel[tp.tri as usize].coplanar_id >= 0 {
			continue;
		}

		tri_rel[tp.tri as usize].coplanar_id = tp.tri;
		if halfedge.start(3 * tp.tri) < 0 {
			continue;
		}
		let base = vert_pos[halfedge.start(3 * tp.tri) as usize];
		let normal = tri_normal[tp.tri as usize];
		interior_halfedges.resize(3, 0);
		interior_halfedges[0] = 3 * tp.tri;
		interior_halfedges[1] = 3 * tp.tri + 1;
		interior_halfedges[2] = 3 * tp.tri + 2;
		while !interior_halfedges.is_empty() {
			let h = next_halfedge(halfedge.pair(interior_halfedges.pop().unwrap()));
			if tri_rel[(h / 3) as usize].coplanar_id >= 0 {
				continue;
			}

			let v = vert_pos[halfedge.end(h) as usize];
			if (v - base).dot(&normal).abs() < tolerance {
				let tri = (h / 3) as usize;
				tri_rel[tri].coplanar_id = tp.tri;
				tri_normal[tri] = normal;

				if interior_halfedges.is_empty()
					|| h != halfedge.pair(*interior_halfedges.last().unwrap())
				{
					interior_halfedges.push(h);
				} else {
					interior_halfedges.pop().unwrap();
				}

				let h_next = next_halfedge(h);
				interior_halfedges.push(h_next);
			}
		}
	}

	tri_normal
}

///the mesh is checked for duplicate edges (more than
///one pair of triangles sharing the same edge), which are removed by
///duplicating one vert and adding two triangles. These degenerate triangles are
///likely to be collapsed again in the subsequent simplification.
pub fn dedupe_edges(tri: &mut Triangles, vert_pos: &mut Vec<Point3<f64>>) {
	loop {
		let nb_edges = tri.halfedge.len();
		let mut duplicates = Vec::<usize>::new();
		let local_loop =
			|start: usize, end: usize, local: &mut Vec<bool>, results: &mut Vec<usize>| {
				// Iterate over all halfedges that start with the same vertex, and check
				// for halfedges with the same ending vertex.
				// Note: we use Vec and linear search when the number of neighbor is
				// small because unordered_set requires allocations and is expensive.
				// We switch to unordered_set when the number of neighbor is
				// larger to avoid making things quadratic.
				// We do it in two pass, the first pass to find the minimal halfedges with
				// the target start and end verts, the second pass flag all the duplicated
				// halfedges that are not having the minimal index as duplicates.
				// This ensures deterministic result.
				//
				// The local store is to store the processed halfedges, so to avoid
				// repetitive processing. Note that it only approximates the processed
				// halfedges because it is thread local.
				let mut end_verts: Vec<(i32, i32)> = Vec::new();
				let mut end_vert_set: DeterministicMap<i32, i32> = DeterministicMap::new();
				for i in start..end {
					if local[i] {
						continue;
					}
					let start_vert = tri.halfedge.start(i as i32);
					let end_vert = tri.halfedge.end(i as i32);
					if start_vert == -1 || end_vert == -1 {
						continue;
					}
					// we want to keep the allocation
					end_verts.clear();
					end_vert_set.clear();

					// first iteration, populate entries
					// this makes sure we always report the same set of entries
					tri.halfedge.for_vert(i as i32, |current| {
						local[current as usize] = true;
						let start_vert = tri.halfedge.start(current);
						let end_v = tri.halfedge.end(current);
						if start_vert == -1 || end_v == -1 {
							return;
						}
						if end_vert_set.is_empty() {
							let iter = end_verts.iter_mut().find(|pair| pair.0 == end_v);

							if let Some(iter) = iter {
								iter.1 = iter.1.min(current);
							} else {
								end_verts.push((end_v, current));
								if end_verts.len() > 32 {
									for &(k, v) in end_verts.iter() {
										end_vert_set.entry(k).or_insert(v);
									}

									end_verts.clear();
								}
							}
						} else {
							let pair = match end_vert_set.entry(end_v) {
								Entry::Vacant(entry) => (entry.insert(current), true),
								Entry::Occupied(entry) => (entry.into_mut(), false),
							};

							if !pair.1 {
								*pair.0 = (*pair.0).min(current);
							}
						}
					});

					// second iteration, actually check for duplicates
					// we always report the same set of duplicates, excluding the smallest
					// halfedge in the set of duplicates
					tri.halfedge.for_vert(i as i32, |current| {
						let start_vert = tri.halfedge.start(current);
						let end_v = tri.halfedge.end(current);
						if start_vert == -1 || end_v == -1 {
							return;
						}
						if end_vert_set.is_empty() {
							let iter = end_verts.iter().find(|pair| pair.0 == end_v).unwrap();

							if iter.1 != current {
								results.push(current as usize);
							}
						} else {
							let iter = *end_vert_set.get(&end_v).unwrap();
							if iter != current {
								results.push(current as usize);
							}
						}
					});
				}
			};

		{
			let mut local = vec![false; nb_edges];
			local_loop(0, nb_edges, &mut local, &mut duplicates);
		}

		let mut num_flagged = 0;
		for i in duplicates {
			edge::dedupe(vert_pos, tri, i as i32);
			num_flagged += 1;
		}

		if num_flagged == 0 {
			break;
		}
	}
}

///Collapses degenerate triangles by removing edges shorter than tolerance_ and
///any edge that is preceeded by an edge that joins the same two face relations.
///
///Note when an edge collapse would result in something non-manifold, the
///vertices are duplicated in such a way as to remove handles or separate
///meshes, thus decreasing the Genus(). It only increases when meshes that have
///collapsed to just a pair of triangles are removed entirely.
///
///Verts with index less than firstNewVert will be left uncollapsed.
pub fn collapse_short_edges(
	halfedge: &mut Halfedges,
	vert_pos: &mut Vec<Point3<f64>>,
	tri_normal: &[Vector3<f64>],
	tri_rel: &[TriRelation],
	prop_stride: usize,
	precision: Precision,
	first_new_vert: i32,
) {
	let mut s = FlagStore::default();
	let mut num_flagged = 0;
	let nb_edges = halfedge.len();

	let mut scratch_buffer = Vec::with_capacity(10);
	// Short edges get to skip several checks and hence remove more classes of
	// degenerate triangles than flagged edges do, but this could in theory lead
	// to error stacking where a vertex moves too far. For this reason this is
	// restricted to epsilon, rather than tolerance. However, in the case of a
	// Boolean operation, we set firstNewVert in order to only operate on
	// newly-created verts, which means error stacking is not a concern, so we
	// allow collapsing up to tolerance in that case.
	let tol = if first_new_vert == 0 {
		precision.epsilon
	} else {
		precision.tolerance
	};

	let short_edge = |(halfedge, vert_pos): &mut (&mut Halfedges, &mut Vec<Point3<f64>>), edge| {
		let edge = edge as i32;
		let pair = halfedge.pair(edge);
		if pair < 0 {
			return false;
		}
		let start = halfedge.start(edge);
		let end = halfedge.end(edge);
		if start < first_new_vert && end < first_new_vert {
			return false;
		}
		// Flag short edges
		let delta = vert_pos[end as usize] - vert_pos[start as usize];
		let len_sq = delta.magnitude_squared();
		// To ensure tolerance_-scale errors don't stack, only collapse these edges
		// if they connect a new vert to an old vert, since old verts are only
		// allowed to move by epsilon_.
		let max_len = if end < first_new_vert {
			tol * tol
		} else {
			precision.epsilon * precision.epsilon
		};
		len_sq < max_len
	};

	s.run(
		(halfedge, vert_pos),
		nb_edges,
		short_edge,
		|(halfedge, vert_pos), i| {
			let did_collapse = edge::collapse(
				i as i32,
				halfedge,
				tri_normal,
				tri_rel,
				vert_pos,
				&mut scratch_buffer,
				prop_stride,
				Precision {
					epsilon: precision.epsilon,
					tolerance: tol,
				},
				first_new_vert,
			);
			if did_collapse {
				num_flagged += 1;
			}
			scratch_buffer.truncate(0);
		},
	);
}

pub fn collapse_colinear_edges(
	halfedge: &mut Halfedges,
	vert_pos: &mut Vec<Point3<f64>>,
	tri_normal: &[Vector3<f64>],
	tri_rel: &[TriRelation],
	prop_stride: usize,
	epsilon: f64,
	first_new_vert: i32,
) {
	let mut s = FlagStore::default();
	let nb_edges = halfedge.len();
	let mut scratch_buffer = Vec::with_capacity(10);
	loop {
		//CollapseFlaggedEdge
		let mut num_flagged = 0;
		// Collapse colinear edges, but only remove new verts, i.e. verts with
		// index
		// >= firstNewVert. This is used to keep the Boolean from changing the
		// non-intersecting parts of the input meshes. Colinear is defined not by a
		// local check, but by the global MarkCoplanar function, which keeps this
		// from being vulnerable to error stacking.
		let colinear_edge = |(halfedge, _): &mut (&mut Halfedges, &mut Vec<Point3<f64>>), edge| {
			let edge = edge as i32;
			let pair = halfedge.pair(edge);
			if pair < 0 || halfedge.start(edge) < first_new_vert {
				return false;
			}
			// Flag redundant edges - those where the startVert is surrounded by only
			// two original triangles.
			let ref0 = tri_rel[(edge / 3) as usize];
			let mut current = next_halfedge(pair);
			let mut ref1 = tri_rel[(current / 3) as usize];
			let mut ref1_updated = !ref0.same_face(&ref1);
			while current != edge {
				current = next_halfedge(halfedge.pair(current));
				let tri = current / 3;
				let tri_rel = tri_rel[tri as usize];
				if !tri_rel.same_face(&ref0) && !tri_rel.same_face(&ref1) {
					if !ref1_updated {
						ref1 = tri_rel;
						ref1_updated = true;
					} else {
						return false;
					}
				}
			}

			true
		};

		s.run(
			(&mut *halfedge, &mut *vert_pos),
			nb_edges,
			colinear_edge,
			|(halfedge, vert_pos), i| {
				let did_collapse = edge::collapse(
					i as i32,
					halfedge,
					tri_normal,
					tri_rel,
					vert_pos,
					&mut scratch_buffer,
					prop_stride,
					Precision {
						epsilon,
						tolerance: epsilon,
					},
					0,
				);
				if did_collapse {
					num_flagged += 1;
				}
				scratch_buffer.truncate(0);
			},
		);

		if num_flagged == 0 {
			break;
		}
	}
}

//performs edge swaps on the long edges of degenerate triangles, though
///there are some configurations of degenerates that cannot be removed this way.
pub fn swap_degenerates(
	tri: &mut Triangles,
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	precision: Precision,
	first_new_vert: i32,
) {
	//RecursiveEdgeSwap
	let mut s = FlagStore::default();
	let mut num_flagged = 0;
	let nb_edges = tri.halfedge.len();
	let mut scratch_buffer = Vec::with_capacity(10);

	let swappable_edge = |(tri, vert_pos): &mut (&mut Triangles, &mut Vec<Point3<f64>>),
	                      edge|
	 -> bool {
		let mut edge = edge as i32;
		let pair = tri.halfedge.pair(edge);
		if pair < 0 {
			return false;
		}
		let tri_edge = edge::tri_of(edge);
		let pair_tri_edge = edge::tri_of(pair);
		if tri.halfedge.start(tri_edge[0]) < first_new_vert
			&& tri.halfedge.start(tri_edge[1]) < first_new_vert
			&& tri.halfedge.start(tri_edge[2]) < first_new_vert
			&& tri.halfedge.start(pair_tri_edge[2]) < first_new_vert
		{
			return false;
		}

		let mut tri_idx = edge / 3;
		let mut projection = get_axis_aligned_projection(tri.normal[tri_idx as usize]);
		let mut v = [Point2::<f64>::default(); 3];
		for i in 0..3 {
			v[i] = projection * vert_pos[tri.halfedge.start(tri_edge[i]) as usize];
		}
		if ccw(v[0], v[1], v[2], precision.tolerance) > 0 || !edge::is_01_longest(v[0], v[1], v[2])
		{
			return false;
		}

		// Switch to neighbor's projection.
		edge = pair;
		tri_idx = edge / 3;
		projection = get_axis_aligned_projection(tri.normal[tri_idx as usize]);
		for i in 0..3 {
			v[i] = projection * vert_pos[tri.halfedge.start(pair_tri_edge[i]) as usize];
		}

		ccw(v[0], v[1], v[2], precision.tolerance) > 0 || edge::is_01_longest(v[0], v[1], v[2])
	};

	let mut edge_swap_stack = Vec::new();
	let mut visited = vec![-1; tri.halfedge.len()];
	let mut tag = 0;
	s.run(
		(tri, vert_pos),
		nb_edges,
		swappable_edge,
		|(tri, vert_pos), i| {
			num_flagged += 1;
			tag += 1;
			edge::recursive_swap(
				i as i32,
				tri,
				vert_pos,
				properties,
				&mut tag,
				&mut visited,
				&mut edge_swap_stack,
				&mut scratch_buffer,
				precision,
			);
			while !edge_swap_stack.is_empty() {
				let last = edge_swap_stack.pop().unwrap();
				edge::recursive_swap(
					last,
					tri,
					vert_pos,
					properties,
					&mut tag,
					&mut visited,
					&mut edge_swap_stack,
					&mut scratch_buffer,
					precision,
				);
			}
		},
	);
}

#[derive(Default)]
struct FlagStore {
	s: Vec<usize>,
}

impl FlagStore {
	fn run<Mut>(
		&mut self,
		borrow: Mut,
		n: usize,
		pred: impl FnMut(&mut Mut, usize) -> bool,
		f: impl FnMut(&mut Mut, usize),
	) {
		self.run_seq(borrow, n, pred, f)
	}

	fn run_seq<Mut>(
		&mut self,
		mut borrow: Mut,
		n: usize,
		mut pred: impl FnMut(&mut Mut, usize) -> bool,
		mut f: impl FnMut(&mut Mut, usize),
	) {
		for i in 0..n {
			if pred(&mut borrow, i) {
				self.s.push(i);
			}
		}

		for &i in &self.s {
			f(&mut borrow, i);
		}
		self.s = Vec::default();
	}
}

pub fn mark_unreferenced_verts(halfedge: &Halfedges, vert_pos: &mut [Point3<f64>]) {
	let mut keep = vec![false; vert_pos.len()];
	for edge in 0..halfedge.len() {
		let start_vert = halfedge.start(edge as i32);
		if start_vert >= 0 {
			keep[start_vert as usize] = true;
		}
	}

	for v in 0..vert_pos.len() {
		if keep[v] == false {
			vert_pos[v] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
		}
	}
}
