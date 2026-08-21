pub use crate::postprocessing::sort::sort_and_compact_geometry;

use crate::halfedge::{Halfedges, next_halfedge};
use crate::mesh_relations::{InstanceRelation, TriRelation};
use crate::postprocessing::edge::Merger;
use crate::util::disjoint_sets::DisjointSets;
use crate::util::math::{ccw, get_axis_aligned_projection, safe_normalize3};
use crate::util::num_convert::OrderedF64;
use crate::util::vec_ext;
use crate::{Precision, Properties, TrianglesWIP};
use nalgebra::{Point2, Point3, Vector3};
use rustc_hash::FxHashMap;
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
	let nb_edges = halfedge.len();

	let mut vert_processed = vec![false; vert_pos.len()];
	let mut halfedge_processed = vec![false; nb_edges];
	for i in 0..nb_edges {
		if halfedge_processed[i] {
			continue;
		}
		let mut vert = halfedge.start[i];
		if vert == -1 {
			continue;
		}
		if vert_processed[vert as usize] {
			vert_pos.push(vert_pos[vert as usize]);
			vert = (vert_pos.len() - 1) as i32;
			halfedge.for_vert_mut(i, |halfedge, current| {
				halfedge_processed[current] = true;
				halfedge.start[current] = vert;
				halfedge.set_end(halfedge.pair[current] as usize, vert);
			});
		} else {
			vert_processed[vert as usize] = true;
			halfedge.for_vert(i, |current| {
				halfedge_processed[current] = true;
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
		let pair = halfedge.pair[edge_idx];
		if pair < 0 {
			continue;
		}
		let pair = pair as usize;
		let edge_face = edge_idx / 3;
		let pair_face = pair / 3;

		if tri_rel[edge_face].instance_id != tri_rel[pair_face].instance_id {
			continue;
		}

		let prop0 = halfedge.prop[edge_idx];
		let prop1 = halfedge.prop[next_halfedge(pair)];
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

	let mut uf = DisjointSets::new(num_prop_vert);
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
	for edge in 0..halfedge.len() {
		halfedge.prop[edge] = label2vert[vert_labels[halfedge.prop[edge] as usize] as usize];
	}
}

pub fn set_normals_and_coplanar(
	tri_rel: &mut [TriRelation],
	instance_rel: &[InstanceRelation],
	halfedge: &Halfedges,
	vert_pos: &[Point3<f64>],
	tolerance: f64,
) -> Vec<Vector3<f64>> {
	let num_tri = halfedge.num_tri();
	struct TriPriority {
		area2: f64,
		tri: i32,
	}

	let (tri_normal, mut tri_priority): (Vec<_>, Vec<_>) = (0..num_tri)
		.map(|tri| {
			if halfedge.start[3 * tri] < 0 {
				return (
					Vector3::new(0.0, 0.0, 1.0),
					TriPriority {
						area2: 0.0,
						tri: tri as i32,
					},
				);
			}

			let v = vert_pos[halfedge.start[3 * tri] as usize];
			let n = (vert_pos[halfedge.end(3 * tri) as usize] - v)
				.cross(&(vert_pos[halfedge.end(3 * tri + 1) as usize] - v));

			let priority = TriPriority {
				area2: n.magnitude_squared(),
				tri: tri as i32,
			};

			(safe_normalize3(n), priority)
		})
		.unzip();

	tri_priority.sort_unstable_by_key(|t| Reverse(OrderedF64(t.area2)));

	let mut coplanar_id = vec![-1; num_tri];
	let mut interior_halfedges: Vec<i32> = Vec::default();
	for tp in tri_priority {
		if coplanar_id[tp.tri as usize] >= 0 {
			continue;
		}

		coplanar_id[tp.tri as usize] = tp.tri;
		if halfedge.start[3 * (tp.tri as usize)] < 0 {
			continue;
		}
		let base = vert_pos[halfedge.start[3 * (tp.tri as usize)] as usize];
		let normal = tri_normal[tp.tri as usize];
		interior_halfedges.resize(3, 0);
		interior_halfedges[0] = 3 * tp.tri;
		interior_halfedges[1] = 3 * tp.tri + 1;
		interior_halfedges[2] = 3 * tp.tri + 2;
		while !interior_halfedges.is_empty() {
			let h =
				next_halfedge(halfedge.pair[interior_halfedges.pop().unwrap() as usize] as usize);
			if coplanar_id[h / 3] >= 0 {
				continue;
			}

			let v = vert_pos[halfedge.end(h) as usize];
			if (v - base).dot(&normal).abs() < tolerance {
				let tri = h / 3;
				coplanar_id[tri] = tp.tri;

				if interior_halfedges.is_empty()
					|| h != halfedge.pair[*interior_halfedges.last().unwrap() as usize] as usize
				{
					interior_halfedges.push(h as i32);
				} else {
					interior_halfedges.pop().unwrap();
				}

				let h_next = next_halfedge(h);
				interior_halfedges.push(h_next as i32);
			}
		}
	}

	//assign coplanar id as face id if user didn't provide a face id
	for (tri, coplanar_id) in coplanar_id.into_iter().enumerate() {
		let tri_rel = &mut tri_rel[tri];
		if !instance_rel[tri_rel.instance_id as usize].user_provided_face_id {
			tri_rel.face_id = coplanar_id;
		}
	}

	tri_normal
}

///the mesh is checked for duplicate edges (more than
///one pair of triangles sharing the same edge), which are removed by
///duplicating one vert and adding two triangles. These degenerate triangles are
///likely to be collapsed again in the subsequent simplification.
pub fn dedupe_edges(tri: &mut TrianglesWIP, vert_pos: &mut Vec<Point3<f64>>) {
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
				let mut end_vert_set: FxHashMap<i32, i32> = FxHashMap::default();
				for i in start..end {
					if local[i] {
						continue;
					}
					let start_vert = tri.halfedge.start[i];
					let end_vert = tri.halfedge.end(i);
					if start_vert == -1 || end_vert == -1 {
						continue;
					}
					// we want to keep the allocation
					end_verts.clear();
					end_vert_set.clear();

					// first iteration, populate entries
					// this makes sure we always report the same set of entries
					tri.halfedge.for_vert(i, |current| {
						local[current] = true;
						let start_vert = tri.halfedge.start[current];
						let end_v = tri.halfedge.end(current);
						if start_vert == -1 || end_v == -1 {
							return;
						}
						if end_vert_set.is_empty() {
							let iter = end_verts.iter_mut().find(|pair| pair.0 == end_v);

							if let Some(iter) = iter {
								iter.1 = iter.1.min(current as i32);
							} else {
								end_verts.push((end_v, current as i32));
								if end_verts.len() > 32 {
									for &(k, v) in end_verts.iter() {
										end_vert_set.entry(k).or_insert(v);
									}

									end_verts.clear();
								}
							}
						} else {
							let pair = match end_vert_set.entry(end_v) {
								Entry::Vacant(entry) => (entry.insert(current as i32), true),
								Entry::Occupied(entry) => (entry.into_mut(), false),
							};

							if !pair.1 {
								*pair.0 = (*pair.0).min(current as i32);
							}
						}
					});

					// second iteration, actually check for duplicates
					// we always report the same set of duplicates, excluding the smallest
					// halfedge in the set of duplicates
					tri.halfedge.for_vert(i, |current| {
						let start_vert = tri.halfedge.start[current];
						let end_v = tri.halfedge.end(current);
						if start_vert == -1 || end_v == -1 {
							return;
						}
						if end_vert_set.is_empty() {
							let iter = end_verts.iter().find(|pair| pair.0 == end_v).unwrap();

							if iter.1 != current as i32 {
								results.push(current);
							}
						} else {
							let iter = *end_vert_set.get(&end_v).unwrap();
							if iter != current as i32 {
								results.push(current);
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
			edge::dedupe(vert_pos, tri, i);
			num_flagged += 1;
		}

		if num_flagged == 0 {
			break;
		}
	}
}

pub fn simplify_topology2(
	tri: &mut TrianglesWIP,
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	instance_rel: &[InstanceRelation],
	precision: Precision,
) {
	let mut edges = Vec::from_iter(0..tri.halfedge.len() as i32);
	let mut edges_end = edges.len();
	let mut verts_visited = vec![false; vert_pos.len()];
	let mut total_cost = vec![0.0_f64; vert_pos.len()];
	let mut merger = vec![Merger::default(); edges.len()];
	let max_cost = precision.max_cost();
	let mut scratch_buffer = Vec::with_capacity(10);

	while 0 != edges_end {
		for &edge in edges[..edges_end].iter() {
			let edge = edge as usize;
			let pair = tri.halfedge.pair[edge] as usize;
			if !tri.halfedge.valid(edge) {
				continue;
			}
			// Optimization: only calculate for forward halfedges, then copy
			// result to the pair. However, this conflicts with the above
			// optimization because forward halfedges with two retained verts
			// get discarded, but can later become edges with new verts,
			// which are then needed.
			if !tri.halfedge.is_forward(edge) {
				continue;
			}

			if merger[edge].valid()
				&& !verts_visited[tri.halfedge.start[edge] as usize]
				&& !verts_visited[tri.halfedge.end(edge) as usize]
				&& !verts_visited[tri.halfedge.end(next_halfedge(edge)) as usize]
				&& !verts_visited[tri.halfedge.end(next_halfedge(pair)) as usize]
			{
				continue;
			}

			// Swappable edges differ on forward and backward, so check before the
			// forward-only optimization.
			let swap_edge = edge::swappable(
				edge,
				&tri.halfedge,
				&tri.normal,
				vert_pos,
				precision.epsilon,
			);
			if swap_edge
				|| edge::swappable(
					pair,
					&tri.halfedge,
					&tri.normal,
					vert_pos,
					precision.epsilon,
				) {
				let v0 = vert_pos[tri.halfedge.start[edge] as usize];
				let l01 = (vert_pos[tri.halfedge.end(edge) as usize] - v0).magnitude();
				let l02 =
					(vert_pos[tri
						.halfedge
						.end(next_halfedge(if swap_edge { edge } else { pair }))
						as usize] - v0)
						.magnitude();
				let a = 0.0_f64.max(1.0_f64.min(l02 / l01));
				merger[if swap_edge { edge } else { pair }] = Merger {
					added_cost: 0.0,
					total_cost: Merger::K_SWAP,
					a: if swap_edge { a } else { 1.0 - a },
					new_pos: Point3::new(f64::NAN, f64::NAN, f64::NAN),
				};
				continue;
			}

			// Optimization: only recalculate when an edge has collapsed into
			// this one. Technically its cost can also change from a
			// neighbor's collapse, but probably not enough to worry about,
			// and this is a much cheaper check.
			if merger[edge].valid()
				&& !verts_visited[tri.halfedge.start[edge] as usize]
				&& !verts_visited[tri.halfedge.end(edge) as usize]
			{
				continue;
			}

			let mut edge_cost = edge::check(
				edge,
				tri,
				vert_pos,
				instance_rel,
				properties.stride,
				precision.epsilon,
			);
			edge_cost.total_cost += total_cost[tri.halfedge.start[edge] as usize]
				.max(total_cost[tri.halfedge.end(edge) as usize]);
			merger[edge] = edge_cost;
			// Forward edge optimization is enabled, so copy the result to
			// the pair.
			edge_cost.a = 1.0 - edge_cost.a;
			merger[pair] = edge_cost;
		}

		edges[..edges_end]
			.sort_unstable_by_key(|&edge| OrderedF64(merger[edge as usize].total_cost));
		verts_visited.iter_mut().for_each(|v| *v = false);
		let mut edges_itr = 0;
		let mut num_collapsed: usize = 0;
		let mut num_swapped: usize = 0;
		// Collapse short edges first so that long edges calculate correct cost.
		let short_collapse = merger[edges[edges_itr] as usize].short();
		let mut increment_itr = false;
		loop {
			if increment_itr {
				edges_itr += 1;
			}
			if edges_itr == edges_end {
				break;
			}
			increment_itr = true;

			let edge = edges[edges_itr] as usize;
			if !tri.halfedge.valid(edge) {
				continue;
			}
			if merger[edge].total_cost > max_cost {
				break; // Sorting means no further edges can be collapsed this round.
			}
			if short_collapse && !merger[edge].free() {
				break; // force recalculation of cost after free edges collapse.
			}
			let start_v = tri.halfedge.start[edge] as usize;
			let end_v = tri.halfedge.end(edge) as usize;
			// Allow short merges to stack to ensure all are collapsed.
			if !short_collapse && (verts_visited[start_v] || verts_visited[end_v]) {
				continue;
			}
			if merger[edge].swap() {
				verts_visited[start_v] = true;
				verts_visited[end_v] = true;
				verts_visited[tri.halfedge.end(next_halfedge(edge)) as usize] = true;
				verts_visited[tri
					.halfedge
					.end(next_halfedge(tri.halfedge.pair[edge] as usize))
					as usize] = true;
				edge::swap(edge, merger[edge].a, tri, vert_pos, properties);
				verts_visited.resize(vert_pos.len(), true);
				total_cost.resize(vert_pos.len(), 0.0);
				num_swapped += 1;
				continue;
			}
			let did_collapse = edge::collapse2(
				edge,
				&mut tri.halfedge,
				&tri.normal,
				vert_pos,
				properties,
				&mut scratch_buffer,
				merger[edge],
				precision.epsilon,
			);
			verts_visited.resize(vert_pos.len(), true);
			total_cost.resize(vert_pos.len(), 0.0);
			if did_collapse {
				total_cost[start_v] += merger[edge].added_cost;
				total_cost[end_v] += merger[edge].added_cost;
				verts_visited[start_v] = true;
				verts_visited[end_v] = true;
				num_collapsed += 1;
			}
		}
		edges_end = vec_ext::unstable_partition(&mut edges[..edges_end], |&edge| {
			tri.halfedge.valid(edge as usize)
		});
		if num_collapsed == 0 && num_swapped == 0 {
			break;
		}

		for tri_i in 0..tri.halfedge.num_tri() {
			if !tri.halfedge.valid(3 * tri_i) {
				continue;
			}
			let mut update = false;
			for i in 0..3 {
				update |= verts_visited[tri.halfedge.start[3 * tri_i + i] as usize];
			}
			if !update {
				continue;
			}

			let center = vert_pos[tri.halfedge.start[3 * tri_i] as usize];
			tri.normal[tri_i] = safe_normalize3(
				(vert_pos[tri.halfedge.start[3 * tri_i + 1] as usize] - center)
					.cross(&(vert_pos[tri.halfedge.start[3 * tri_i + 2] as usize] - center)),
			);
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
	instance_rel: &[InstanceRelation],
	prop_stride: usize,
	precision: Precision,
	first_new_vert: usize,
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
		let pair = halfedge.pair[edge];
		if pair < 0 {
			return false;
		}
		let start = halfedge.start[edge] as usize;
		let end = halfedge.end(edge) as usize;
		if start < first_new_vert && end < first_new_vert {
			return false;
		}
		// Flag short edges
		let delta = vert_pos[end] - vert_pos[start];
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
				i,
				halfedge,
				tri_normal,
				tri_rel,
				instance_rel,
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
	instance_rel: &[InstanceRelation],
	prop_stride: usize,
	epsilon: f64,
	first_new_vert: usize,
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
		// non-intersecting parts of the input meshes. Colinear is defined not by
		// a local check, but by the global MarkCoplanar function, which keeps
		// this from being vulnerable to error stacking.
		let colinear_edge = |(halfedge, _): &mut (&mut Halfedges, &mut Vec<Point3<f64>>), edge| {
			let pair = halfedge.pair[edge];
			if pair < 0 || (halfedge.start[edge] as usize) < first_new_vert {
				return false;
			}
			// Flag redundant edges - those where the startVert is surrounded by
			// only two original triangles.
			let ref0 = tri_rel[edge / 3];
			let mut current = next_halfedge(pair as usize);
			let mut ref1 = tri_rel[current / 3];
			let mut ref1_updated = ref0 != ref1;
			while current != edge {
				current = next_halfedge(halfedge.pair[current] as usize);
				let tri = current / 3;
				let tri_rel = tri_rel[tri];
				if tri_rel != ref0 && tri_rel != ref1 {
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
					i,
					halfedge,
					tri_normal,
					tri_rel,
					instance_rel,
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
	tri: &mut TrianglesWIP,
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	instance_rel: &[InstanceRelation],
	precision: Precision,
	first_new_vert: usize,
) {
	//RecursiveEdgeSwap
	let mut s = FlagStore::default();
	let mut num_flagged = 0;
	let nb_edges = tri.halfedge.len();
	let mut scratch_buffer = Vec::with_capacity(10);

	let swappable_edge = |(tri, vert_pos): &mut (&mut TrianglesWIP, &mut Vec<Point3<f64>>),
	                      mut edge|
	 -> bool {
		let pair = tri.halfedge.pair[edge];
		if pair < 0 {
			return false;
		}
		let pair = pair as usize;
		let tri_edge = edge::tri_of(edge);
		let pair_tri_edge = edge::tri_of(pair);
		if (tri.halfedge.start[tri_edge[0] as usize] as usize) < first_new_vert
			&& (tri.halfedge.start[tri_edge[1] as usize] as usize) < first_new_vert
			&& (tri.halfedge.start[tri_edge[2] as usize] as usize) < first_new_vert
			&& (tri.halfedge.start[pair_tri_edge[2] as usize] as usize) < first_new_vert
		{
			return false;
		}

		let mut tri_idx = edge / 3;
		let mut projection = get_axis_aligned_projection(tri.normal[tri_idx]);
		let mut v = [Point2::<f64>::default(); 3];
		for i in 0..3 {
			v[i] = projection * vert_pos[tri.halfedge.start[tri_edge[i] as usize] as usize];
		}
		if ccw(v[0], v[1], v[2], precision.tolerance) > 0
			|| !edge::is_01_longest_2(v[0], v[1], v[2])
		{
			return false;
		}

		// Switch to neighbor's projection.
		edge = pair;
		tri_idx = edge / 3;
		projection = get_axis_aligned_projection(tri.normal[tri_idx]);
		for i in 0..3 {
			v[i] = projection * vert_pos[tri.halfedge.start[pair_tri_edge[i] as usize] as usize];
		}

		ccw(v[0], v[1], v[2], precision.tolerance) > 0 || edge::is_01_longest_2(v[0], v[1], v[2])
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
				i,
				tri,
				vert_pos,
				properties,
				&mut tag,
				&mut visited,
				&mut edge_swap_stack,
				&mut scratch_buffer,
				instance_rel,
				precision,
			);
			while !edge_swap_stack.is_empty() {
				let last = edge_swap_stack.pop().unwrap();
				// The stack is fed from halfedge.pair, which uses -1 as the
				// "unpaired" sentinel. recursive_swap used to filter these out
				// itself; now that it takes a usize, the check has to happen
				// before the cast or -1 becomes usize::MAX.
				if last < 0 {
					continue;
				}
				edge::recursive_swap(
					last as usize,
					tri,
					vert_pos,
					properties,
					&mut tag,
					&mut visited,
					&mut edge_swap_stack,
					&mut scratch_buffer,
					instance_rel,
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
		let start_vert = halfedge.start[edge];
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
