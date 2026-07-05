use crate::halfedge::{Halfedges, next_halfedge};
use crate::mesh_relations::TriRelation;
use crate::util::math::{ccw, get_axis_aligned_projection};
use crate::{Precision, Properties, Triangles};
use nalgebra::{Point2, Point3, Vector3, distance};
use std::f64;

// Deduplicate the given 4-manifold edge by duplicating endVert, thus making the
// edges distinct. Also duplicates startVert if it becomes pinched.
pub fn dedupe(vert_pos: &mut Vec<Point3<f64>>, tri: &mut Triangles, edge: i32) {
	// Orbit endVert
	let next_edge = next_halfedge(edge);
	let start_vert = tri.halfedge.start(edge);
	let end_vert = tri.halfedge.start(next_edge);
	let end_prop = tri.halfedge.prop(next_edge);
	let mut current = tri.halfedge.pair(next_edge);
	while current != edge {
		let vert = tri.halfedge.start(current);
		if vert == start_vert {
			// Single topological unit needs 2 faces added to be split
			let new_vert = vert_pos.len() as i32;
			vert_pos.push(vert_pos[end_vert as usize]);
			current = tri.halfedge.pair(next_halfedge(current));
			let opposite = tri.halfedge.pair(next_halfedge(edge));

			tri.halfedge.update_vert(new_vert, current, opposite);

			let mut new_halfedge = tri.halfedge.len() as i32;
			let mut old_face = current / 3;
			let mut outside_vert = tri.halfedge.start(current);
			tri.halfedge.push(end_vert, -1, end_prop);
			tri.halfedge.push(new_vert, -1, end_prop);
			tri.halfedge
				.push(outside_vert, -1, tri.halfedge.prop(current));
			tri.halfedge
				.pair_up(new_halfedge + 2, tri.halfedge.pair(current));
			tri.halfedge.pair_up(new_halfedge + 1, current);
			tri.relation.push(tri.relation[old_face as usize]);
			tri.normal.push(tri.normal[old_face as usize]);

			new_halfedge += 3;
			old_face = opposite / 3;
			outside_vert = tri.halfedge.start(opposite);
			tri.halfedge.push(new_vert, -1, end_prop); // fix prop
			tri.halfedge.push(end_vert, -1, end_prop);
			tri.halfedge
				.push(outside_vert, -1, tri.halfedge.prop(opposite));
			tri.halfedge
				.pair_up(new_halfedge + 2, tri.halfedge.pair(opposite));
			tri.halfedge.pair_up(new_halfedge + 1, opposite);
			tri.halfedge.pair_up(new_halfedge, new_halfedge - 3);
			tri.relation.push(tri.relation[old_face as usize]);
			tri.normal.push(tri.normal[old_face as usize]);

			break;
		}

		current = tri.halfedge.pair(next_halfedge(current));
	}

	if current == edge {
		// Separate topological unit needs no new faces to be split
		let new_vert = vert_pos.len() as i32;
		vert_pos.push(vert_pos[end_vert as usize]);

		tri.halfedge
			.for_vert_mut(next_halfedge(current), |halfedge, e| {
				halfedge.set_start(e, new_vert);
				halfedge.set_end(halfedge.pair(e), new_vert);
			});
	}

	// Orbit startVert
	let pair = tri.halfedge.pair(edge);
	current = tri.halfedge.pair(next_halfedge(pair));
	while current != pair {
		let vert = tri.halfedge.start(current);
		if vert == end_vert {
			break; //connected: not a pinched vert
		}

		current = tri.halfedge.pair(next_halfedge(current));
	}

	if current == pair {
		// Split the pinched vert the previous split created.
		let new_vert = vert_pos.len() as i32;
		vert_pos.push(vert_pos[end_vert as usize]);

		tri.halfedge
			.for_vert_mut(next_halfedge(current), |halfedge, e| {
				halfedge.set_start(e, new_vert);
				halfedge.set_end(halfedge.pair(e), new_vert);
			});
	}
}

pub fn recursive_swap(
	edge: i32,
	tri: &mut Triangles,
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	tag: &mut i32,
	visited: &mut [i32],
	edge_swap_stack: &mut Vec<i32>,
	edges: &mut Vec<i32>,
	precision: Precision,
) {
	if edge < 0 {
		return;
	}
	let pair = tri.halfedge.pair(edge);
	if pair < 0 {
		return;
	}

	// avoid infinite recursion
	if visited[edge as usize] == *tag && visited[pair as usize] == *tag {
		return;
	}

	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(pair);

	let projection = get_axis_aligned_projection(tri.normal[(edge / 3) as usize]);
	let mut v = [Point2::default(); 4];
	for i in 0..3 {
		v[i] = projection * vert_pos[tri.halfedge.start(tri0_edge[i]) as usize];
	}

	// Only operate on the long edge of a degenerate triangle.
	if ccw(v[0], v[1], v[2], precision.tolerance) > 0 || !is_01_longest(v[0], v[1], v[2]) {
		return;
	}

	// Switch to neighbor's projection.
	let projection = get_axis_aligned_projection(tri.normal[(pair / 3) as usize]);
	for i in 0..3 {
		v[i] = projection * vert_pos[tri.halfedge.start(tri0_edge[i]) as usize];
	}

	v[3] = projection * vert_pos[tri.halfedge.start(tri1_edge[2]) as usize];

	let mut swap_edge = |tri: &mut Triangles| {
		// The 0-verts are swapped to the opposite 2-verts.
		let v0 = tri.halfedge.start(tri0_edge[2]);
		let v1 = tri.halfedge.start(tri1_edge[2]);
		tri.halfedge.set_start(tri0_edge[0], v1);
		tri.halfedge.set_end(tri0_edge[2], v1);
		tri.halfedge.set_start(tri1_edge[0], v0);
		tri.halfedge.set_end(tri1_edge[2], v0);
		tri.halfedge
			.pair_up(tri0_edge[0], tri.halfedge.pair(tri1_edge[2]));
		tri.halfedge
			.pair_up(tri1_edge[0], tri.halfedge.pair(tri0_edge[2]));
		tri.halfedge.pair_up(tri0_edge[2], tri1_edge[2]);
		// Both triangles are now subsets of the neighboring triangle.
		let tri0 = (tri0_edge[0] / 3) as usize;
		let tri1 = (tri1_edge[0] / 3) as usize;
		tri.normal[tri0] = tri.normal[tri1];
		tri.relation[tri0] = tri.relation[tri1];
		let l01 = distance(&v[1], &v[0]);
		let l02 = distance(&v[2], &v[0]);
		let a = (l02 / l01).clamp(0.0, 1.0);
		// Update properties if applicable
		if properties.data.len() > 0 {
			tri.halfedge
				.set_prop(tri0_edge[1], tri.halfedge.prop(tri1_edge[0]));
			tri.halfedge
				.set_prop(tri0_edge[0], tri.halfedge.prop(tri1_edge[2]));
			tri.halfedge
				.set_prop(tri0_edge[2], tri.halfedge.prop(tri1_edge[2]));
			let prop_stride = properties.stride;
			let new_prop = properties.data.len() / prop_stride;
			let prop_idx0 = tri.halfedge.prop(tri1_edge[0]) as usize;
			let prop_idx1 = tri.halfedge.prop(tri1_edge[1]) as usize;
			for p in 0..prop_stride {
				properties.data.push(
					a * properties.data[prop_stride * prop_idx0 + p]
						+ (1.0 - a) * properties.data[prop_stride * prop_idx1 + p],
				);
			}

			tri.halfedge.set_prop(tri1_edge[0], new_prop as i32);
			tri.halfedge.set_prop(tri0_edge[2], new_prop as i32);
		}

		// if the new edge already exists, duplicate the verts and split the mesh.
		let mut current = tri.halfedge.pair(tri1_edge[0]);
		let end_vert = tri.halfedge.end(tri1_edge[1]);
		while current != tri0_edge[1] {
			current = next_halfedge(current);
			if tri.halfedge.end(current) == end_vert {
				form_loop(&mut tri.halfedge, vert_pos, tri0_edge[2], current);
				remove_if_folded(tri0_edge[2], &mut tri.halfedge, vert_pos);
				return;
			}

			current = tri.halfedge.pair(current);
		}
	};

	// Only operate if the other triangles are not degenerate.
	if ccw(v[1], v[0], v[3], precision.tolerance) <= 0 {
		if !is_01_longest(v[1], v[0], v[3]) {
			return;
		}
		// Two facing, long-edge degenerates can swap.
		swap_edge(tri);
		let e23 = v[3] - v[2];
		if e23.magnitude_squared() < precision.tolerance * precision.tolerance {
			*tag += 1;
			collapse(
				tri0_edge[2],
				&mut tri.halfedge,
				&tri.normal,
				&tri.relation,
				vert_pos,
				edges,
				properties.stride,
				Precision {
					epsilon: precision.epsilon,
					tolerance: precision.epsilon,
				},
				0,
			);
			edges.truncate(0);
		} else {
			visited[edge as usize] = *tag;
			visited[pair as usize] = *tag;
			edge_swap_stack.extend([tri1_edge[1], tri1_edge[0], tri0_edge[1], tri0_edge[0]]);
		}

		return;
	} else if ccw(v[0], v[3], v[2], precision.tolerance) <= 0
		|| ccw(v[1], v[2], v[3], precision.tolerance) <= 0
	{
		return;
	}

	//normal path
	swap_edge(tri);
	visited[edge as usize] = *tag;
	visited[pair as usize] = *tag;
	edge_swap_stack.extend([
		tri.halfedge.pair(tri1_edge[0]),
		tri.halfedge.pair(tri0_edge[1]),
	]);
}

///Collapses the given edge by removing startVert - returns false if the edge
///cannot be collapsed. May split the mesh topologically if the collapse would
///have resulted in a 4-manifold edge. Do not collapse an edge if startVert is
///pinched - the vert would be marked NaN, but other edges could still be
///pointing to it.
pub fn collapse(
	edge: i32,
	halfedge: &mut Halfedges,
	tri_normal: &[Vector3<f64>],
	tri_rel: &[TriRelation],
	vert_pos: &mut Vec<Point3<f64>>,
	edges: &mut Vec<i32>,
	prop_stride: usize,
	precision: Precision,
	first_new_vert: i32,
) -> bool {
	let pair = halfedge.pair(edge);
	if pair < 0 {
		return false;
	}

	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(pair);
	let start_vert = halfedge.start(tri0_edge[0]);
	let end_vert = halfedge.start(tri0_edge[1]);

	let p_new = vert_pos[end_vert as usize];
	let p_old = vert_pos[start_vert as usize];
	let delta = p_new - p_old;
	// We don't check that startVert is still new here - it may have been
	// collapsed to a different neighbor. However, it's still fine to collapse it
	// further, as it's still only collapsing its own original neighbors together,
	// which can't stack errors arbitrarily far.
	let max_len = if end_vert < first_new_vert {
		precision.tolerance * precision.tolerance
	} else {
		precision.epsilon * precision.epsilon
	};
	let short_edge = delta.magnitude_squared() < max_len;

	// Orbit startVert
	let mut start = halfedge.pair(tri1_edge[1]);
	if !short_edge {
		let mut current = start;
		let mut ref_check = tri_rel[(pair / 3) as usize];
		let mut p_last = vert_pos[halfedge.start(tri1_edge[2]) as usize];
		while current != tri1_edge[0] {
			current = next_halfedge(current);
			let p_next = vert_pos[halfedge.end(current) as usize];
			let tri = (current / 3) as usize;
			let rel = tri_rel[tri];
			let projection = get_axis_aligned_projection(tri_normal[tri]);
			// Don't collapse if the edge is not redundant (this may have changed due
			// to the collapse of neighbors).
			if !rel.same_face(&ref_check) {
				let old_ref = ref_check;
				ref_check = tri_rel[(edge / 3) as usize];
				if !rel.same_face(&ref_check) {
					return false;
				}

				if rel.instance_id != old_ref.instance_id
					|| rel.face_id != old_ref.face_id
					|| tri_normal[(pair / 3) as usize].dot(&tri_normal[tri]) < -0.5
				{
					// Restrict collapse to colinear edges when the edge separates faces
					// or the edge is sharp. This ensures large shifts are not introduced
					// parallel to the tangent plane.
					if ccw(
						projection * p_last,
						projection * p_old,
						projection * p_new,
						precision.tolerance,
					) != 0
					{
						return false;
					}
				}
			}

			// Don't collapse edge if it would cause a triangle to invert.
			if ccw(
				projection * p_next,
				projection * p_last,
				projection * p_new,
				precision.epsilon,
			) < 0
			{
				return false;
			}

			p_last = p_next;
			current = halfedge.pair(current);
		}
	}

	// Orbit endVert
	{
		let mut current = halfedge.pair(tri0_edge[1]);
		while current != tri1_edge[2] {
			current = next_halfedge(current);
			edges.push(current);
			current = halfedge.pair(current);
		}
	}

	// Remove toRemove.startVert and replace with endVert.
	vert_pos[start_vert as usize] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
	collapse_tri(halfedge, &tri1_edge);

	// Orbit startVert
	let tri0 = (edge / 3) as usize;
	let tri1 = (pair / 3) as usize;
	let mut current = start;
	while current != tri0_edge[2] {
		current = next_halfedge(current);

		if prop_stride > 0 {
			// Update the shifted triangles to the vertBary of endVert
			let tri = (current / 3) as usize;
			if tri_rel[tri].same_face(&tri_rel[tri0]) {
				halfedge.set_prop(current, halfedge.prop(next_halfedge(edge)));
			} else if tri_rel[tri].same_face(&tri_rel[tri1]) {
				halfedge.set_prop(current, halfedge.prop(pair));
			}
		}

		let vert = halfedge.end(current);
		let next = halfedge.pair(current);
		for i in 0..edges.len() {
			if vert == halfedge.end(edges[i]) {
				form_loop(halfedge, vert_pos, edges[i], current);
				start = next;
				edges.truncate(i);
				break;
			}
		}

		current = next;
	}

	halfedge.update_vert(end_vert, start, tri0_edge[2]);
	collapse_tri(halfedge, &tri0_edge);
	remove_if_folded(start, halfedge, vert_pos);
	true
}

fn collapse_tri(halfedges: &mut Halfedges, tri_edge: &Vector3<i32>) {
	if halfedges.pair(tri_edge[1]) == -1 {
		return;
	}
	let pair1 = halfedges.pair(tri_edge[1]);
	let pair2 = halfedges.pair(tri_edge[2]);
	halfedges.pair_up(pair1, pair2);
	for i in 0..3 {
		halfedges.set(tri_edge[i], -1, -1, halfedges.prop(tri_edge[i]));
	}
}

///In the event that the edge collapse would create a non-manifold edge,
///instead we duplicate the two verts and attach the manifolds the other way
///across this edge.
fn form_loop(halfedge: &mut Halfedges, vert_pos: &mut Vec<Point3<f64>>, current: i32, end: i32) {
	let start_vert = vert_pos.len() as i32;
	let end_vert = start_vert + 1;
	vert_pos.extend([
		vert_pos[halfedge.start(current) as usize],
		vert_pos[halfedge.end(current) as usize],
	]);

	let old_match = halfedge.pair(current);
	let new_match = halfedge.pair(end);

	halfedge.update_vert(start_vert, old_match, new_match);
	halfedge.update_vert(end_vert, end, current);

	halfedge.pair_up(current, new_match);
	halfedge.pair_up(end, old_match);

	remove_if_folded(end, halfedge, vert_pos);
}

///Rather than actually removing the edges, this step merely marks them for
///removal, by setting vertPos to NaN and halfedge to {-1, -1, -1, -1}.
fn remove_if_folded(edge: i32, halfedge: &mut Halfedges, vert_pos: &mut [Point3<f64>]) {
	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(halfedge.pair(edge));
	if halfedge.pair(tri0_edge[1]) == -1 {
		return;
	}
	if halfedge.start(tri0_edge[2]) == halfedge.start(tri1_edge[2]) {
		if halfedge.pair(tri0_edge[1]) == tri1_edge[2] {
			if halfedge.pair(tri0_edge[2]) == tri1_edge[1] {
				for i in 0..3 {
					vert_pos[halfedge.start(tri0_edge[i]) as usize] =
						Point3::new(f64::NAN, f64::NAN, f64::NAN);
				}
			} else {
				vert_pos[halfedge.start(tri0_edge[1]) as usize] =
					Point3::new(f64::NAN, f64::NAN, f64::NAN);
			}
		} else {
			if halfedge.pair(tri0_edge[2]) == tri1_edge[1] {
				vert_pos[halfedge.start(tri1_edge[1]) as usize] =
					Point3::new(f64::NAN, f64::NAN, f64::NAN);
			}
		}

		halfedge.pair_up(halfedge.pair(tri0_edge[1]), halfedge.pair(tri1_edge[2]));
		halfedge.pair_up(halfedge.pair(tri0_edge[2]), halfedge.pair(tri1_edge[1]));

		for i in 0..3 {
			halfedge.set(tri0_edge[i], -1, -1, -1);
			halfedge.set(tri1_edge[i], -1, -1, -1);
		}
	}
}

pub fn tri_of(edge: i32) -> Vector3<i32> {
	let mut tri_edge = Vector3::default();
	tri_edge[0] = edge;
	tri_edge[1] = next_halfedge(tri_edge[0]);
	tri_edge[2] = next_halfedge(tri_edge[1]);
	tri_edge
}

pub fn is_01_longest(v0: Point2<f64>, v1: Point2<f64>, v2: Point2<f64>) -> bool {
	let e = [v1 - v0, v2 - v1, v0 - v2];
	let mut l = [0.0; 3];
	for i in 0..3 {
		l[i] = e[i].magnitude_squared();
	}
	l[0] > l[1] && l[0] > l[2]
}
