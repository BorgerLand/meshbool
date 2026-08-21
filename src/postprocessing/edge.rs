use crate::halfedge::{Halfedges, next_halfedge, prev_halfedge};
use crate::mesh_relations::{InstanceRelation, TriRelation};
use crate::util::math::{ccw, get_axis_aligned_projection, lerp, next3_usize, safe_normalize3};
use crate::{Precision, Properties, TrianglesWIP};
use nalgebra::{Matrix2, Matrix3, Matrix3x2, Point2, Point3, Vector3, Vector4, distance};
use std::f64;

// Deduplicate the given 4-manifold edge by duplicating endVert, thus making the
// edges distinct. Also duplicates startVert if it becomes pinched.
pub fn dedupe(vert_pos: &mut Vec<Point3<f64>>, tri: &mut TrianglesWIP, edge: usize) {
	// Orbit endVert
	let next_edge = next_halfedge(edge);
	let start_vert = tri.halfedge.start[edge];
	let end_vert = tri.halfedge.start[next_edge];
	let end_prop = tri.halfedge.prop[next_edge];
	let mut current = tri.halfedge.pair[next_edge] as usize;
	while current != edge {
		let vert = tri.halfedge.start[current];
		if vert == start_vert {
			// Single topological unit needs 2 faces added to be split
			let new_vert = vert_pos.len() as i32;
			vert_pos.push(vert_pos[end_vert as usize]);
			current = tri.halfedge.pair[next_halfedge(current)] as usize;
			let opposite = tri.halfedge.pair[next_halfedge(edge)] as usize;

			tri.halfedge.update_vert(new_vert, current, opposite);

			let mut new_halfedge = tri.halfedge.len();
			let mut old_face = current / 3;
			let mut outside_vert = tri.halfedge.start[current];
			tri.halfedge.push(end_vert, -1, end_prop);
			tri.halfedge.push(new_vert, -1, end_prop);
			tri.halfedge
				.push(outside_vert, -1, tri.halfedge.prop[current]);
			tri.halfedge
				.pair_up(new_halfedge + 2, tri.halfedge.pair[current] as usize);
			tri.halfedge.pair_up(new_halfedge + 1, current);
			tri.relation.push(tri.relation[old_face]);
			tri.normal.push(tri.normal[old_face]);

			new_halfedge += 3;
			old_face = opposite / 3;
			outside_vert = tri.halfedge.start[opposite];
			tri.halfedge.push(new_vert, -1, end_prop); // fix prop
			tri.halfedge.push(end_vert, -1, end_prop);
			tri.halfedge
				.push(outside_vert, -1, tri.halfedge.prop[opposite]);
			tri.halfedge
				.pair_up(new_halfedge + 2, tri.halfedge.pair[opposite] as usize);
			tri.halfedge.pair_up(new_halfedge + 1, opposite);
			tri.halfedge.pair_up(new_halfedge, new_halfedge - 3);
			tri.relation.push(tri.relation[old_face]);
			tri.normal.push(tri.normal[old_face]);

			break;
		}

		current = tri.halfedge.pair[next_halfedge(current)] as usize;
	}

	if current == edge {
		// Separate topological unit needs no new faces to be split
		let new_vert = vert_pos.len() as i32;
		vert_pos.push(vert_pos[end_vert as usize]);

		tri.halfedge
			.for_vert_mut(next_halfedge(current), |halfedge, e| {
				halfedge.start[e] = new_vert;
				halfedge.set_end(halfedge.pair[e] as usize, new_vert);
			});
	}

	// Orbit startVert
	let pair = tri.halfedge.pair[edge] as usize;
	current = tri.halfedge.pair[next_halfedge(pair)] as usize;
	while current != pair {
		let vert = tri.halfedge.start[current];
		if vert == end_vert {
			break; //connected: not a pinched vert
		}

		current = tri.halfedge.pair[next_halfedge(current)] as usize;
	}

	if current == pair {
		// Split the pinched vert the previous split created.
		let new_vert = vert_pos.len() as i32;
		vert_pos.push(vert_pos[end_vert as usize]);

		tri.halfedge
			.for_vert_mut(next_halfedge(current), |halfedge, e| {
				halfedge.start[e] = new_vert;
				halfedge.set_end(halfedge.pair[e] as usize, new_vert);
			});
	}
}

pub fn recursive_swap(
	edge: usize,
	tri: &mut TrianglesWIP,
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	tag: &mut i32,
	visited: &mut [i32],
	edge_swap_stack: &mut Vec<i32>,
	edges: &mut Vec<i32>,
	instance_rel: &[InstanceRelation],
	precision: Precision,
) {
	let pair = tri.halfedge.pair[edge];
	if pair < 0 {
		return;
	}
	let pair = pair as usize;

	// avoid infinite recursion
	if visited[edge] == *tag && visited[pair] == *tag {
		return;
	}

	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(pair);

	let projection = get_axis_aligned_projection(tri.normal[edge / 3]);
	let mut v = [Point2::default(); 4];
	for i in 0..3 {
		v[i] = projection * vert_pos[tri.halfedge.start[tri0_edge[i] as usize] as usize];
	}

	// Only operate on the long edge of a degenerate triangle.
	if ccw(v[0], v[1], v[2], precision.tolerance) > 0 || !is_01_longest_2(v[0], v[1], v[2]) {
		return;
	}

	// Switch to neighbor's projection.
	let projection = get_axis_aligned_projection(tri.normal[pair / 3]);
	for i in 0..3 {
		v[i] = projection * vert_pos[tri.halfedge.start[tri0_edge[i] as usize] as usize];
	}

	v[3] = projection * vert_pos[tri.halfedge.start[tri1_edge[2] as usize] as usize];

	let l01 = distance(&v[1], &v[0]);
	let l02 = distance(&v[2], &v[0]);
	let ratio = l02 / l01;
	let capped = if ratio < 1.0 { ratio } else { 1.0 };
	let a = if capped > 0.0 { capped } else { 0.0 };

	// Only operate if the other triangles are not degenerate.
	if ccw(v[1], v[0], v[3], precision.tolerance) <= 0 {
		if !is_01_longest_2(v[1], v[0], v[3]) {
			return;
		}
		// Two facing, long-edge degenerates can swap.
		swap(edge, a, tri, vert_pos, properties);
		let e23 = v[3] - v[2];
		if e23.magnitude_squared() < precision.tolerance * precision.tolerance {
			*tag += 1;
			collapse(
				tri0_edge[2] as usize,
				&mut tri.halfedge,
				&tri.normal,
				&tri.relation,
				instance_rel,
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
			visited[edge] = *tag;
			visited[pair] = *tag;
			edge_swap_stack
				.extend([tri1_edge[1], tri1_edge[0], tri0_edge[1], tri0_edge[0]].map(|x| x as i32));
		}

		return;
	} else if ccw(v[0], v[3], v[2], precision.tolerance) <= 0
		|| ccw(v[1], v[2], v[3], precision.tolerance) <= 0
	{
		return;
	}

	//normal path
	swap(edge, a, tri, vert_pos, properties);
	visited[edge] = *tag;
	visited[pair] = *tag;
	edge_swap_stack.extend([
		tri.halfedge.pair[tri1_edge[0] as usize],
		tri.halfedge.pair[tri0_edge[1] as usize],
	]);
}

///Collapses the given edge by removing startVert - returns false if the edge
///cannot be collapsed. May split the mesh topologically if the collapse would
///have resulted in a 4-manifold edge. Do not collapse an edge if startVert is
///pinched - the vert would be marked NaN, but other edges could still be
///pointing to it.
pub fn collapse(
	edge: usize,
	halfedge: &mut Halfedges,
	tri_normal: &[Vector3<f64>],
	tri_rel: &[TriRelation],
	instance_rel: &[InstanceRelation],
	vert_pos: &mut Vec<Point3<f64>>,
	edges: &mut Vec<i32>,
	prop_stride: usize,
	precision: Precision,
	first_new_vert: usize,
) -> bool {
	let pair = halfedge.pair[edge];
	if pair < 0 {
		return false;
	}
	let pair = pair as usize;

	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(pair);
	let start_vert = halfedge.start[tri0_edge[0]] as usize;
	let end_vert = halfedge.start[tri0_edge[1]];

	let p_new = vert_pos[end_vert as usize];
	let p_old = vert_pos[start_vert];
	let delta = p_new - p_old;
	// We don't check that startVert is still new here - it may have been
	// collapsed to a different neighbor. However, it's still fine to collapse it
	// further, as it's still only collapsing its own original neighbors together,
	// which can't stack errors arbitrarily far.
	let max_len = if (end_vert as usize) < first_new_vert {
		precision.tolerance * precision.tolerance
	} else {
		precision.epsilon * precision.epsilon
	};
	let short_edge = delta.magnitude_squared() < max_len;

	// Orbit startVert
	let mut start = halfedge.pair[tri1_edge[1]] as usize;
	if !short_edge {
		let mut current = start;
		let mut ref_check = tri_rel[pair / 3];
		let mut p_last = vert_pos[halfedge.start[tri1_edge[2]] as usize];
		while current != tri1_edge[0] as usize {
			current = next_halfedge(current);
			let p_next = vert_pos[halfedge.end(current) as usize];
			let tri = (current / 3) as usize;
			let rel = tri_rel[tri];
			let projection = get_axis_aligned_projection(tri_normal[tri]);
			// Don't collapse if the edge is not redundant (this may have changed
			// due to the collapse of neighbors).
			if rel != ref_check {
				let old_rel = ref_check;
				ref_check = tri_rel[edge / 3];
				if rel != ref_check {
					return false;
				}

				//if these are from different meshes.
				//OR if they are from the same mesh, check if user allows them to collapse (different faces).
				//OR if the user says "no collapsin" do this final normals check
				if rel.instance_id != old_rel.instance_id
					|| (rel.face_id != old_rel.face_id
						&& instance_rel[rel.instance_id as usize].user_provided_face_id)
					|| tri_normal[pair / 3].dot(&tri_normal[tri]) < -0.5
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
			current = halfedge.pair[current] as usize;
		}
	}

	// Orbit endVert
	{
		let mut current = halfedge.pair[tri0_edge[1]] as usize;
		while current != tri1_edge[2] {
			current = next_halfedge(current);
			edges.push(current as i32);
			current = halfedge.pair[current] as usize;
		}
	}

	// Remove toRemove.startVert and replace with endVert.
	vert_pos[start_vert] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
	collapse_tri(halfedge, tri1_edge);

	let start_prop0 = halfedge.prop[tri0_edge[0]];
	let end_prop0 = halfedge.prop[tri0_edge[1]];
	let start_prop1 = halfedge.prop[tri1_edge[1]];
	let end_prop1 = halfedge.prop[tri1_edge[0]];
	// Orbit startVert
	let mut current = start;
	while current != tri0_edge[2] {
		current = next_halfedge(current);

		if prop_stride > 0 {
			if halfedge.prop[current] == start_prop0 {
				halfedge.prop[current] = end_prop0;
			} else if halfedge.prop[current] == start_prop1 {
				halfedge.prop[current] = end_prop1;
			}
		}

		let vert = halfedge.end(current);
		let next = halfedge.pair[current] as usize;
		for i in 0..edges.len() {
			if vert == halfedge.end(edges[i] as usize) {
				form_loop(halfedge, vert_pos, edges[i] as usize, current);
				start = next;
				edges.truncate(i);
				break;
			}
		}

		current = next;
	}

	halfedge.update_vert(end_vert, start, tri0_edge[2]);
	collapse_tri(halfedge, tri0_edge);
	remove_if_folded(start, halfedge, vert_pos);
	true
}

///A degenerate sliver whose long edge can be flipped into its neighbor.
pub fn swappable(
	mut edge: usize,
	halfedge: &Halfedges,
	tri_normal: &[Vector3<f64>],
	vert_pos: &[Point3<f64>],
	epsilon: f64,
) -> bool {
	let pair = halfedge.pair[edge] as usize;
	let quad_edge = Vector4::new(
		edge,
		next_halfedge(edge),
		prev_halfedge(edge),
		prev_halfedge(pair),
	);

	let mut tri = edge / 3;
	let mut projection = get_axis_aligned_projection(tri_normal[tri]);
	let mut v = [Point3::default(); 4];
	let mut vp = [Point2::default(); 4];
	for i in 0..4 {
		let vert = halfedge.start[quad_edge[i]] as usize;
		v[i] = vert_pos[vert];
		vp[i] = projection * v[i];
	}

	if ccw(vp[0], vp[1], vp[2], epsilon) > 0 || !is_01_longest_3(v[0], v[1], v[2]) {
		return false;
	}

	// Switch to neighbor's projection.
	edge = pair;
	tri = edge / 3;
	projection = get_axis_aligned_projection(tri_normal[tri]);
	for i in 0..4 {
		vp[i] = projection * v[i];
	}

	is_01_longest_3(v[1], v[0], v[3])
		|| (ccw(vp[1], vp[0], vp[3], epsilon) > 0
			&& ccw(vp[1], vp[2], vp[3], epsilon) > 0
			&& ccw(vp[2], vp[0], vp[3], epsilon) > 0)
}

#[derive(Clone, Copy)]
pub struct Merger {
	///This edge's own contribution to the error budget.
	pub added_cost: f64,
	///added_cost plus the accumulated cost already spent at either endpoint,
	///or one of the sentinels above.
	pub total_cost: f64,
	///Interpolation parameter along the edge, used for properties.
	pub a: f64,
	///Where the surviving vert lands.
	pub new_pos: Point3<f64>,
}

impl Default for Merger {
	fn default() -> Self {
		Self {
			added_cost: f64::INFINITY,
			total_cost: f64::INFINITY,
			a: f64::NAN,
			new_pos: Point3::new(f64::NAN, f64::NAN, f64::NAN),
		}
	}
}

impl Merger {
	pub const K_SHORT: f64 = -2.0;
	pub const K_SWAP: f64 = -1.0;

	pub fn valid(&self) -> bool {
		self.added_cost.is_finite()
	}

	pub fn free(&self) -> bool {
		self.total_cost < 0.0
	}

	pub fn short(&self) -> bool {
		self.total_cost == Self::K_SHORT
	}

	pub fn swap(&self) -> bool {
		self.total_cost == Self::K_SWAP
	}
}

///Prices the collapse of this edge and reports where the merged vert should go.
pub fn check(
	edge: usize,
	tri: &TrianglesWIP,
	vert_pos: &[Point3<f64>],
	instance_rel: &[InstanceRelation],
	prop_stride: usize,
	epsilon: f64,
) -> Merger {
	let pair = tri.halfedge.pair[edge] as usize;
	let start = tri.halfedge.start[edge] as usize;
	let end = tri.halfedge.end(edge) as usize;
	let delta = vert_pos[end] - vert_pos[start];
	let len_sq = delta.magnitude_squared();
	let mid = vert_pos[start] + delta / 2.0;
	if len_sq < epsilon * epsilon {
		return Merger {
			added_cost: 0.0,
			total_cost: Merger::K_SHORT,
			a: 0.5,
			new_pos: mid,
		};
	}
	let mut a = Matrix3::zeros();
	let mut b = Vector3::zeros();
	let mut c = 0.0;

	let mut add_cost = |normal: Vector3<f64>, pos| {
		a += normal * normal.transpose();
		let d = normal.dot(&(pos - mid));
		b += normal * d;
		c += d * d;
	};

	let mut add_tri = |current, first_edge: usize| {
		if current == first_edge {
			return; //don't double-count the collapsing triangles
		}
		let normal = tri.normal[tri.halfedge.tri(current)];
		let pos = vert_pos[tri.halfedge.start[current] as usize];
		// Equal-weighted per triangle, keeps the cost in terms of distance.
		// Angle-weighting like pseudo-normals may be better, but it is more
		// expensive to compute and may be less stable on degenerates.
		add_cost(normal, pos);

		if !continuous(
			current,
			&tri.halfedge,
			&tri.relation,
			instance_rel,
			prop_stride,
		) {
			// Penalize motion across an edge that contains a property boundary.
			add_cost(
				safe_normalize3(
					normal.cross(&(vert_pos[tri.halfedge.end(current) as usize] - pos)),
				),
				pos,
			);
		}
	};

	tri.halfedge
		.for_vert(edge, |current| add_tri(current, edge));
	tri.halfedge.for_vert(pair, |current| {
		add_tri(current, tri.halfedge.pair[edge] as usize)
	});

	// Constrain the solution to the plane containing the edge and its normal.
	let p = Matrix3x2::from_columns(&[delta, (tri.normal[edge / 3] + tri.normal[pair / 3]) / 2.0]);
	// Epsilon stabilizes the inverse, driving the solution toward the midpoint.
	let a2 = p.transpose() * a * p;
	let b2 = p.transpose() * b;
	let mut u = a2
		.try_inverse()
		.unwrap_or_else(|| Matrix2::from_element(f64::NAN))
		* b2;
	if !u[0].is_finite() {
		return Merger {
			added_cost: 0.0,
			total_cost: 0.0,
			a: 0.5,
			new_pos: mid,
		};
	}
	// u[0] is the interpolation along the collapsed edge, which is used to
	// interpolate the properties. It is clamped to avoid extrapolation.
	u[0] = u[0].clamp(-0.5, 0.5);
	// Cost has units of length^2.
	let cost = (u.dot(&(a2 * u)) - 2.0 * b2.dot(&u) + c).max(0.0);
	Merger {
		added_cost: cost,
		total_cost: cost,
		a: u[0] + 0.5,
		new_pos: mid + p * u,
	}
}

///True when the two triangles sharing this edge belong to the same face and
///agree on properties across it, i.e. nothing about the surface changes here.
fn continuous(
	edge: usize,
	halfedge: &Halfedges,
	tri_rel: &[TriRelation],
	instance_rel: &[InstanceRelation],
	prop_stride: usize,
) -> bool {
	let pair = halfedge.pair[edge] as usize;
	let rel0 = tri_rel[halfedge.tri(edge)];
	let rel1 = tri_rel[halfedge.tri(pair)];
	return rel0.instance_id == rel1.instance_id
		&& (rel0.face_id == rel1.face_id
			|| !instance_rel[rel0.instance_id as usize].user_provided_face_id)
		&& (prop_stride == 0
			|| (halfedge.prop[edge] == halfedge.prop_end(pair)
				&& halfedge.prop[pair] == halfedge.prop_end(edge)));
}

pub fn swap(
	edge: usize,
	a: f64,
	tri: &mut TrianglesWIP,
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
) {
	let pair = tri.halfedge.pair[edge] as usize;
	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(pair);

	// The 0-verts are swapped to the opposite 2-verts.
	tri.halfedge.start[tri0_edge[0]] = tri.halfedge.start[tri1_edge[2]];
	tri.halfedge.start[tri1_edge[0]] = tri.halfedge.start[tri0_edge[2]];
	tri.halfedge
		.pair_up(tri0_edge[0], tri.halfedge.pair[tri1_edge[2]] as usize);
	tri.halfedge
		.pair_up(tri1_edge[0], tri.halfedge.pair[tri0_edge[2]] as usize);
	tri.halfedge.pair_up(tri0_edge[2], tri1_edge[2]);
	// Both triangles are now subsets of the neighboring triangle.
	let tri0 = tri.halfedge.tri(tri0_edge[0]);
	let tri1 = tri.halfedge.tri(tri1_edge[0]);
	tri.normal[tri0] = tri.normal[tri1];
	tri.relation[tri0] = tri.relation[tri1];

	let prop_stride = properties.stride;
	if prop_stride > 0 {
		let prop_idx0 = tri.halfedge.prop[tri1_edge[0]];
		let prop_idx1 = tri.halfedge.prop[tri1_edge[1]];
		tri.halfedge.prop[tri0_edge[1]] = tri.halfedge.prop[tri1_edge[0]];
		tri.halfedge.prop[tri0_edge[0]] = tri.halfedge.prop[tri1_edge[2]];
		if prop_idx0
			== tri
				.halfedge
				.prop_end(tri.halfedge.pair[tri0_edge[1]] as usize)
		{
			let mid_prop = tri.halfedge.prop[tri.halfedge.pair[tri0_edge[1]] as usize];
			tri.halfedge.prop[tri1_edge[0]] = mid_prop;
			tri.halfedge.prop[tri0_edge[2]] = mid_prop;
		} else if prop_idx1 == tri.halfedge.prop[tri.halfedge.pair[tri1_edge[0]] as usize] {
			let mid_prop = tri
				.halfedge
				.prop_end(tri.halfedge.pair[tri1_edge[0]] as usize);
			tri.halfedge.prop[tri1_edge[0]] = mid_prop;
			tri.halfedge.prop[tri0_edge[2]] = mid_prop;
		} else {
			let new_prop = (properties.data.len() / prop_stride) as i32;
			for p in 0..prop_stride {
				properties.data.push(
					a * properties.data[prop_stride * (prop_idx0 as usize) + p]
						+ (1.0 - a) * properties.data[prop_stride * (prop_idx1 as usize) + p],
				);
			}
			tri.halfedge.prop[tri1_edge[0]] = new_prop;
			tri.halfedge.prop[tri0_edge[2]] = new_prop;
		}
	}

	// if the new edge already exists, duplicate the verts and split the mesh.
	let mut current = tri.halfedge.pair[tri1_edge[0]] as usize;
	let end_vert = tri.halfedge.end(tri1_edge[1]);
	while current != tri0_edge[1] {
		current = next_halfedge(current);
		if tri.halfedge.end(current) == end_vert {
			form_loop(&mut tri.halfedge, vert_pos, tri0_edge[2], current);
			remove_if_folded(tri0_edge[2], &mut tri.halfedge, vert_pos);
			return;
		}

		current = tri.halfedge.pair[current] as usize;
	}
}

///Collapses the given edge by removing startVert - returns false if the edge
///cannot be collapsed. May split the mesh topologically if the collapse would
///have resulted in a 4-manifold edge. Do not collapse an edge if startVert is
///pinched - the vert would be marked NaN, but other edges could still be
///pointing to it.
pub fn collapse2(
	edge: usize,
	halfedge: &mut Halfedges,
	tri_normal: &[Vector3<f64>],
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	edges: &mut Vec<i32>,
	merger: Merger,
	epsilon: f64,
) -> bool {
	edges.clear();

	let pair = halfedge.pair[edge];
	if pair < 0 {
		return false;
	}
	let pair = pair as usize;

	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(pair);
	let start_vert = halfedge.start[tri0_edge[0] as usize] as usize;
	let end_vert = halfedge.start[tri0_edge[1] as usize] as usize;

	if !merger.short() {
		let mut first_edge = edge;
		let mut old_worst: f64 = 0.0;
		let mut new_worst: f64 = 0.0;
		let mut collapse = true;
		let mut check_tri = |current, first_edge: usize, collapse: &mut bool| {
			let last = halfedge.pair[prev_halfedge(current)] as usize;
			if !*collapse || current == first_edge || last == first_edge {
				return; // ignore the collapsing triangles.
			}
			let p_old = vert_pos[halfedge.start[current] as usize];
			let p_curr = vert_pos[halfedge.end(current) as usize];
			let p_last = vert_pos[halfedge.end(last) as usize];
			let old_edges = Matrix3::from_columns(&[
				(p_old - p_last).normalize(),
				(p_last - p_curr).normalize(),
				(p_curr - p_old).normalize(),
			]);
			let new_edges = Matrix3::from_columns(&[
				(merger.new_pos - p_last).normalize(),
				(p_last - p_curr).normalize(),
				(p_curr - merger.new_pos).normalize(),
			]);
			if new_edges
				.column(1)
				.cross(&new_edges.column(0))
				.dot(&tri_normal[current / 3])
				< 0.0
			{
				*collapse = false;
				return;
			}
			for i in 0..3 {
				old_worst =
					old_worst.max(old_edges.column(i).dot(&old_edges.column(next3_usize(i))));
				new_worst =
					new_worst.max(new_edges.column(i).dot(&new_edges.column(next3_usize(i))));
			}
		};
		halfedge.for_vert(first_edge, |current| {
			check_tri(current, first_edge, &mut collapse)
		});
		if !collapse {
			return false;
		};
		first_edge = halfedge.pair[edge] as usize;
		halfedge.for_vert(first_edge, |current| {
			check_tri(current, first_edge, &mut collapse)
		});
		if !collapse {
			return false;
		}
		// Reject a collapse that would worsen the Delaunay condition too much,
		// which is equivalent to having no obtuse angles. This would be a
		// threshold of zero on this dot product, but 0.5 is used to allow obtuse
		// angles up to 120 degrees to allow more edges to collapse. Planar cases
		// (small cost) relax this threshold to allow generic polygon
		// triangulation, but not so much as to create new degenerate triangles.
		let threshold = if merger.added_cost > epsilon * epsilon {
			0.5
		} else {
			0.99
		};
		if new_worst > threshold && new_worst > old_worst {
			return false;
		}
	}

	// Orbit endVert
	let mut current = halfedge.pair[tri0_edge[1] as usize] as usize;
	while current != tri1_edge[2] as usize {
		current = next_halfedge(current);
		edges.push(current as i32);
		current = halfedge.pair[current] as usize;
	}

	let start_prop0 = halfedge.prop[tri0_edge[0] as usize];
	let end_prop0 = halfedge.prop[tri0_edge[1] as usize];
	let start_prop1 = halfedge.prop[tri1_edge[1] as usize];
	let end_prop1 = halfedge.prop[tri1_edge[0] as usize];
	let prop_stride = properties.stride;
	for p in 0..prop_stride {
		properties.data[prop_stride * end_prop0 as usize + p] = lerp(
			properties.data[prop_stride * start_prop0 as usize + p],
			properties.data[prop_stride * end_prop0 as usize + p],
			merger.a,
		);
		if end_prop1 != end_prop0 {
			properties.data[prop_stride * end_prop1 as usize + p] = lerp(
				properties.data[prop_stride * start_prop1 as usize + p],
				properties.data[prop_stride * end_prop1 as usize + p],
				merger.a,
			);
		}
	}

	let mut start = halfedge.pair[tri1_edge[1] as usize] as usize;
	// Remove toRemove.startVert and replace with endVert.
	vert_pos[start_vert] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
	vert_pos[end_vert] = merger.new_pos;
	collapse_tri(halfedge, tri1_edge);

	// Orbit startVert
	let mut current = start;
	while current != tri0_edge[2] as usize {
		current = next_halfedge(current);
		if prop_stride > 0 {
			if halfedge.prop[current] == start_prop0 {
				halfedge.prop[current] = end_prop0;
			} else if halfedge.prop[current] == start_prop1 {
				halfedge.prop[current] = end_prop1;
			}
		}

		let vert = halfedge.end(current);
		let next = halfedge.pair[current] as usize;
		for i in 0..edges.len() {
			if vert == halfedge.end(edges[i] as usize) {
				form_loop(halfedge, vert_pos, edges[i] as usize, current);
				start = next;
				edges.truncate(i);
				break;
			}
		}

		current = next;
	}

	halfedge.update_vert(end_vert as i32, start, tri0_edge[2] as usize);

	collapse_tri(halfedge, tri0_edge);
	remove_if_folded(start, halfedge, vert_pos);
	true
}

fn collapse_tri(halfedges: &mut Halfedges, tri_edge: Vector3<usize>) {
	if halfedges.pair[tri_edge[1]] == -1 {
		return;
	}
	let pair1 = halfedges.pair[tri_edge[1]] as usize;
	let pair2 = halfedges.pair[tri_edge[2]] as usize;
	halfedges.pair_up(pair1, pair2);
	for i in 0..3 {
		halfedges.set(tri_edge[i] as usize, -1, -1, halfedges.prop[tri_edge[i]]);
	}
}

///In the event that the edge collapse would create a non-manifold edge,
///instead we duplicate the two verts and attach the manifolds the other way
///across this edge.
fn form_loop(
	halfedge: &mut Halfedges,
	vert_pos: &mut Vec<Point3<f64>>,
	current: usize,
	end: usize,
) {
	let start_vert = vert_pos.len() as i32;
	let end_vert = start_vert + 1;
	vert_pos.extend([
		vert_pos[halfedge.start[current] as usize],
		vert_pos[halfedge.end(current) as usize],
	]);

	let old_match = halfedge.pair[current] as usize;
	let new_match = halfedge.pair[end] as usize;

	halfedge.update_vert(start_vert, old_match, new_match);
	halfedge.update_vert(end_vert, end, current);

	halfedge.pair_up(current, new_match);
	halfedge.pair_up(end, old_match);

	remove_if_folded(end, halfedge, vert_pos);
}

///Rather than actually removing the edges, this step merely marks them for
///removal, by setting vertPos to NaN and halfedge to {-1, -1, -1, -1}.
fn remove_if_folded(edge: usize, halfedge: &mut Halfedges, vert_pos: &mut [Point3<f64>]) {
	let tri0_edge = tri_of(edge);
	let tri1_edge = tri_of(halfedge.pair[edge] as usize);
	if halfedge.pair[tri0_edge[1] as usize] == -1 {
		return;
	}
	if halfedge.start[tri0_edge[2]] == halfedge.start[tri1_edge[2]] {
		if halfedge.pair[tri0_edge[1]] as usize == tri1_edge[2] {
			if halfedge.pair[tri0_edge[2]] as usize == tri1_edge[1] {
				for i in 0..3 {
					vert_pos[halfedge.start[tri0_edge[i]] as usize] =
						Point3::new(f64::NAN, f64::NAN, f64::NAN);
				}
			} else {
				vert_pos[halfedge.start[tri0_edge[1]] as usize] =
					Point3::new(f64::NAN, f64::NAN, f64::NAN);
			}
		} else {
			if halfedge.pair[tri0_edge[2]] as usize == tri1_edge[1] {
				vert_pos[halfedge.start[tri1_edge[1]] as usize] =
					Point3::new(f64::NAN, f64::NAN, f64::NAN);
			}
		}

		halfedge.pair_up(
			halfedge.pair[tri0_edge[1]] as usize,
			halfedge.pair[tri1_edge[2]] as usize,
		);
		halfedge.pair_up(
			halfedge.pair[tri0_edge[2]] as usize,
			halfedge.pair[tri1_edge[1]] as usize,
		);

		for i in 0..3 {
			halfedge.set(tri0_edge[i], -1, -1, -1);
			halfedge.set(tri1_edge[i], -1, -1, -1);
		}
	}
}

pub fn tri_of(edge: usize) -> Vector3<usize> {
	let mut tri_edge = Vector3::default();
	tri_edge[0] = edge;
	tri_edge[1] = next_halfedge(tri_edge[0]);
	tri_edge[2] = next_halfedge(tri_edge[1]);
	tri_edge
}

pub fn is_01_longest_2(v0: Point2<f64>, v1: Point2<f64>, v2: Point2<f64>) -> bool {
	let e = [v1 - v0, v2 - v1, v0 - v2];
	let mut l = [0.0; 3];
	for i in 0..3 {
		l[i] = e[i].magnitude_squared();
	}
	l[0] > l[1] && l[0] > l[2]
}

pub fn is_01_longest_3(v0: Point3<f64>, v1: Point3<f64>, v2: Point3<f64>) -> bool {
	let e = [v1 - v0, v2 - v1, v0 - v2];
	let mut l = [0.0; 3];
	for i in 0..3 {
		l[i] = e[i].magnitude_squared();
	}
	l[0] > l[1] && l[0] > l[2]
}
