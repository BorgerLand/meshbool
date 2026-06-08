use crate::common::DeterministicMap;
use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{get_axis_aligned_projection, next_halfedge};
use crate::utils::ccw;
use nalgebra::{Point2, Point3, Vector3, distance};
use std::collections::hash_map::Entry;

fn tri_of(edge: i32) -> Vector3<i32> {
	let mut tri_edge = Vector3::default();
	tri_edge[0] = edge;
	tri_edge[1] = next_halfedge(tri_edge[0]);
	tri_edge[2] = next_halfedge(tri_edge[1]);
	tri_edge
}

fn is_01_longest(v0: Point2<f64>, v1: Point2<f64>, v2: Point2<f64>) -> bool {
	let e = [v1 - v0, v2 - v1, v0 - v2];
	let mut l = [0.0; 3];
	for i in 0..3 {
		l[i] = e[i].magnitude_squared();
	}
	l[0] > l[1] && l[0] > l[2]
}

#[derive(Default)]
struct FlagStore {
	s: Vec<usize>,
}

impl FlagStore {
	fn run_seq(
		&mut self,
		mesh: &mut MeshBoolImpl,
		n: usize,
		mut pred: impl FnMut(&mut MeshBoolImpl, usize) -> bool,
		mut f: impl FnMut(&mut MeshBoolImpl, usize),
	) {
		for i in 0..n {
			if pred(mesh, i) {
				self.s.push(i);
			}
		}

		for &i in &self.s {
			f(mesh, i);
		}
		self.s = Vec::default();
	}

	fn run(
		&mut self,
		mesh: &mut MeshBoolImpl,
		n: usize,
		pred: impl FnMut(&mut MeshBoolImpl, usize) -> bool,
		f: impl FnMut(&mut MeshBoolImpl, usize),
	) {
		self.run_seq(mesh, n, pred, f)
	}
}

impl MeshBoolImpl {
	///Duplicates just enough verts to covert an even-manifold to a proper
	///2-manifold, splitting non-manifold verts and edges with too many triangles.
	pub fn cleanup_topology(&mut self) {
		if self.halfedge.is_empty() {
			return;
		}
		debug_assert!(self.is_manifold(), "polygon mesh is not manifold!");

		// In the case of a very bad triangulation, it is possible to create pinched
		// verts. They must be removed before edge collapse.
		self.split_pinched_verts();
		self.dedupe_edges();
	}

	///Collapses degenerate triangles by removing edges shorter than tolerance_ and
	///any edge that is preceeded by an edge that joins the same two face relations.
	///It also performs edge swaps on the long edges of degenerate triangles, though
	///there are some configurations of degenerates that cannot be removed this way.
	///
	///Before collapsing edges, the mesh is checked for duplicate edges (more than
	///one pair of triangles sharing the same edge), which are removed by
	///duplicating one vert and adding two triangles. These degenerate triangles are
	///likely to be collapsed again in the subsequent simplification.
	///
	///Note when an edge collapse would result in something non-manifold, the
	///vertices are duplicated in such a way as to remove handles or separate
	///meshes, thus decreasing the Genus(). It only increases when meshes that have
	///collapsed to just a pair of triangles are removed entirely.
	///
	///Verts with index less than firstNewVert will be left uncollapsed. This is
	///zero by default so that everything can be collapsed.
	///
	///Rather than actually removing the edges, this step merely marks them for
	///removal, by setting vertPos to NaN and halfedge to {-1, -1, -1, -1}.
	pub fn simplify_topology(&mut self, first_new_vert: i32) {
		if self.halfedge.is_empty() {
			return;
		}

		self.cleanup_topology();
		self.collapse_short_edges(first_new_vert);
		self.collapse_colinear_edges(first_new_vert);
		self.swap_degenerates(first_new_vert);
		// Merging verts causes their normals to change
		self.calculate_vert_normals();
	}

	pub fn remove_degenerates(&mut self, first_new_vert: Option<i32>) {
		let first_new_vert = first_new_vert.unwrap_or(0);
		if self.halfedge.is_empty() {
			return;
		}

		self.cleanup_topology();
		self.collapse_short_edges(first_new_vert);
		self.swap_degenerates(first_new_vert);
		// Merging verts causes their normals to change
		self.calculate_vert_normals();
	}

	fn collapse_short_edges(&mut self, first_new_vert: i32) {
		let mut s = FlagStore::default();
		let mut num_flagged = 0;
		let nb_edges = self.halfedge.len();

		let mut scratch_buffer = Vec::with_capacity(10);
		// Short edges get to skip several checks and hence remove more classes of
		// degenerate triangles than flagged edges do, but this could in theory lead
		// to error stacking where a vertex moves too far. For this reason this is
		// restricted to epsilon, rather than tolerance. However, in the case of a
		// Boolean operation, we set firstNewVert in order to only operate on
		// newly-created verts, which means error stacking is not a concern, so we
		// allow collapsing up to tolerance in that case.
		let tol = if first_new_vert == 0 {
			self.epsilon
		} else {
			self.tolerance
		};

		let short_edge = |myself: &mut MeshBoolImpl, edge| {
			let edge = edge as i32;
			let pair = myself.halfedge.pair(edge);
			if pair < 0 {
				return false;
			}
			let start = myself.halfedge.start(edge);
			let end = myself.halfedge.end(edge);
			if start < first_new_vert && end < first_new_vert {
				return false;
			}
			// Flag short edges
			let delta = myself.vert_pos[end as usize] - myself.vert_pos[start as usize];
			let len_sq = delta.magnitude_squared();
			// To ensure tolerance_-scale errors don't stack, only collapse these edges
			// if they connect a new vert to an old vert, since old verts are only
			// allowed to move by epsilon_.
			let max_len = if end < first_new_vert {
				tol * tol
			} else {
				myself.epsilon * myself.epsilon
			};
			len_sq < max_len
		};

		s.run(self, nb_edges, short_edge, |myself, i| {
			let did_collapse =
				myself.collapse_edge(i as i32, &mut scratch_buffer, tol, first_new_vert);
			if did_collapse {
				num_flagged += 1;
			}
			scratch_buffer.truncate(0);
		});
	}

	fn collapse_colinear_edges(&mut self, first_new_vert: i32) {
		let mut s = FlagStore::default();
		let nb_edges = self.halfedge.len();
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
			let colinear_edge = |myself: &mut MeshBoolImpl, edge| {
				let edge = edge as i32;
				let pair = myself.halfedge.pair(edge);
				if pair < 0 || myself.halfedge.start(edge) < first_new_vert {
					return false;
				}
				// Flag redundant edges - those where the startVert is surrounded by only
				// two original triangles.
				let ref0 = myself.mesh_relation.tri_ref[(edge / 3) as usize];
				let mut current = next_halfedge(pair);
				let mut ref1 = myself.mesh_relation.tri_ref[(current / 3) as usize];
				let mut ref1_updated = !ref0.same_face(&ref1);
				while current != edge {
					current = next_halfedge(myself.halfedge.pair(current));
					let tri = current / 3;
					let tri_ref = myself.mesh_relation.tri_ref[tri as usize];
					if !tri_ref.same_face(&ref0) && !tri_ref.same_face(&ref1) {
						if !ref1_updated {
							ref1 = tri_ref;
							ref1_updated = true;
						} else {
							return false;
						}
					}
				}

				true
			};

			s.run(self, nb_edges, colinear_edge, |myself, i| {
				let did_collapse = myself.collapse_edge(i as i32, &mut scratch_buffer, -1.0, 0);
				if did_collapse {
					num_flagged += 1;
				}
				scratch_buffer.truncate(0);
			});

			if num_flagged == 0 {
				break;
			}
		}
	}

	fn swap_degenerates(&mut self, first_new_vert: i32) {
		//RecursiveEdgeSwap
		let mut s = FlagStore::default();
		let mut num_flagged = 0;
		let nb_edges = self.halfedge.len();
		let mut scratch_buffer = Vec::with_capacity(10);

		let swappable_edge = |myself: &mut MeshBoolImpl, edge| -> bool {
			let mut edge = edge as i32;
			let pair = myself.halfedge.pair(edge);
			if pair < 0 {
				return false;
			}
			let tri_edge = tri_of(edge);
			let pair_tri_edge = tri_of(pair);
			if myself.halfedge.start(tri_edge[0]) < first_new_vert
				&& myself.halfedge.start(tri_edge[1]) < first_new_vert
				&& myself.halfedge.start(tri_edge[2]) < first_new_vert
				&& myself.halfedge.start(pair_tri_edge[2]) < first_new_vert
			{
				return false;
			}

			let mut tri = edge / 3;
			let mut projection = get_axis_aligned_projection(myself.face_normal[tri as usize]);
			let mut v = [Point2::<f64>::default(); 3];
			for i in 0..3 {
				v[i] = projection * myself.vert_pos[myself.halfedge.start(tri_edge[i]) as usize];
			}
			if ccw(v[0], v[1], v[2], myself.tolerance) > 0 || !is_01_longest(v[0], v[1], v[2]) {
				return false;
			}

			// Switch to neighbor's projection.
			edge = pair;
			tri = edge / 3;
			projection = get_axis_aligned_projection(myself.face_normal[tri as usize]);
			for i in 0..3 {
				v[i] =
					projection * myself.vert_pos[myself.halfedge.start(pair_tri_edge[i]) as usize];
			}

			ccw(v[0], v[1], v[2], myself.tolerance) > 0 || is_01_longest(v[0], v[1], v[2])
		};

		let mut edge_swap_stack = Vec::new();
		let mut visited = vec![-1; self.halfedge.len()];
		let mut tag = 0;
		s.run(self, nb_edges, swappable_edge, |myself, i| {
			num_flagged += 1;
			tag += 1;
			myself.recursive_edge_swap(
				i as i32,
				&mut tag,
				&mut visited,
				&mut edge_swap_stack,
				&mut scratch_buffer,
			);
			while !edge_swap_stack.is_empty() {
				let last = edge_swap_stack.pop().unwrap();
				myself.recursive_edge_swap(
					last,
					&mut tag,
					&mut visited,
					&mut edge_swap_stack,
					&mut scratch_buffer,
				);
			}
		});
	}

	// Deduplicate the given 4-manifold edge by duplicating endVert, thus making the
	// edges distinct. Also duplicates startVert if it becomes pinched.
	fn dedupe_edge(&mut self, edge: i32) {
		// Orbit endVert
		let next_edge = next_halfedge(edge);
		let start_vert = self.halfedge.start(edge);
		let end_vert = self.halfedge.start(next_edge);
		let end_prop = self.halfedge.prop(next_edge);
		let mut current = self.halfedge.pair(next_edge);
		while current != edge {
			let vert = self.halfedge.start(current);
			if vert == start_vert {
				// Single topological unit needs 2 faces added to be split
				let new_vert = self.vert_pos.len() as i32;
				self.vert_pos.push(self.vert_pos[end_vert as usize]);
				if self.vert_normal.len() > 0 {
					self.vert_normal.push(self.vert_normal[end_vert as usize]);
				}

				current = self.halfedge.pair(next_halfedge(current));
				let opposite = self.halfedge.pair(next_halfedge(edge));

				self.update_vert(new_vert, current, opposite);

				let mut new_halfedge = self.halfedge.len() as i32;
				let mut old_face = current / 3;
				let mut outside_vert = self.halfedge.start(current);
				self.halfedge.push(end_vert, -1, end_prop);
				self.halfedge.push(new_vert, -1, end_prop);
				self.halfedge
					.push(outside_vert, -1, self.halfedge.prop(current));
				self.pair_up(new_halfedge + 2, self.halfedge.pair(current));
				self.pair_up(new_halfedge + 1, current);
				if self.mesh_relation.tri_ref.len() > 0 {
					self.mesh_relation
						.tri_ref
						.push(self.mesh_relation.tri_ref[old_face as usize]);
				}
				if self.face_normal.len() > 0 {
					self.face_normal.push(self.face_normal[old_face as usize]);
				}

				new_halfedge += 3;
				old_face = opposite / 3;
				outside_vert = self.halfedge.start(opposite);
				self.halfedge.push(new_vert, -1, end_prop); // fix prop
				self.halfedge.push(end_vert, -1, end_prop);
				self.halfedge
					.push(outside_vert, -1, self.halfedge.prop(opposite));
				self.pair_up(new_halfedge + 2, self.halfedge.pair(opposite));
				self.pair_up(new_halfedge + 1, opposite);
				self.pair_up(new_halfedge, new_halfedge - 3);
				if self.mesh_relation.tri_ref.len() > 0 {
					self.mesh_relation
						.tri_ref
						.push(self.mesh_relation.tri_ref[old_face as usize]);
				}
				if self.face_normal.len() > 0 {
					self.face_normal.push(self.face_normal[old_face as usize]);
				}

				break;
			}

			current = self.halfedge.pair(next_halfedge(current));
		}

		if current == edge {
			// Separate topological unit needs no new faces to be split
			let new_vert = self.vert_pos.len() as i32;
			self.vert_pos.push(self.vert_pos[end_vert as usize]);
			if self.vert_normal.len() > 0 {
				self.vert_normal.push(self.vert_normal[end_vert as usize]);
			}

			self.for_vert_mut(next_halfedge(current), |myself, e| {
				myself.halfedge.set_start(e, new_vert);
				myself.halfedge.set_end(myself.halfedge.pair(e), new_vert);
			});
		}

		// Orbit startVert
		let pair = self.halfedge.pair(edge);
		current = self.halfedge.pair(next_halfedge(pair));
		while current != pair {
			let vert = self.halfedge.start(current);
			if vert == end_vert {
				break; //connected: not a pinched vert
			}

			current = self.halfedge.pair(next_halfedge(current));
		}

		if current == pair {
			// Split the pinched vert the previous split created.
			let new_vert = self.vert_pos.len() as i32;
			self.vert_pos.push(self.vert_pos[end_vert as usize]);
			if self.vert_normal.len() > 0 {
				self.vert_normal.push(self.vert_normal[end_vert as usize]);
			}

			self.for_vert_mut(next_halfedge(current), |myself, e| {
				myself.halfedge.set_start(e, new_vert);
				myself.halfedge.set_end(myself.halfedge.pair(e), new_vert);
			});
		}
	}

	fn pair_up(&mut self, edge0: i32, edge1: i32) {
		self.halfedge.set_pair(edge0, edge1);
		self.halfedge.set_pair(edge1, edge0);
	}

	// Traverses CW around startEdge.endVert from startEdge to endEdge
	// (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
	// to vert instead.
	fn update_vert(&mut self, vert: i32, start_edge: i32, end_edge: i32) {
		let mut current = start_edge;
		while current != end_edge {
			self.halfedge.set_end(current, vert);
			current = next_halfedge(current);
			self.halfedge.set_start(current, vert);
			current = self.halfedge.pair(current);
			debug_assert!(current != start_edge, "infinite loop in decimator!");
		}
	}

	// In the event that the edge collapse would create a non-manifold edge,
	// instead we duplicate the two verts and attach the manifolds the other way
	// across this edge.
	fn form_loop(&mut self, current: i32, end: i32) {
		let start_vert = self.vert_pos.len() as i32;
		self.vert_pos
			.push(self.vert_pos[self.halfedge.start(current) as usize]);
		let end_vert = self.vert_pos.len() as i32;
		self.vert_pos
			.push(self.vert_pos[self.halfedge.end(current) as usize]);

		let old_match = self.halfedge.pair(current);
		let new_match = self.halfedge.pair(end);

		self.update_vert(start_vert, old_match, new_match);
		self.update_vert(end_vert, end, current);

		self.pair_up(current, new_match);
		self.pair_up(end, old_match);

		self.remove_if_folded(end);
	}

	fn collapse_tri(&mut self, tri_edge: &Vector3<i32>) {
		if self.halfedge.pair(tri_edge[1]) == -1 {
			return;
		}
		let pair1 = self.halfedge.pair(tri_edge[1]);
		let pair2 = self.halfedge.pair(tri_edge[2]);
		self.pair_up(pair1, pair2);
		for i in 0..3 {
			self.halfedge
				.set(tri_edge[i], -1, -1, self.halfedge.prop(tri_edge[i]));
		}
	}

	fn remove_if_folded(&mut self, edge: i32) {
		let tri0_edge = tri_of(edge);
		let tri1_edge = tri_of(self.halfedge.pair(edge));
		if self.halfedge.pair(tri0_edge[1]) == -1 {
			return;
		}
		if self.halfedge.start(tri0_edge[2]) == self.halfedge.start(tri1_edge[2]) {
			if self.halfedge.pair(tri0_edge[1]) == tri1_edge[2] {
				if self.halfedge.pair(tri0_edge[2]) == tri1_edge[1] {
					for i in 0..3 {
						self.vert_pos[self.halfedge.start(tri0_edge[i]) as usize] =
							Point3::new(f64::NAN, f64::NAN, f64::NAN);
					}
				} else {
					self.vert_pos[self.halfedge.start(tri0_edge[1]) as usize] =
						Point3::new(f64::NAN, f64::NAN, f64::NAN);
				}
			} else {
				if self.halfedge.pair(tri0_edge[2]) == tri1_edge[1] {
					self.vert_pos[self.halfedge.start(tri1_edge[1]) as usize] =
						Point3::new(f64::NAN, f64::NAN, f64::NAN);
				}
			}

			self.pair_up(
				self.halfedge.pair(tri0_edge[1]),
				self.halfedge.pair(tri1_edge[2]),
			);
			self.pair_up(
				self.halfedge.pair(tri0_edge[2]),
				self.halfedge.pair(tri1_edge[1]),
			);

			for i in 0..3 {
				self.halfedge.set(tri0_edge[i], -1, -1, -1);
				self.halfedge.set(tri1_edge[i], -1, -1, -1);
			}
		}
	}

	///Collapses the given edge by removing startVert - returns false if the edge
	///cannot be collapsed. May split the mesh topologically if the collapse would
	///have resulted in a 4-manifold edge. Do not collapse an edge if startVert is
	///pinched - the vert would be marked NaN, but other edges could still be
	///pointing to it.
	fn collapse_edge(
		&mut self,
		edge: i32,
		edges: &mut Vec<i32>,
		mut tol: f64,
		first_new_vert: i32,
	) -> bool {
		if tol < 0.0 {
			tol = self.epsilon;
		}

		let pair = self.halfedge.pair(edge);
		if pair < 0 {
			return false;
		}

		let tri0_edge = tri_of(edge);
		let tri1_edge = tri_of(pair);
		let start_vert = self.halfedge.start(tri0_edge[0]);
		let end_vert = self.halfedge.start(tri0_edge[1]);

		let p_new = self.vert_pos[end_vert as usize];
		let p_old = self.vert_pos[start_vert as usize];
		let delta = p_new - p_old;
		// We don't check that startVert is still new here - it may have been
		// collapsed to a different neighbor. However, it's still fine to collapse it
		// further, as it's still only collapsing its own original neighbors together,
		// which can't stack errors arbitrarily far.
		let max_len = if end_vert < first_new_vert {
			tol * tol
		} else {
			self.epsilon * self.epsilon
		};
		let short_edge = delta.magnitude_squared() < max_len;

		// Orbit startVert
		let mut start = self.halfedge.pair(tri1_edge[1]);
		if !short_edge {
			let mut current = start;
			let mut ref_check = self.mesh_relation.tri_ref[(pair / 3) as usize];
			let mut p_last = self.vert_pos[self.halfedge.start(tri1_edge[2]) as usize];
			while current != tri1_edge[0] {
				current = next_halfedge(current);
				let p_next = self.vert_pos[self.halfedge.end(current) as usize];
				let tri = (current / 3) as usize;
				let tri_ref = self.mesh_relation.tri_ref[tri];
				let projection = get_axis_aligned_projection(self.face_normal[tri]);
				// Don't collapse if the edge is not redundant (this may have changed due
				// to the collapse of neighbors).
				if !tri_ref.same_face(&ref_check) {
					let old_ref = ref_check;
					ref_check = self.mesh_relation.tri_ref[(edge / 3) as usize];
					if !tri_ref.same_face(&ref_check) {
						return false;
					}

					if tri_ref.mesh_id != old_ref.mesh_id
						|| tri_ref.face_id != old_ref.face_id
						|| self.face_normal[(pair / 3) as usize].dot(&self.face_normal[tri]) < -0.5
					{
						// Restrict collapse to colinear edges when the edge separates faces
						// or the edge is sharp. This ensures large shifts are not introduced
						// parallel to the tangent plane.
						if ccw(
							projection * p_last,
							projection * p_old,
							projection * p_new,
							tol,
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
					self.epsilon,
				) < 0
				{
					return false;
				}

				p_last = p_next;
				current = self.halfedge.pair(current);
			}
		}

		// Orbit endVert
		{
			let mut current = self.halfedge.pair(tri0_edge[1]);
			while current != tri1_edge[2] {
				current = next_halfedge(current);
				edges.push(current);
				current = self.halfedge.pair(current);
			}
		}

		// Remove toRemove.startVert and replace with endVert.
		self.vert_pos[start_vert as usize] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
		self.collapse_tri(&tri1_edge);

		// Orbit startVert
		let tri0 = (edge / 3) as usize;
		let tri1 = (pair / 3) as usize;
		let mut current = start;
		while current != tri0_edge[2] {
			current = next_halfedge(current);

			if self.num_prop() > 0 {
				// Update the shifted triangles to the vertBary of endVert
				let tri = (current / 3) as usize;
				if self.mesh_relation.tri_ref[tri].same_face(&self.mesh_relation.tri_ref[tri0]) {
					self.halfedge
						.set_prop(current, self.halfedge.prop(next_halfedge(edge)));
				} else if self.mesh_relation.tri_ref[tri]
					.same_face(&self.mesh_relation.tri_ref[tri1])
				{
					self.halfedge.set_prop(current, self.halfedge.prop(pair));
				}
			}

			let vert = self.halfedge.end(current);
			let next = self.halfedge.pair(current);
			for i in 0..edges.len() {
				if vert == self.halfedge.end(edges[i]) {
					self.form_loop(edges[i], current);
					start = next;
					edges.truncate(i);
					break;
				}
			}

			current = next;
		}

		self.update_vert(end_vert, start, tri0_edge[2]);
		self.collapse_tri(&tri0_edge);
		self.remove_if_folded(start);
		true
	}

	fn recursive_edge_swap(
		&mut self,
		edge: i32,
		tag: &mut i32,
		visited: &mut [i32],
		edge_swap_stack: &mut Vec<i32>,
		edges: &mut Vec<i32>,
	) {
		if edge < 0 {
			return;
		}
		let pair = self.halfedge.pair(edge);
		if pair < 0 {
			return;
		}

		// avoid infinite recursion
		if visited[edge as usize] == *tag && visited[pair as usize] == *tag {
			return;
		}

		let tri0_edge = tri_of(edge);
		let tri1_edge = tri_of(pair);

		let projection = get_axis_aligned_projection(self.face_normal[(edge / 3) as usize]);
		let mut v = [Point2::default(); 4];
		for i in 0..3 {
			v[i] = projection * self.vert_pos[self.halfedge.start(tri0_edge[i]) as usize];
		}

		// Only operate on the long edge of a degenerate triangle.
		if ccw(v[0], v[1], v[2], self.tolerance) > 0 || !is_01_longest(v[0], v[1], v[2]) {
			return;
		}

		// Switch to neighbor's projection.
		let projection = get_axis_aligned_projection(self.face_normal[(pair / 3) as usize]);
		for i in 0..3 {
			v[i] = projection * self.vert_pos[self.halfedge.start(tri0_edge[i]) as usize];
		}

		v[3] = projection * self.vert_pos[self.halfedge.start(tri1_edge[2]) as usize];

		let swap_edge = |myself: &mut MeshBoolImpl| {
			// The 0-verts are swapped to the opposite 2-verts.
			let v0 = myself.halfedge.start(tri0_edge[2]);
			let v1 = myself.halfedge.start(tri1_edge[2]);
			myself.halfedge.set_start(tri0_edge[0], v1);
			myself.halfedge.set_end(tri0_edge[2], v1);
			myself.halfedge.set_start(tri1_edge[0], v0);
			myself.halfedge.set_end(tri1_edge[2], v0);
			myself.pair_up(tri0_edge[0], myself.halfedge.pair(tri1_edge[2]));
			myself.pair_up(tri1_edge[0], myself.halfedge.pair(tri0_edge[2]));
			myself.pair_up(tri0_edge[2], tri1_edge[2]);
			// Both triangles are now subsets of the neighboring triangle.
			let tri0 = (tri0_edge[0] / 3) as usize;
			let tri1 = (tri1_edge[0] / 3) as usize;
			myself.face_normal[tri0] = myself.face_normal[tri1];
			myself.mesh_relation.tri_ref[tri0] = myself.mesh_relation.tri_ref[tri1];
			let l01 = distance(&v[1], &v[0]);
			let l02 = distance(&v[2], &v[0]);
			let a = (l02 / l01).clamp(0.0, 1.0);
			// Update properties if applicable
			if myself.properties.len() > 0 {
				myself
					.halfedge
					.set_prop(tri0_edge[1], myself.halfedge.prop(tri1_edge[0]));
				myself
					.halfedge
					.set_prop(tri0_edge[0], myself.halfedge.prop(tri1_edge[2]));
				myself
					.halfedge
					.set_prop(tri0_edge[2], myself.halfedge.prop(tri1_edge[2]));
				let num_prop = myself.num_prop();
				let new_prop = myself.properties.len() / num_prop;
				let prop_idx0 = myself.halfedge.prop(tri1_edge[0]) as usize;
				let prop_idx1 = myself.halfedge.prop(tri1_edge[1]) as usize;
				for p in 0..num_prop {
					myself.properties.push(
						a * myself.properties[num_prop * prop_idx0 + p]
							+ (1.0 - a) * myself.properties[num_prop * prop_idx1 + p],
					);
				}

				myself.halfedge.set_prop(tri1_edge[0], new_prop as i32);
				myself.halfedge.set_prop(tri0_edge[2], new_prop as i32);
			}

			// if the new edge already exists, duplicate the verts and split the mesh.
			let mut current = myself.halfedge.pair(tri1_edge[0]);
			let end_vert = myself.halfedge.end(tri1_edge[1]);
			while current != tri0_edge[1] {
				current = next_halfedge(current);
				if myself.halfedge.end(current) == end_vert {
					myself.form_loop(tri0_edge[2], current);
					myself.remove_if_folded(tri0_edge[2]);
					return;
				}

				current = myself.halfedge.pair(current);
			}
		};

		// Only operate if the other triangles are not degenerate.
		if ccw(v[1], v[0], v[3], self.tolerance) <= 0 {
			if !is_01_longest(v[1], v[0], v[3]) {
				return;
			}
			// Two facing, long-edge degenerates can swap.
			swap_edge(self);
			let e23 = v[3] - v[2];
			if e23.magnitude_squared() < self.tolerance.powi(2) {
				*tag += 1;
				self.collapse_edge(tri0_edge[2], edges, -1.0, 0);
				edges.truncate(0);
			} else {
				visited[edge as usize] = *tag;
				visited[pair as usize] = *tag;
				edge_swap_stack.extend([tri1_edge[1], tri1_edge[0], tri0_edge[1], tri0_edge[0]]);
			}

			return;
		} else if ccw(v[0], v[3], v[2], self.tolerance) <= 0
			|| ccw(v[1], v[2], v[3], self.tolerance) <= 0
		{
			return;
		}

		//normal path
		swap_edge(self);
		visited[edge as usize] = *tag;
		visited[pair as usize] = *tag;
		edge_swap_stack.extend([
			self.halfedge.pair(tri1_edge[0]),
			self.halfedge.pair(tri0_edge[1]),
		]);
	}

	fn split_pinched_verts(&mut self) {
		let nb_edges = self.halfedge.len();

		let mut vert_processed = vec![false; self.num_vert()];
		let mut halfedge_processed = vec![false; nb_edges];
		for i in 0..nb_edges as i32 {
			if halfedge_processed[i as usize] {
				continue;
			}
			let mut vert = self.halfedge.start(i);
			if vert == -1 {
				continue;
			}
			if vert_processed[vert as usize] {
				self.vert_pos.push(self.vert_pos[vert as usize]);
				vert = (self.num_vert() - 1) as i32;
				self.for_vert_mut(i, |myself, current| {
					halfedge_processed[current as usize] = true;
					myself.halfedge.set_start(current, vert);
					myself.halfedge.set_end(myself.halfedge.pair(current), vert);
				});
			} else {
				vert_processed[vert as usize] = true;
				self.for_vert_mut(i, |_, current| {
					halfedge_processed[current as usize] = true;
				});
			}
		}
	}

	fn dedupe_edges(&mut self) {
		loop {
			//DedupeEdge

			let nb_edges = self.halfedge.len();
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
						let start_vert = self.halfedge.start(i as i32);
						let end_vert = self.halfedge.end(i as i32);
						if start_vert == -1 || end_vert == -1 {
							continue;
						}
						// we want to keep the allocation
						end_verts.clear();
						end_vert_set.clear();

						// first iteration, populate entries
						// this makes sure we always report the same set of entries
						self.for_vert(i as i32, |current| {
							local[current as usize] = true;
							let start_vert = self.halfedge.start(current);
							let end_v = self.halfedge.end(current);
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
						self.for_vert(i as i32, |current| {
							let start_vert = self.halfedge.start(current);
							let end_v = self.halfedge.end(current);
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
				self.dedupe_edge(i as i32);
				num_flagged += 1;
			}

			if num_flagged == 0 {
				break;
			}
		}
	}
}
