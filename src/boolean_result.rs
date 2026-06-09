use crate::MeshBoolError;
use crate::boolean3::Boolean3;
use crate::common::{AABB, OpType, OrderedF64};
use crate::meshboolimpl::{MESH_ID_COUNTER, MeshBoolImpl};
use crate::parallel::{
	copy_if, exclusive_scan_transformed, gather, gather_transformed, inclusive_scan,
};
use crate::shared::{Halfedge, Halfedges, TriRef, get_barycentric, next_halfedge};
use crate::utils::{atomic_add_i32, next3_i32, prev3_i32};
use crate::vec::{partition, vec_resize_nofill, vec_uninit};
use nalgebra::{Matrix3, Point3, Vector3, Vector4};
use std::collections::BTreeMap;
use std::mem;
use std::ops::Deref;
use std::sync::atomic::Ordering;

#[cfg(feature = "test_thoroughly")]
use crate::test::get_intermediate_checks;

fn abs_sum(a: i32, b: i32) -> i32 {
	a.abs() + b.abs()
}

struct DuplicateVerts<'a> {
	vert_pos_r: &'a mut [Point3<f64>],
	inclusion: &'a [i32],
	vert_r: &'a [i32],
	vert_pos_p: &'a [Point3<f64>],
}

impl<'a> DuplicateVerts<'a> {
	fn call(&mut self, vert: usize) {
		let n = self.inclusion[vert].abs();
		for i in 0..n {
			self.vert_pos_r[(self.vert_r[vert] + i) as usize] = self.vert_pos_p[vert];
		}
	}
}

struct CountVerts<'a> {
	halfedges: &'a Halfedges,
	count: &'a mut [i32],
	inclusion: &'a [i32],
}

impl<'a> CountVerts<'a> {
	fn call(&mut self, i: usize) {
		for j in 0..3 {
			self.count[i] +=
				(self.inclusion[self.halfedges.start((3 * i + j) as i32) as usize]).abs();
		}
	}
}

struct CountNewVerts<'a, const INVERTED: bool, const ATOMIC: bool> {
	count_p: &'a mut [i32],
	count_q: &'a mut [i32],
	i12: &'a [i32],
	pq: &'a [[i32; 2]],
	halfedges: &'a Halfedges,
}

impl<'a, const INVERTED: bool, const ATOMIC: bool> CountNewVerts<'a, INVERTED, ATOMIC> {
	fn call(&mut self, idx: usize) {
		let edge_p = self.pq[idx][if INVERTED { 1 } else { 0 }];
		let face_q = self.pq[idx][if INVERTED { 0 } else { 1 }];
		let inclusion = self.i12[idx].abs();

		if ATOMIC {
			unsafe {
				atomic_add_i32(&mut self.count_q[face_q as usize], inclusion);
				atomic_add_i32(&mut self.count_p[(edge_p / 3) as usize], inclusion);
				atomic_add_i32(
					&mut self.count_p[(self.halfedges.pair(edge_p) / 3) as usize],
					inclusion,
				);
			}
		} else {
			self.count_q[face_q as usize] += inclusion;
			self.count_p[(edge_p / 3) as usize] += inclusion;
			self.count_p[(self.halfedges.pair(edge_p) / 3) as usize] += inclusion;
		}
	}
}

fn size_output(
	out_r: &mut MeshBoolImpl,
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	i03: &[i32],
	i30: &[i32],
	i12: Vec<i32>,
	i21: Vec<i32>,
	p1q2: &[[i32; 2]],
	p2q1: &[[i32; 2]],
	invert_q: bool,
) -> (Vec<i32>, Vec<i32>) {
	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	let mut sides_per_face_pq = vec![0; in_p.num_tri() + in_q.num_tri()];
	// note: numFaceR <= facePQ2R.size() = sidesPerFacePQ.size() + 1

	let (mut sides_per_face_p, mut sides_per_face_q) =
		sides_per_face_pq.split_at_mut(in_p.num_tri());

	for i in 0..in_p.halfedge.len() / 3 {
		CountVerts {
			halfedges: &in_p.halfedge,
			count: &mut sides_per_face_p,
			inclusion: i03,
		}
		.call(i);
	}
	for i in 0..in_q.halfedge.len() / 3 {
		CountVerts {
			halfedges: &in_q.halfedge,
			count: &mut sides_per_face_q,
			inclusion: i30,
		}
		.call(i);
	}

	for i in 0..i12.len() {
		CountNewVerts::<false, false> {
			count_p: &mut sides_per_face_p,
			count_q: &mut sides_per_face_q,
			i12: &i12,
			pq: p1q2,
			halfedges: &in_p.halfedge,
		}
		.call(i);
	}
	for i in 0..i21.len() {
		CountNewVerts::<true, false> {
			count_p: &mut sides_per_face_q,
			count_q: &mut sides_per_face_p,
			i12: &i21,
			pq: p2q1,
			halfedges: &in_q.halfedge,
		}
		.call(i);
	}

	let mut face_pq2r: Vec<i32> = vec![0; in_p.num_tri() + in_q.num_tri() + 1];
	inclusive_scan(
		sides_per_face_pq.iter().map(|&x| if x > 0 { 1 } else { 0 }),
		&mut face_pq2r[1..],
	);
	let num_face_r = face_pq2r.pop().unwrap() as usize;

	unsafe {
		vec_resize_nofill(&mut out_r.face_normal, num_face_r);
	}

	let mut tmp_buffer = unsafe { vec_uninit(out_r.face_normal.len()) };

	let face_ids_p = (0..in_p.face_normal.len()).map(|i| {
		if sides_per_face_pq[i] > 0 {
			i
		} else {
			usize::MAX
		}
	});

	let next = copy_if(face_ids_p, &mut tmp_buffer, |v| v != usize::MAX);

	gather(
		&tmp_buffer[..next],
		&in_p.face_normal,
		&mut out_r.face_normal,
	);

	let face_ids_q = (0..in_q.face_normal.len()).map(|i| {
		if sides_per_face_pq[i + in_p.face_normal.len()] > 0 {
			i
		} else {
			usize::MAX
		}
	});

	let end = next + copy_if(face_ids_q, &mut tmp_buffer[next..], &|v| v != usize::MAX);

	if invert_q {
		gather_transformed(
			&tmp_buffer[next..end],
			&in_q.face_normal,
			&mut out_r.face_normal[next..],
			|normal: Vector3<f64>| -normal,
		);
	} else {
		gather(
			&tmp_buffer[next..end],
			&in_q.face_normal,
			&mut out_r.face_normal[next..],
		);
	}

	sides_per_face_pq.retain(|&v| v != 0);
	let mut face_edge = vec![0; sides_per_face_pq.len() + 1];
	inclusive_scan(sides_per_face_pq.into_iter(), &mut face_edge[1..]);

	(face_edge, face_pq2r)
}

#[derive(Copy, Clone, Debug)]
struct EdgePos {
	edge_pos: f64,
	vert: i32,
	collision_id: i32,
	is_start: bool,
}

fn sort_edge_pos(edge_pos: &mut [EdgePos]) {
	edge_pos.sort_by_key(|i| (OrderedF64(i.edge_pos), i.collision_id));
}

fn add_new_edge_verts(
	// we need concurrent_map because we will be adding things concurrently
	edges_p: &mut BTreeMap<i32, Vec<EdgePos>>,
	edges_new: &mut BTreeMap<(i32, i32), Vec<EdgePos>>,
	p1q2: &[[i32; 2]],
	i12: &[i32],
	v12_r: &[i32],
	halfedge_p: &Halfedges,
	forward: bool,
	offset: usize,
) {
	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	// For each edge of P that intersects a face of Q (p1q2), add this vertex to
	// P's corresponding edge vector and to the two new edges, which are
	// intersections between the face of Q and the two faces of P attached to the
	// edge. The direction and duplicity are given by i12, while v12R remaps to
	// the output vert index. When forward is false, all is reversed.
	for i in 0..p1q2.len() {
		let edge_p = p1q2[i][if forward { 0 } else { 1 }];
		let face_q = p1q2[i][if forward { 1 } else { 0 }];
		let vert = v12_r[i];
		let inclusion = i12[i];

		let mut key_right = (halfedge_p.pair(edge_p) / 3, face_q);
		if !forward {
			mem::swap(&mut key_right.0, &mut key_right.1);
		}

		let mut key_left = (edge_p / 3, face_q);
		if !forward {
			mem::swap(&mut key_left.0, &mut key_left.1);
		}

		let direction = inclusion < 0;
		for k in 0..3 {
			let tuple = match k {
				0 => (direction, edges_p.entry(edge_p).or_default()),
				1 => (
					direction ^ !forward,
					edges_new.entry(key_right).or_default(),
				), //revert if not forward
				2 => (direction ^ forward, edges_new.entry(key_left).or_default()),
				_ => unreachable!(),
			};

			for j in 0..inclusion.abs() {
				tuple.1.push(EdgePos {
					edge_pos: 0.0,
					vert: vert + j,
					collision_id: (i + offset) as i32,
					is_start: tuple.0,
				});
			}
		}
	}
}

fn pair_up(edge_pos: &mut [EdgePos], mut f: impl FnMut(Halfedge)) {
	// Pair start vertices with end vertices to form edges. The choice of pairing
	// is arbitrary for the manifoldness guarantee, but must be ordered to be
	// geometrically valid. If the order does not go start-end-start-end... then
	// the input and output are not geometrically valid and this algorithm becomes
	// a heuristic.
	debug_assert!(
		edge_pos.len() % 2 == 0,
		"Non-manifold edge! Not an even number of points."
	);
	let n_edges = edge_pos.len() / 2;
	let middle = partition(edge_pos, |x| x.is_start);
	debug_assert!(middle == n_edges, "Non-manifold edge!");
	sort_edge_pos(&mut edge_pos[..middle]);
	sort_edge_pos(&mut edge_pos[middle..]);
	for i in 0..n_edges {
		f(Halfedge {
			start_vert: edge_pos[i].vert,
			end_vert: edge_pos[i + n_edges].vert,
			paired_halfedge: -1,
			prop_vert: 0,
		});
	}
}

fn append_partial_edges(
	out_r: &mut MeshBoolImpl,
	halfedge_r: &mut [Halfedge],
	whole_halfedge_p: &mut [bool],
	face_ptr_r: &mut [i32],
	edges_p: BTreeMap<i32, Vec<EdgePos>>,
	halfedge_ref: &mut [TriRef],
	in_p: &MeshBoolImpl,
	i03: &[i32],
	v_p2r: &[i32],
	face_p2r: &[i32],
	forward: bool,
) {
	// Each edge in the map is partially retained; for each of these, look up
	// their original verts and include them based on their winding number (i03),
	// while remapping them to the output using vP2R. Use the verts position
	// projected along the edge vector to pair them up, then distribute these
	// edges to their faces.
	let vert_pos_p = &in_p.vert_pos;
	let halfedge_p = &in_p.halfedge;

	// Per-iter cancel check; the caller's post-call IsCancelled discards the
	// partial outR.
	for (edge_p, mut edge_pos_p) in edges_p {
		sort_edge_pos(&mut edge_pos_p);
		let pair_p = halfedge_p.pair(edge_p);
		whole_halfedge_p[edge_p as usize] = false;
		whole_halfedge_p[pair_p as usize] = false;

		let v_start = halfedge_p.start(edge_p) as usize;
		let v_end = halfedge_p.end(edge_p) as usize;
		let edge_vec = vert_pos_p[v_end] - vert_pos_p[v_start];
		// Fill in the edge positions of the old points.
		for edge in edge_pos_p.iter_mut() {
			edge.edge_pos = out_r.vert_pos[edge.vert as usize].coords.dot(&edge_vec);
		}

		let mut inclusion = i03[v_start];
		let mut edge_pos = EdgePos {
			edge_pos: out_r.vert_pos[v_p2r[v_start] as usize]
				.coords
				.dot(&edge_vec),
			vert: v_p2r[v_start],
			collision_id: i32::MAX,
			is_start: inclusion > 0,
		};

		for _ in 0..inclusion.abs() {
			edge_pos_p.push(edge_pos);
			edge_pos.vert += 1;
		}

		inclusion = i03[v_end];
		edge_pos = EdgePos {
			edge_pos: out_r.vert_pos[v_p2r[v_end] as usize].coords.dot(&edge_vec),
			vert: v_p2r[v_end],
			collision_id: i32::MAX,
			is_start: inclusion < 0,
		};

		for _ in 0..inclusion.abs() {
			edge_pos_p.push(edge_pos);
			edge_pos.vert += 1;
		}

		// add halfedges to result
		let face_left_p = edge_p / 3;
		let face_left = face_p2r[face_left_p as usize];
		let face_right_p = pair_p / 3;
		let face_right = face_p2r[face_right_p as usize];
		// Negative inclusion means the halfedges are reversed, which means our
		// reference is now to the endVert instead of the startVert, which is one
		// position advanced CCW. This is only valid if this is a retained vert; it
		// will be ignored later if the vert is new.
		let forward_ref = TriRef {
			mesh_id: if forward { 0 } else { 1 },
			original_id: -1,
			face_id: face_left_p,
			coplanar_id: -1,
		};
		let backward_ref = TriRef {
			mesh_id: if forward { 0 } else { 1 },
			original_id: -1,
			face_id: face_right_p,
			coplanar_id: -1,
		};

		pair_up(&mut edge_pos_p, |mut e| {
			let forward_edge = face_ptr_r[face_left as usize];
			face_ptr_r[face_left as usize] += 1;
			let backward_edge = face_ptr_r[face_right as usize];
			face_ptr_r[face_right as usize] += 1;

			e.paired_halfedge = backward_edge;
			halfedge_r[forward_edge as usize] = e;
			halfedge_ref[forward_edge as usize] = forward_ref;

			mem::swap(&mut e.start_vert, &mut e.end_vert);
			e.paired_halfedge = forward_edge;
			halfedge_r[backward_edge as usize] = e;
			halfedge_ref[backward_edge as usize] = backward_ref;
		});
	}
}

fn append_new_edges(
	out_r: &mut MeshBoolImpl,
	halfedge_r: &mut [Halfedge],
	face_ptr_r: &mut [i32],
	edges_new: BTreeMap<(i32, i32), Vec<EdgePos>>,
	halfedge_ref: &mut [TriRef],
	face_pq2r: &[i32],
	num_face_p: usize,
) {
	// Pair up each edge's verts and distribute to faces based on indices in key.
	let vert_pos_r = &out_r.vert_pos;

	// Per-iter cancel check; the caller's post-call IsCancelled discards the
	// partial outR.
	for ((face_p, face_q), mut edge_pos) in edges_new {
		sort_edge_pos(&mut edge_pos);

		let mut bbox = AABB::default();
		for edge in edge_pos.iter() {
			bbox.union_point(vert_pos_r[edge.vert as usize]);
		}

		let size = bbox.size();
		// Order the points along their longest dimension.
		let i = if size.x > size.y && size.x > size.z {
			0
		} else if size.y > size.z {
			1
		} else {
			2
		};

		for edge in edge_pos.iter_mut() {
			edge.edge_pos = vert_pos_r[edge.vert as usize][i];
		}

		// add halfedges to result
		let face_left = face_pq2r[face_p as usize] as usize;
		let face_right = face_pq2r[num_face_p + face_q as usize] as usize;
		let forward_ref = TriRef {
			mesh_id: 0,
			original_id: -1,
			face_id: face_p,
			coplanar_id: -1,
		};
		let backward_ref = TriRef {
			mesh_id: 1,
			original_id: -1,
			face_id: face_q,
			coplanar_id: -1,
		};

		pair_up(&mut edge_pos, |mut e| {
			let forward_edge = face_ptr_r[face_left];
			face_ptr_r[face_left] += 1;
			let backward_edge = face_ptr_r[face_right];
			face_ptr_r[face_right] += 1;

			e.paired_halfedge = backward_edge;
			halfedge_r[forward_edge as usize] = e;
			halfedge_ref[forward_edge as usize] = forward_ref;

			mem::swap(&mut e.start_vert, &mut e.end_vert);
			e.paired_halfedge = forward_edge;
			halfedge_r[backward_edge as usize] = e;
			halfedge_ref[backward_edge as usize] = backward_ref;
		});
	}
}

fn append_whole_edges(
	face_ptr_r: &mut [i32],
	halfedges_r: &mut [Halfedge],
	halfedge_ref: &mut [TriRef],
	in_p: &MeshBoolImpl,
	whole_halfedge_p: Vec<bool>,
	i03: Vec<i32>,
	v_p2r: Vec<i32>,
	face_p2r: &[i32],
	forward: bool,
) {
	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	//(struct DuplicateHalfedges is inlined here)
	for idx in 0..in_p.halfedge.len() as i32 {
		if !whole_halfedge_p[idx as usize] {
			continue;
		}

		let mut start_vert = in_p.halfedge.start(idx);
		let mut end_vert = in_p.halfedge.start(next_halfedge(idx));
		if start_vert >= end_vert {
			continue;
		}
		let inclusion = i03[start_vert as usize];
		if inclusion == 0 {
			continue;
		}
		if inclusion < 0
		// reverse
		{
			mem::swap(&mut start_vert, &mut end_vert);
		}

		start_vert = v_p2r[start_vert as usize];
		end_vert = v_p2r[end_vert as usize];
		let prop_vert = in_p.halfedge.prop(idx);
		let pair = in_p.halfedge.pair(idx);
		let pair_prop_vert = in_p.halfedge.prop(pair);
		let face_left_p = idx / 3;
		let new_face = face_p2r[face_left_p as usize];
		let face_right_p = pair / 3;
		let face_right = face_p2r[face_right_p as usize];
		// Negative inclusion means the halfedges are reversed, which means our
		// reference is now to the endVert instead of the startVert, which is one
		// position advanced CCW.
		let forward_ref = TriRef {
			mesh_id: if forward { 0 } else { 1 },
			original_id: -1,
			face_id: face_left_p as i32,
			coplanar_id: -1,
		};
		let backward_ref = TriRef {
			mesh_id: if forward { 0 } else { 1 },
			original_id: -1,
			face_id: face_right_p,
			coplanar_id: -1,
		};

		for _ in 0..inclusion.abs() {
			let forward_edge = unsafe { atomic_add_i32(&mut face_ptr_r[new_face as usize], 1) };
			let backward_edge = unsafe { atomic_add_i32(&mut face_ptr_r[face_right as usize], 1) };

			halfedges_r[forward_edge as usize] = Halfedge {
				start_vert: start_vert,
				end_vert: end_vert,
				paired_halfedge: backward_edge,
				prop_vert,
			};
			halfedges_r[backward_edge as usize] = Halfedge {
				start_vert: end_vert,
				end_vert: start_vert,
				paired_halfedge: forward_edge,
				prop_vert: pair_prop_vert,
			};

			halfedge_ref[forward_edge as usize] = forward_ref;
			halfedge_ref[backward_edge as usize] = backward_ref;

			start_vert += 1;
			end_vert += 1;
		}
	}
}

struct MapTriRef<'a> {
	tri_ref_p: &'a [TriRef],
	tri_ref_q: &'a [TriRef],
	offset_q: i32,
}

impl<'a> MapTriRef<'a> {
	fn call(&self, tri_ref: &mut TriRef) {
		let tri = tri_ref.face_id as usize;
		let pq = tri_ref.mesh_id == 0;
		*tri_ref = if pq {
			self.tri_ref_p[tri]
		} else {
			self.tri_ref_q[tri]
		};

		if !pq {
			tri_ref.mesh_id += self.offset_q;
		}
	}
}

fn update_reference(
	out_r: &mut MeshBoolImpl,
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	invert_q: bool,
) {
	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	let offset_q = MESH_ID_COUNTER.load(Ordering::SeqCst) as i32;
	let num_tri = out_r.num_tri();
	for tri_ref in &mut out_r.mesh_relation.tri_ref[..num_tri] {
		MapTriRef {
			tri_ref_p: &in_p.mesh_relation.tri_ref,
			tri_ref_q: &in_q.mesh_relation.tri_ref,
			offset_q,
		}
		.call(tri_ref);
	}

	out_r
		.mesh_relation
		.mesh_id_transform
		.extend(&in_p.mesh_relation.mesh_id_transform);

	out_r
		.mesh_relation
		.mesh_id_transform
		.extend(in_q.mesh_relation.mesh_id_transform.iter().map(|pair| {
			let mut relation = *pair.1;
			relation.back_side ^= invert_q;
			(pair.0 + offset_q, relation)
		}));
}

struct Barycentric<'a> {
	uvw: &'a mut [Vector3<f64>],
	tri_ref: &'a [TriRef],
	vert_pos_p: &'a [Point3<f64>],
	vert_pos_q: &'a [Point3<f64>],
	vert_pos_r: &'a [Point3<f64>],
	halfedge_p: &'a Halfedges,
	halfedge_q: &'a Halfedges,
	halfedge_r: &'a Halfedges,
	epsilon: f64,
}

impl<'a> Barycentric<'a> {
	fn call(&mut self, tri: i32) {
		let ref_pq = self.tri_ref[tri as usize];
		if self.halfedge_r.start(3 * tri) < 0 {
			return;
		}

		let tri_pq = ref_pq.face_id;
		let pq = ref_pq.mesh_id == 0;
		let vert_pos = if pq {
			&self.vert_pos_p
		} else {
			&self.vert_pos_q
		};
		let halfedge = if pq {
			&self.halfedge_p
		} else {
			&self.halfedge_q
		};

		let mut tri_pos = Matrix3::default();
		for j in 0..3 {
			*tri_pos.column_mut(j as usize) =
				*vert_pos[halfedge.start(3 * tri_pq + j) as usize].deref();
		}

		for i in 0..3 {
			let vert = self.halfedge_r.start(3 * tri + i);
			self.uvw[(3 * tri + i) as usize] =
				get_barycentric(&self.vert_pos_r[vert as usize], &tri_pos, self.epsilon);
		}
	}
}

fn create_properties(
	out_r: &mut MeshBoolImpl,
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	invert_q: bool,
) {
	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	let num_prop_p = in_p.num_prop();
	let num_prop_q = in_q.num_prop();
	let num_prop = num_prop_p.max(num_prop_q);
	out_r.num_prop = num_prop as i32;
	if num_prop == 0 {
		return;
	}

	let num_tri = out_r.num_tri();
	let mut bary = unsafe { vec_uninit(out_r.halfedge.len()) };
	for tri in 0..num_tri {
		Barycentric {
			uvw: &mut bary,
			tri_ref: &out_r.mesh_relation.tri_ref,
			vert_pos_p: &in_p.vert_pos,
			vert_pos_q: &in_q.vert_pos,
			vert_pos_r: &out_r.vert_pos,
			halfedge_p: &in_p.halfedge,
			halfedge_q: &in_q.halfedge,
			halfedge_r: &out_r.halfedge,
			epsilon: out_r.epsilon,
		}
		.call(tri as i32);
	}

	let id_miss_prop = out_r.num_vert() as i32;
	let mut prop_idx: Vec<Vec<(Vector3<i32>, i32)>> = vec![Vec::new(); out_r.num_vert() + 1];
	let mut prop_miss_idx = [
		vec![-1; in_q.num_prop_vert()],
		vec![-1; in_p.num_prop_vert()],
	];

	out_r
		.properties
		.reserve_exact((out_r.num_vert() * num_prop).saturating_sub(out_r.properties.len()));
	let mut idx = 0;

	for tri in 0..num_tri as i32 {
		// Skip collapsed triangles
		if out_r.halfedge.start(3 * tri) < 0 {
			continue;
		}

		let tri_ref = out_r.mesh_relation.tri_ref[tri as usize];
		let pq = tri_ref.mesh_id == 0;
		let old_num_prop = (if pq { num_prop_p } else { num_prop_q }) as i32;
		let properties = if pq {
			&in_p.properties
		} else {
			&in_q.properties
		};
		let halfedge = if pq { &in_p.halfedge } else { &in_q.halfedge };

		// For Subtract, Q's triangles are flipped in the result, so Q's
		// world-frame vertex normals (slot 0..2 when hasNormals) need a sign
		// flip to point outward from the result's solid (into the cavity).
		// Check is per-source-triangle, not whole input - inQ may be a mixed
		// Boolean result.
		let negate_normals = !pq
			&& invert_q
			&& old_num_prop >= 3
			&& MeshBoolImpl::tri_has_normals(&in_q.mesh_relation, tri_ref.face_id);

		for i in 0..3 {
			let vert = out_r.halfedge.start(3 * tri + i);
			let uvw = &bary[(3 * tri + i) as usize];

			let mut key = Vector4::new(pq as i32, id_miss_prop, -1, -1);
			if old_num_prop > 0 {
				let mut edge = -2;
				for j in 0..3 {
					if uvw[j as usize] == 1.0 {
						// On a retained vert, the propVert must also match
						key[2] = halfedge.prop(3 * tri_ref.face_id + j);
						edge = -1;
						break;
					}

					if uvw[j as usize] == 0.0 {
						edge = j
					};
				}

				if edge >= 0 {
					// On an edge, both propVerts must match
					let p0 = halfedge.prop(3 * tri_ref.face_id + next3_i32(edge));
					let p1 = halfedge.prop(3 * tri_ref.face_id + prev3_i32(edge));
					key[1] = vert;
					key[2] = p0.min(p1);
					key[3] = p0.max(p1);
				} else if edge == -2 {
					key[1] = vert;
				}
			}

			if key.y == id_miss_prop && key.z >= 0 {
				// only key.x/key.z matters
				let entry = &mut prop_miss_idx[key.x as usize][key.z as usize];
				if *entry >= 0 {
					out_r.halfedge.set_prop(3 * tri + i, *entry);
					continue;
				}

				*entry = idx;
			} else {
				let bin = &mut prop_idx[key.y as usize];
				let mut b_found = false;
				for b in bin.iter() {
					if b.0 == Vector3::new(key.x, key.z, key.w) {
						b_found = true;
						out_r.halfedge.set_prop(3 * tri + i, b.1);
						break;
					}
				}

				if b_found {
					continue;
				}
				bin.push((Vector3::new(key.x, key.z, key.w), idx));
			}

			out_r.halfedge.set_prop(3 * tri + i, idx);
			idx += 1;
			for p in 0..num_prop {
				let p = p as i32;

				if p < old_num_prop {
					let mut old_props = Vector3::default();
					for j in 0..3 {
						old_props[j as usize] = properties
							[(old_num_prop * halfedge.prop(3 * tri_ref.face_id + j) + p) as usize];
					}

					let mut val = uvw.dot(&old_props);
					if negate_normals && p < 3 {
						val = -val;
					}
					out_r.properties.push(val);
				} else {
					out_r.properties.push(0.0);
				}
			}
		}
	}
}

impl<'a> Boolean3<'a> {
	pub fn result(&self, op: OpType) -> MeshBoolImpl {
		debug_assert!(
			self.expand_p == (op == OpType::Add),
			"Result op type not compatible with constructor op type."
		);
		let c1 = if op == OpType::Intersect { 0 } else { 1 };
		let c2 = if op == OpType::Add { 1 } else { 0 };
		let c3 = if op == OpType::Intersect { 1 } else { -1 };

		if self.in_p.status != MeshBoolError::NoError {
			let mut meshbool_impl = MeshBoolImpl::default();
			meshbool_impl.status = self.in_p.status;
			return meshbool_impl;
		}

		if self.in_q.status != MeshBoolError::NoError {
			let mut meshbool_impl = MeshBoolImpl::default();
			meshbool_impl.status = self.in_q.status;
			return meshbool_impl;
		}

		if self.in_p.is_empty() {
			if !self.in_q.is_empty() && op == OpType::Add {
				return self.in_q.clone();
			}

			return MeshBoolImpl::default();
		} else if self.in_q.is_empty() {
			if op == OpType::Intersect {
				return MeshBoolImpl::default();
			}

			return self.in_p.clone();
		}

		if !self.valid {
			let mut meshbool_impl = MeshBoolImpl::default();
			meshbool_impl.status = MeshBoolError::ResultTooLarge;
			return meshbool_impl;
		}

		let invert_q = op == OpType::Subtract;

		// Convert winding numbers to inclusion values based on operation type.
		let i12: Vec<_> = self.xv12.x12.iter().copied().map(|v| c3 * v).collect();
		let i21: Vec<_> = self.xv21.x12.iter().copied().map(|v| c3 * v).collect();
		let i03: Vec<_> = self.w03.iter().copied().map(|v| c1 + c3 * v).collect();
		let i30: Vec<_> = self.w30.iter().copied().map(|v| c2 + c3 * v).collect();

		let v_p2r = exclusive_scan_transformed(&i03, 0, &abs_sum);
		let mut num_vert_r = v_p2r.last().unwrap().abs() + i03.last().unwrap().abs();
		let n_pv = num_vert_r;

		let v_q2r = exclusive_scan_transformed(&i30, num_vert_r, &abs_sum);
		num_vert_r = abs_sum(*v_q2r.last().unwrap(), *i30.last().unwrap());
		let n_qv = num_vert_r - n_pv;

		let v12_r = if self.xv12.v12.len() == 0 {
			Vec::new()
		} else {
			let v12_r = exclusive_scan_transformed(&i12, num_vert_r, &abs_sum);
			num_vert_r = abs_sum(*v12_r.last().unwrap(), *i12.last().unwrap());
			v12_r
		};

		//let n12 = num_vert_r - n_pv - n_qv;

		let v21_r = if self.xv21.v12.len() == 0 {
			Vec::new()
		} else {
			let v21_r = exclusive_scan_transformed(&i21, num_vert_r, &abs_sum);
			num_vert_r = abs_sum(*v21_r.last().unwrap(), *i21.last().unwrap());
			v21_r
		};

		//let n21 = num_vert_r - n_pv - n_qv - n12;

		// Create the output Manifold
		let mut out_r = MeshBoolImpl::default();

		if num_vert_r == 0 {
			return out_r;
		}

		out_r.epsilon = self.in_p.epsilon.max(self.in_q.epsilon);
		out_r.tolerance = self.in_p.tolerance.max(self.in_q.tolerance);

		unsafe {
			vec_resize_nofill(&mut out_r.vert_pos, num_vert_r as usize);
		}
		// Add vertices, duplicating for inclusion numbers not in [-1, 1].
		// Retained vertices from P and Q:
		for vert in 0..self.in_p.num_vert() {
			DuplicateVerts {
				vert_pos_r: &mut out_r.vert_pos,
				inclusion: &i03,
				vert_r: &v_p2r,
				vert_pos_p: &self.in_p.vert_pos,
			}
			.call(vert);
		}
		for vert in 0..self.in_q.num_vert() {
			DuplicateVerts {
				vert_pos_r: &mut out_r.vert_pos,
				inclusion: &i30,
				vert_r: &v_q2r,
				vert_pos_p: &self.in_q.vert_pos,
			}
			.call(vert);
		}
		// New vertices created from intersections:
		for vert in 0..i12.len() {
			DuplicateVerts {
				vert_pos_r: &mut out_r.vert_pos,
				inclusion: &i12,
				vert_r: &v12_r,
				vert_pos_p: &self.xv12.v12,
			}
			.call(vert);
		}
		for vert in 0..i21.len() {
			DuplicateVerts {
				vert_pos_r: &mut out_r.vert_pos,
				inclusion: &i21,
				vert_r: &v21_r,
				vert_pos_p: &self.xv21.v12,
			}
			.call(vert);
		}

		// Build up new polygonal faces from triangle intersections. At this point the
		// calculation switches from parallel to serial.

		// Level 3

		// This key is the forward halfedge index of P or Q. Only includes intersected
		// edges.
		let mut edges_p: BTreeMap<i32, Vec<EdgePos>> = BTreeMap::new();
		let mut edges_q: BTreeMap<i32, Vec<EdgePos>> = BTreeMap::new();
		// This key is the face index of <P, Q>
		let mut edges_new: BTreeMap<(i32, i32), Vec<EdgePos>> = BTreeMap::new();

		add_new_edge_verts(
			&mut edges_p,
			&mut edges_new,
			&self.xv12.p1q2,
			&i12,
			&v12_r,
			&self.in_p.halfedge,
			true,
			0,
		);
		add_new_edge_verts(
			&mut edges_q,
			&mut edges_new,
			&self.xv21.p1q2,
			&i21,
			&v21_r,
			&self.in_q.halfedge,
			false,
			self.xv12.p1q2.len(),
		);

		drop(v12_r);
		drop(v21_r);

		// Level 4
		let (face_edge, face_pq2r) = size_output(
			&mut out_r,
			&self.in_p,
			&self.in_q,
			&i03,
			&i30,
			i12,
			i21,
			&self.xv12.p1q2,
			&self.xv21.p1q2,
			invert_q,
		);

		// This gets incremented for each halfedge that's added to a face so that the
		// next one knows where to slot in.
		let mut face_ptr_r = face_edge.clone();
		// Intersected halfedges are marked false.
		let mut whole_halfedge_p = vec![true; self.in_p.halfedge.len()];
		let mut whole_halfedge_q = vec![true; self.in_q.halfedge.len()];
		let face_edge_back = *face_edge.last().unwrap() as usize;
		// The halfedgeRef contains the data that will become triRef once the faces
		// are triangulated.
		let mut halfedge_ref = unsafe { vec_uninit(face_edge_back) };
		// Note that we are working with Vec<Halfedge> instead of Halfedges here,
		// since the faces can be arbitrary polygons before feeding into the
		// triangulator.
		let mut face_halfedges = unsafe { vec_uninit(face_edge_back) };

		append_partial_edges(
			&mut out_r,
			&mut face_halfedges,
			&mut whole_halfedge_p,
			&mut face_ptr_r,
			edges_p,
			&mut halfedge_ref,
			self.in_p,
			&i03,
			&v_p2r,
			&face_pq2r,
			true,
		);
		append_partial_edges(
			&mut out_r,
			&mut face_halfedges,
			&mut whole_halfedge_q,
			&mut face_ptr_r,
			edges_q,
			&mut halfedge_ref,
			self.in_q,
			&i30,
			&v_q2r,
			&face_pq2r[self.in_p.num_tri()..],
			false,
		);

		append_new_edges(
			&mut out_r,
			&mut face_halfedges,
			&mut face_ptr_r,
			edges_new,
			&mut halfedge_ref,
			&face_pq2r,
			self.in_p.num_tri(),
		);

		append_whole_edges(
			&mut face_ptr_r,
			&mut face_halfedges,
			&mut halfedge_ref,
			&self.in_p,
			whole_halfedge_p,
			i03,
			v_p2r,
			&face_pq2r[..self.in_p.num_tri()],
			true,
		);
		append_whole_edges(
			&mut face_ptr_r,
			&mut face_halfedges,
			&mut halfedge_ref,
			&self.in_q,
			whole_halfedge_q,
			i30,
			v_q2r,
			&face_pq2r[self.in_p.num_tri()..],
			false,
		);

		drop(face_ptr_r);
		drop(face_pq2r);

		// Level 6
		out_r.face2tri(&face_edge, &face_halfedges, &halfedge_ref, false);

		out_r.reorder_halfedges();

		#[cfg(feature = "test_thoroughly")]
		if get_intermediate_checks() {
			debug_assert!(out_r.is_manifold(), "triangulated mesh is not manifold!");
		}

		create_properties(&mut out_r, &self.in_p, &self.in_q, invert_q);

		update_reference(&mut out_r, &self.in_p, &self.in_q, invert_q);

		out_r.simplify_topology(n_pv + n_qv);
		out_r.remove_unreferenced_verts();

		#[cfg(feature = "test_thoroughly")]
		if get_intermediate_checks() {
			debug_assert!(out_r.is_2_manifold(), "simplified mesh is not 2-manifold!");
		}

		out_r.calculate_bbox();
		out_r.sort_geometry();
		out_r.increment_mesh_ids();

		out_r
	}
}
