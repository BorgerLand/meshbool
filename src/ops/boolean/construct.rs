use crate::MeshBool;
use crate::halfedge::{Halfedge, Halfedges, next_halfedge};
use crate::mesh_relations::{TriRelation, tri_has_normals};
use crate::spatial::aabb::Box3D;
use crate::util::hash_table::DeterministicMap;
use crate::util::math::{atomic_add, get_barycentric, next3_i32, prev3_i32};
use crate::util::num_convert::OrderedF64;
use crate::util::vec_ext;
use nalgebra::{Matrix3, Point3, Vector3, Vector4};
use std::collections::BTreeMap;
use std::mem;
use std::ops::Deref;

pub struct DuplicateVerts<'a> {
	pub vert_pos_r: &'a mut [Point3<f64>],
	pub inclusion: &'a [i32],
	pub vert_r: &'a [i32],
	pub vert_pos_p: &'a [Point3<f64>],
}

impl<'a> DuplicateVerts<'a> {
	pub fn call(&mut self, vert: usize) {
		let n = self.inclusion[vert].abs();
		for i in 0..n {
			self.vert_pos_r[(self.vert_r[vert] + i) as usize] = self.vert_pos_p[vert];
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct EdgePos {
	edge_pos: f64,
	vert: i32,
	collision_id: i32,
	is_start: bool,
}

pub fn add_new_edge_verts(
	// we need concurrent_map because we will be adding things concurrently
	edges_old: &mut BTreeMap<i32, Vec<EdgePos>>,
	edges_new: &mut BTreeMap<(i32, i32), Vec<EdgePos>>,
	p1q2: &[[i32; 2]],
	i12: &[i32],
	v12_r: Vec<i32>,
	halfedge_p: &Halfedges,
	forward: bool,
	offset: usize,
) {
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
				0 => (direction, edges_old.entry(edge_p).or_default()),
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

struct CountNewVerts<'a, const INVERTED: bool> {
	count_p: &'a mut [i32],
	count_q: &'a mut [i32],
	i12: &'a [i32],
	pq: &'a [[i32; 2]],
	halfedges: &'a Halfedges,
}

impl<'a, const INVERTED: bool> CountNewVerts<'a, INVERTED> {
	fn call(&mut self, idx: usize) {
		let edge_p = self.pq[idx][if INVERTED { 1 } else { 0 }];
		let face_q = self.pq[idx][if INVERTED { 0 } else { 1 }];
		let inclusion = self.i12[idx].abs();

		self.count_q[face_q as usize] += inclusion;
		self.count_p[(edge_p / 3) as usize] += inclusion;
		self.count_p[(self.halfedges.pair(edge_p) / 3) as usize] += inclusion;
	}
}

pub fn size_sides_per_face_pq(
	halfedge_p: &Halfedges,
	halfedge_q: &Halfedges,
	i03: &[i32],
	i30: &[i32],
	i12: Vec<i32>,
	i21: Vec<i32>,
	p1q2: Vec<[i32; 2]>,
	p2q1: Vec<[i32; 2]>,
) -> Vec<i32> {
	let mut sides_per_face_pq = vec![0; halfedge_p.num_tri() + halfedge_q.num_tri()];
	// note: numFaceR <= facePQ2R.size() = sidesPerFacePQ.size() + 1

	let (mut sides_per_face_p, mut sides_per_face_q) =
		sides_per_face_pq.split_at_mut(halfedge_p.num_tri());

	let mut count_p = CountVerts {
		halfedges: halfedge_p,
		count: &mut sides_per_face_p,
		inclusion: i03,
	};
	for i in 0..halfedge_p.len() / 3 {
		count_p.call(i);
	}
	let mut count_q = CountVerts {
		halfedges: halfedge_q,
		count: &mut sides_per_face_q,
		inclusion: i30,
	};
	for i in 0..halfedge_q.len() / 3 {
		count_q.call(i);
	}

	let mut count12 = CountNewVerts::<false> {
		count_p: &mut sides_per_face_p,
		count_q: &mut sides_per_face_q,
		i12: &i12,
		pq: &p1q2,
		halfedges: halfedge_p,
	};
	for i in 0..i12.len() {
		count12.call(i);
	}
	drop(i12);
	drop(p1q2);
	let mut count21 = CountNewVerts::<true> {
		count_p: &mut sides_per_face_q,
		count_q: &mut sides_per_face_p,
		i12: &i21,
		pq: &p2q1,
		halfedges: halfedge_q,
	};
	for i in 0..i21.len() {
		count21.call(i);
	}

	sides_per_face_pq
}

pub fn size_face_pq2r(sides_per_face_pq: &[i32]) -> Vec<i32> {
	let mut face_pq2r: Vec<i32> = vec![0; sides_per_face_pq.len() + 1];
	vec_ext::inclusive_scan(
		sides_per_face_pq.iter().map(|&x| if x > 0 { 1 } else { 0 }),
		&mut face_pq2r[1..],
	);

	face_pq2r
}

pub fn size_face_normal(
	face_normal_p: &[Vector3<f64>],
	face_normal_q: &[Vector3<f64>],
	sides_per_face_pq: &[i32],
	num_face_r: usize,
	invert_q: bool,
) -> Vec<Vector3<f64>> {
	let mut face_normal = unsafe { vec_ext::uninit(num_face_r) };
	let mut tmp_buffer = unsafe { vec_ext::uninit(num_face_r) };

	let face_ids_p = (0..face_normal_p.len()).map(|i| {
		if sides_per_face_pq[i] > 0 {
			i
		} else {
			usize::MAX
		}
	});

	let next = vec_ext::copy_if(face_ids_p, &mut tmp_buffer, |v| v != usize::MAX);

	vec_ext::gather(&tmp_buffer[..next], face_normal_p, &mut face_normal);

	let face_ids_q = (0..face_normal_q.len()).map(|i| {
		if sides_per_face_pq[i + face_normal_p.len()] > 0 {
			i
		} else {
			usize::MAX
		}
	});

	let end = next + vec_ext::copy_if(face_ids_q, &mut tmp_buffer[next..], &|v| v != usize::MAX);

	if invert_q {
		vec_ext::gather_transformed(
			&tmp_buffer[next..end],
			face_normal_q,
			&mut face_normal[next..],
			|normal: Vector3<f64>| -normal,
		);
	} else {
		vec_ext::gather(
			&tmp_buffer[next..end],
			face_normal_q,
			&mut face_normal[next..],
		);
	}

	face_normal
}

pub fn size_face_edge(mut sides_per_face_pq: Vec<i32>) -> Vec<i32> {
	sides_per_face_pq.retain(|&v| v != 0);
	let mut face_edge = vec![0; sides_per_face_pq.len() + 1];
	vec_ext::inclusive_scan(sides_per_face_pq.into_iter(), &mut face_edge[1..]);

	face_edge
}

fn sort_edge_pos(edge_pos: &mut [EdgePos]) {
	edge_pos.sort_by_key(|i| (OrderedF64(i.edge_pos), i.collision_id));
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
	let middle = vec_ext::partition(edge_pos, |x| x.is_start);
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

pub fn append_partial_edges(
	vert_pos: &mut [Point3<f64>],
	halfedge_r: &mut [Halfedge],
	face_ptr_r: &mut [i32],
	face_rel_r: &mut [TriRelation],
	in_a: &MeshBool,
	edges_a: BTreeMap<i32, Vec<EdgePos>>,
	whole_halfedge_a: &mut [bool],
	i03: &[i32],
	v_p2r: &[i32],
	face_pq2r: &[i32],
	tri_rel_a: &[TriRelation],
	instance_id_old2new: &DeterministicMap<(bool, u32), u32>,
	write_tri2face: bool,
	forward: bool,
) {
	// Each edge in the map is partially retained; for each of these, look up
	// their original verts and include them based on their winding number (i03),
	// while remapping them to the output using vP2R. Use the verts position
	// projected along the edge vector to pair them up, then distribute these
	// edges to their faces.
	let vert_pos_a = &in_a.vert_pos;
	let halfedge_a = &in_a.tri.halfedge;

	// Per-iter cancel check; the caller's post-call IsCancelled discards the
	// partial outR.
	for (edge_a, mut edge_pos_a) in edges_a {
		sort_edge_pos(&mut edge_pos_a);
		let pair_a = halfedge_a.pair(edge_a);
		whole_halfedge_a[edge_a as usize] = false;
		whole_halfedge_a[pair_a as usize] = false;

		let v_start = halfedge_a.start(edge_a) as usize;
		let v_end = halfedge_a.end(edge_a) as usize;
		let edge_vec = vert_pos_a[v_end] - vert_pos_a[v_start];
		// Fill in the edge positions of the old points.
		for edge in edge_pos_a.iter_mut() {
			edge.edge_pos = vert_pos[edge.vert as usize].coords.dot(&edge_vec);
		}

		let mut inclusion = i03[v_start];
		let mut edge_pos = EdgePos {
			edge_pos: vert_pos[v_p2r[v_start] as usize].coords.dot(&edge_vec),
			vert: v_p2r[v_start],
			collision_id: i32::MAX,
			is_start: inclusion > 0,
		};

		for _ in 0..inclusion.abs() {
			edge_pos_a.push(edge_pos);
			edge_pos.vert += 1;
		}

		inclusion = i03[v_end];
		edge_pos = EdgePos {
			edge_pos: vert_pos[v_p2r[v_end] as usize].coords.dot(&edge_vec),
			vert: v_p2r[v_end],
			collision_id: i32::MAX,
			is_start: inclusion < 0,
		};

		for _ in 0..inclusion.abs() {
			edge_pos_a.push(edge_pos);
			edge_pos.vert += 1;
		}

		// add halfedges to result
		let face_left_a = edge_a / 3;
		let face_left = face_pq2r[face_left_a as usize] as usize;
		let face_right_a = pair_a / 3;
		let face_right = face_pq2r[face_right_a as usize] as usize;
		// Negative inclusion means the halfedges are reversed, which means our
		// reference is now to the endVert instead of the startVert, which is one
		// position advanced CCW. This is only valid if this is a retained vert; it
		// will be ignored later if the vert is new.
		let mut forward_rel = tri_rel_a[face_left_a as usize];
		forward_rel.instance_id = *instance_id_old2new
			.get(&(forward, forward_rel.instance_id))
			.unwrap();
		let mut backward_rel = tri_rel_a[face_right_a as usize];
		backward_rel.instance_id = *instance_id_old2new
			.get(&(forward, backward_rel.instance_id))
			.unwrap();

		if write_tri2face {
			forward_rel.face_id = face_left_a;
			backward_rel.face_id = face_right_a;
		}

		pair_up(&mut edge_pos_a, |mut e| {
			let forward_edge = face_ptr_r[face_left];
			face_ptr_r[face_left] += 1;
			let backward_edge = face_ptr_r[face_right];
			face_ptr_r[face_right] += 1;

			e.paired_halfedge = backward_edge;
			halfedge_r[forward_edge as usize] = e;
			face_rel_r[forward_edge as usize] = forward_rel;

			mem::swap(&mut e.start_vert, &mut e.end_vert);
			e.paired_halfedge = forward_edge;
			halfedge_r[backward_edge as usize] = e;
			face_rel_r[backward_edge as usize] = backward_rel;
		});
	}
}

/// Pair up each edge's verts and distribute to faces based on indices in key.
pub fn append_new_edges(
	vert_pos_r: &mut [Point3<f64>],
	halfedge_r: &mut [Halfedge],
	face_ptr_r: &mut [i32],
	face_rel: &mut [TriRelation],
	edges_new: BTreeMap<(i32, i32), Vec<EdgePos>>,
	face_pq2r: &[i32],
	num_face_p: usize,
	tri_rel_p: &[TriRelation],
	tri_rel_q: &[TriRelation],
	instance_id_old2new: &DeterministicMap<(bool, u32), u32>,
	write_tri2face: bool,
) {
	// Per-iter cancel check; the caller's post-call IsCancelled discards the
	// partial outR.
	for ((face_p, face_q), mut edge_pos) in edges_new {
		let face_p = face_p as usize;
		let face_q = face_q as usize;

		sort_edge_pos(&mut edge_pos);

		let mut bbox = Box3D::default();
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
		let face_left = face_pq2r[face_p] as usize;
		let face_right = face_pq2r[num_face_p + face_q] as usize;
		let mut forward_ref = tri_rel_p[face_p];
		forward_ref.instance_id = *instance_id_old2new
			.get(&(true, forward_ref.instance_id))
			.unwrap();
		let mut backward_ref = tri_rel_q[face_q];
		backward_ref.instance_id = *instance_id_old2new
			.get(&(false, backward_ref.instance_id))
			.unwrap();

		if write_tri2face {
			forward_ref.face_id = face_p as i32;
			backward_ref.face_id = face_q as i32;
		}

		pair_up(&mut edge_pos, |mut e| {
			let forward_edge = face_ptr_r[face_left];
			face_ptr_r[face_left] += 1;
			let backward_edge = face_ptr_r[face_right];
			face_ptr_r[face_right] += 1;

			e.paired_halfedge = backward_edge;
			halfedge_r[forward_edge as usize] = e;
			face_rel[forward_edge as usize] = forward_ref;

			mem::swap(&mut e.start_vert, &mut e.end_vert);
			e.paired_halfedge = forward_edge;
			halfedge_r[backward_edge as usize] = e;
			face_rel[backward_edge as usize] = backward_ref;
		});
	}
}

pub fn append_whole_edges(
	face_ptr_r: &mut [i32],
	halfedges_r: &mut [Halfedge],
	halfedge_rel: &mut [TriRelation],
	halfedge_a: &Halfedges,
	whole_halfedge_a: Vec<bool>,
	i03: Vec<i32>,
	v_p2r: Vec<i32>,
	face_pq2r: &[i32],
	tri_rel_a: &[TriRelation],
	instance_id_old2new: &DeterministicMap<(bool, u32), u32>,
	write_tri2face: bool,
	forward: bool,
) {
	//(struct DuplicateHalfedges is inlined here)
	for idx in 0..halfedge_a.len() as i32 {
		if !whole_halfedge_a[idx as usize] {
			continue;
		}

		let mut start_vert = halfedge_a.start(idx);
		let mut end_vert = halfedge_a.start(next_halfedge(idx));
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
		let pair = halfedge_a.pair(idx);
		let face_left_a = idx / 3;
		let new_face = face_pq2r[face_left_a as usize] as usize;
		let face_right_a = pair / 3;
		let face_right = face_pq2r[face_right_a as usize] as usize;
		// Negative inclusion means the halfedges are reversed, which means our
		// reference is now to the endVert instead of the startVert, which is one
		// position advanced CCW.
		let mut forward_rel = tri_rel_a[face_left_a as usize];
		forward_rel.instance_id = *instance_id_old2new
			.get(&(forward, forward_rel.instance_id))
			.unwrap();
		let mut backward_rel = tri_rel_a[face_right_a as usize];
		backward_rel.instance_id = *instance_id_old2new
			.get(&(forward, backward_rel.instance_id))
			.unwrap();

		if write_tri2face {
			forward_rel.face_id = face_left_a;
			backward_rel.face_id = face_right_a;
		}

		for _ in 0..inclusion.abs() {
			let forward_edge = atomic_add(&mut face_ptr_r[new_face], 1);
			let backward_edge = atomic_add(&mut face_ptr_r[face_right], 1);

			halfedges_r[forward_edge as usize] = Halfedge {
				start_vert,
				end_vert,
				paired_halfedge: backward_edge,
				prop_vert: 0,
			};
			halfedges_r[backward_edge as usize] = Halfedge {
				start_vert: end_vert,
				end_vert: start_vert,
				paired_halfedge: forward_edge,
				prop_vert: 0,
			};

			halfedge_rel[forward_edge as usize] = forward_rel;
			halfedge_rel[backward_edge as usize] = backward_rel;

			start_vert += 1;
			end_vert += 1;
		}
	}
}

struct Barycentric<'a> {
	uvw: &'a mut [Vector3<f64>],
	halfedge_r: &'a Halfedges,
	tri_rel_r: &'a [TriRelation],
	instance_id_offset_q: u32,
	vert_pos_p: &'a [Point3<f64>],
	vert_pos_q: &'a [Point3<f64>],
	vert_pos_r: &'a [Point3<f64>],
	halfedge_p: &'a Halfedges,
	halfedge_q: &'a Halfedges,
	epsilon: f64,
}

impl<'a> Barycentric<'a> {
	fn call(&mut self, tri: i32) {
		let ref_pq = self.tri_rel_r[tri as usize];
		if self.halfedge_r.start(3 * tri) < 0 {
			return;
		}

		let tri_pq = ref_pq.face_id;
		let pq = ref_pq.instance_id < self.instance_id_offset_q;
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
				get_barycentric(self.vert_pos_r[vert as usize], tri_pos, self.epsilon);
		}
	}
}

pub fn create_properties(
	halfedge_r: &mut Halfedges,
	tri_rel_r: &mut [TriRelation],
	vert_pos_r: &[Point3<f64>],
	instance_id_offset_q: u32,
	in_p: &MeshBool,
	in_q: &MeshBool,
	invert_q: bool,
	epsilon: f64,
) -> Vec<f64> {
	let prop_stride_p = in_p.prop_stride();
	let prop_stride_q = in_q.prop_stride();
	let prop_stride = prop_stride_p.max(prop_stride_q);
	if prop_stride == 0 {
		return vec![];
	}

	let num_tri = halfedge_r.num_tri();
	let mut bary = unsafe { vec_ext::uninit(halfedge_r.len()) };
	let mut bary_closure = Barycentric {
		uvw: &mut bary,
		halfedge_r,
		tri_rel_r,
		instance_id_offset_q,
		vert_pos_p: &in_p.vert_pos,
		vert_pos_q: &in_q.vert_pos,
		vert_pos_r,
		halfedge_p: &in_p.tri.halfedge,
		halfedge_q: &in_q.tri.halfedge,
		epsilon,
	};
	for tri in 0..num_tri {
		bary_closure.call(tri as i32);
	}

	let id_miss_prop = vert_pos_r.len() as i32;
	let mut prop_idx: Vec<Vec<(Vector3<i32>, i32)>> = vec![Vec::new(); vert_pos_r.len() + 1];
	let mut prop_miss_idx = [
		vec![-1; in_q.num_prop_vert()],
		vec![-1; in_p.num_prop_vert()],
	];

	let mut properties = Vec::with_capacity(vert_pos_r.len() * prop_stride);
	let mut idx = 0;

	for tri in 0..num_tri as i32 {
		// Skip collapsed triangles
		if halfedge_r.start(3 * tri) < 0 {
			continue;
		}

		let tri_rel = &mut tri_rel_r[tri as usize];
		//append_x_edges wrote a triangle id instead of a face id (write_tri2face
		//was true), purely for create_properties to consume and overwrite
		let tri_id = tri_rel.face_id;
		let pq = tri_rel.instance_id < instance_id_offset_q;
		tri_rel.face_id = if pq {
			in_p.tri.relation[tri_id as usize].face_id
		} else {
			in_q.tri.relation[tri_id as usize].face_id
		};
		let old_prop_stride = (if pq { prop_stride_p } else { prop_stride_q }) as i32;
		let properties_pq = if pq {
			&in_p.properties.data
		} else {
			&in_q.properties.data
		};
		let halfedge_pq = if pq {
			&in_p.tri.halfedge
		} else {
			&in_q.tri.halfedge
		};

		// For Subtract, Q's triangles are flipped in the result, so Q's
		// world-frame vertex normals (slot 0..2 when hasNormals) need a sign
		// flip to point outward from the result's solid (into the cavity).
		// Check is per-source-triangle, not whole input - inQ may be a mixed
		// Boolean result.
		let negate_normals = !pq
			&& invert_q
			&& old_prop_stride >= 3
			&& tri_has_normals(&in_q.instance_relation, in_q.tri.relation[tri_id as usize]);

		for i in 0..3 {
			let vert = halfedge_r.start(3 * tri + i);
			let uvw = &bary[(3 * tri + i) as usize];

			let mut key = Vector4::new(pq as i32, id_miss_prop, -1, -1);
			if old_prop_stride > 0 {
				let mut edge = -2;
				for j in 0..3 {
					if uvw[j as usize] == 1.0 {
						// On a retained vert, the propVert must also match
						key[2] = halfedge_pq.prop(3 * tri_id + j);
						edge = -1;
						break;
					}

					if uvw[j as usize] == 0.0 {
						edge = j
					};
				}

				if edge >= 0 {
					// On an edge, both propVerts must match
					let p0 = halfedge_pq.prop(3 * tri_id + next3_i32(edge));
					let p1 = halfedge_pq.prop(3 * tri_id + prev3_i32(edge));
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
					halfedge_r.set_prop(3 * tri + i, *entry);
					continue;
				}

				*entry = idx;
			} else {
				let bin = &mut prop_idx[key.y as usize];
				let mut b_found = false;
				for b in bin.iter() {
					if b.0 == Vector3::new(key.x, key.z, key.w) {
						b_found = true;
						halfedge_r.set_prop(3 * tri + i, b.1);
						break;
					}
				}

				if b_found {
					continue;
				}
				bin.push((Vector3::new(key.x, key.z, key.w), idx));
			}

			halfedge_r.set_prop(3 * tri + i, idx);
			idx += 1;
			for p in 0..prop_stride {
				let p = p as i32;

				if p < old_prop_stride {
					let mut old_props = Vector3::default();
					for j in 0..3 {
						old_props[j as usize] = properties_pq
							[(old_prop_stride * halfedge_pq.prop(3 * tri_id + j) + p) as usize];
					}

					let mut val = uvw.dot(&old_props);
					if negate_normals && p < 3 {
						val = -val;
					}
					properties.push(val);
				} else {
					properties.push(0.0);
				}
			}
		}
	}

	properties
}
