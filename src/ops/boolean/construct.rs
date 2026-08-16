use crate::Properties;
use crate::halfedge::{Halfedge, Halfedges, next_halfedge};
use crate::mesh_relations::{InstanceRelation, TriRelation, tri_has_normals};
use crate::spatial::aabb::Box3D;
use crate::util::math::{atomic_add, get_barycentric, next3_usize, prev3_usize};
use crate::util::num_convert::OrderedF64;
use nalgebra::{Matrix3, Point3, Vector3, Vector4};
use rustc_hash::FxHashMap;
use std::ops::Deref;
use std::rc::Rc;
use std::{array, mem};

pub struct DuplicateVerts<'a, T> {
	pub vert_pos_r: &'a mut [Point3<f64>],
	pub inclusion: &'a [T],
	pub vert_r: &'a [i32],
	pub vert_pos_a: &'a [Point3<f64>],
}

impl<'a, T> DuplicateVerts<'a, T>
where
	T: Copy + Into<i32>,
{
	pub fn call(&mut self, vert: usize) {
		let n: i32 = self.inclusion[vert].into().abs();
		for i in 0..n {
			let a = self.vert_r[vert];
			let b = self.vert_pos_a[vert];
			self.vert_pos_r[(a + i) as usize] = b;
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct EdgePos {
	edge_pos: f64,
	vert: i32,
	is_start: bool,
}

pub fn add_new_edge_verts(
	// we need concurrent_map because we will be adding things concurrently
	edges_old: &mut FxHashMap<i32, Vec<EdgePos>>,
	edges_new: &mut FxHashMap<(i32, i32), Vec<EdgePos>>,
	p1q2: &[[i32; 2]],
	i12: &[i8],
	v12_r: Vec<i32>,
	halfedge_p: &Halfedges,
	forward: bool,
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

		let mut key_right = (halfedge_p.pair[edge_p as usize] / 3, face_q);
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
					vert: vert + (j as i32),
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
			self.count[i] += (self.inclusion[self.halfedges.start[3 * i + j] as usize]).abs();
		}
	}
}

struct CountNewVerts<'a, const INVERTED: bool> {
	count_p: &'a mut [i32],
	count_q: &'a mut [i32],
	i12: &'a [i8],
	pq: &'a [[i32; 2]],
	halfedges: &'a Halfedges,
}

impl<'a, const INVERTED: bool> CountNewVerts<'a, INVERTED> {
	fn call(&mut self, idx: usize) {
		let edge_p = self.pq[idx][if INVERTED { 1 } else { 0 }] as usize;
		let face_q = self.pq[idx][if INVERTED { 0 } else { 1 }] as usize;
		let inclusion = (self.i12[idx] as i32).abs();

		self.count_q[face_q] += inclusion;
		self.count_p[edge_p / 3] += inclusion;
		self.count_p[(self.halfedges.pair[edge_p] / 3) as usize] += inclusion;
	}
}

pub fn size_sides_per_face_pq(
	halfedge_p: &Halfedges,
	halfedge_q: &Halfedges,
	i03: &[i32],
	i30: &[i32],
	i12: Vec<i8>,
	i21: Vec<i8>,
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
	for i in 0..halfedge_p.num_tri() {
		count_p.call(i);
	}
	let mut count_q = CountVerts {
		halfedges: halfedge_q,
		count: &mut sides_per_face_q,
		inclusion: i30,
	};
	for i in 0..halfedge_q.num_tri() {
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

pub fn size_face_normal(
	tri_normal_p: Rc<Vec<Vector3<f64>>>,
	tri_normal_q: Rc<Vec<Vector3<f64>>>,
	sides_per_face_pq: &[i32],
	num_face_r: usize,
	invert_q: bool,
) -> Vec<Vector3<f64>> {
	let tri_normal_p_iter = sides_per_face_pq[..tri_normal_p.len()]
		.iter()
		.cloned()
		.enumerate()
		.filter(|&(_, sides)| sides > 0)
		.map(|(i, _)| tri_normal_p[i]);
	let tri_normal_q_iter = sides_per_face_pq[tri_normal_p.len()..]
		.iter()
		.cloned()
		.enumerate()
		.filter(|&(_, sides)| sides > 0)
		.map(|(i, _)| tri_normal_q[i]);

	let mut face_normal = Vec::with_capacity(num_face_r);

	face_normal.extend(tri_normal_p_iter);
	if invert_q {
		face_normal.extend(tri_normal_q_iter.map(|normal| -normal));
	} else {
		face_normal.extend(tri_normal_q_iter);
	}

	face_normal
}

fn pair_up(mut edge_pos: Vec<EdgePos>, mut f: impl FnMut(Halfedge)) {
	// Pair start vertices with end vertices to form edges. The choice of pairing
	// is arbitrary for the manifoldness guarantee, but must be ordered to be
	// geometrically valid. If the order does not go start-end-start-end... then
	// the input and output are not geometrically valid and this algorithm becomes
	// a heuristic.
	debug_assert!(
		edge_pos.len() % 2 == 0,
		"Non-manifold edge! Not an even number of points."
	);

	if edge_pos.len() == 2 {
		debug_assert!(
			edge_pos[0].is_start != edge_pos[1].is_start,
			"Non-manifold edge!"
		);

		//append_new_edges will always run this branch, except when
		//surfaces/vertices of p are coincident/laying directly on
		//those of q, causing len of either 4 or 6
		//append_partial_edges depends on number of times this edge
		//(represented by edge_pos) is chopped to pieces by the other
		//mesh
		//this optimization avoids sorting overhead on the common case

		let (start_i, end_i) = if edge_pos[0].is_start { (0, 1) } else { (1, 0) };

		f(Halfedge {
			start_vert: edge_pos[start_i].vert,
			end_vert: edge_pos[end_i].vert,
			paired_halfedge: -1,
			prop_vert: 0,
		});
		return;
	}

	let n_edges = edge_pos.len() / 2;
	edge_pos.sort_unstable_by_key(|i| (!i.is_start, OrderedF64(i.edge_pos), i.vert));
	debug_assert!(
		edge_pos.partition_point(|i| i.is_start) == n_edges,
		"Non-manifold edge!"
	);
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
	vert_pos_a: &[Point3<f64>],
	halfedge_a: &Halfedges,
	edges_a: FxHashMap<i32, Vec<EdgePos>>,
	whole_halfedge_a: &mut [bool],
	i03: &[i32],
	v_p2r: &[i32],
	face_pq2r: &[i32],
	tri_rel_a: &[TriRelation],
	instance_id_offset: u32,
	write_tri2face: bool,
) {
	// Each edge in the map is partially retained; for each of these, look up
	// their original verts and include them based on their winding number (i03),
	// while remapping them to the output using vP2R. Use the verts position
	// projected along the edge vector to pair them up, then distribute these
	// edges to their faces.

	// Per-iter cancel check; the caller's post-call IsCancelled discards the
	// partial outR.
	for (edge_a, mut edge_pos_a) in edges_a {
		let edge_a = edge_a as usize;
		let pair_a = halfedge_a.pair[edge_a] as usize;
		whole_halfedge_a[edge_a] = false;
		whole_halfedge_a[pair_a] = false;

		let v_start = halfedge_a.start[edge_a] as usize;
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
			is_start: inclusion < 0,
		};

		for _ in 0..inclusion.abs() {
			edge_pos_a.push(edge_pos);
			edge_pos.vert += 1;
		}

		// add halfedges to result
		let face_left_a = edge_a / 3;
		let face_left = face_pq2r[face_left_a] as usize;
		let face_right_a = pair_a / 3;
		let face_right = face_pq2r[face_right_a] as usize;
		// Negative inclusion means the halfedges are reversed, which means our
		// reference is now to the endVert instead of the startVert, which is one
		// position advanced CCW. This is only valid if this is a retained vert; it
		// will be ignored later if the vert is new.
		let mut forward_rel = tri_rel_a[face_left_a];
		forward_rel.instance_id += instance_id_offset;
		let mut backward_rel = tri_rel_a[face_right_a];
		backward_rel.instance_id += instance_id_offset;

		if write_tri2face {
			forward_rel.face_id = face_left_a as i32;
			backward_rel.face_id = face_right_a as i32;
		}

		pair_up(edge_pos_a, |mut e| {
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
	edges_new: FxHashMap<(i32, i32), Vec<EdgePos>>,
	face_pq2r: &[i32],
	num_face_p: usize,
	tri_rel_p: &[TriRelation],
	tri_rel_q: &[TriRelation],
	instance_id_offset_q: u32,
	write_tri2face: bool,
) {
	// Per-iter cancel check; the caller's post-call IsCancelled discards the
	// partial outR.
	for ((face_p, face_q), mut edge_pos) in edges_new {
		let face_p = face_p as usize;
		let face_q = face_q as usize;

		let mut bbox = Box3D::empty();
		for edge in edge_pos.iter() {
			bbox.union_point_mut(vert_pos_r[edge.vert as usize]);
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
		let mut backward_ref = tri_rel_q[face_q];
		backward_ref.instance_id += instance_id_offset_q;

		if write_tri2face {
			forward_ref.face_id = face_p as i32;
			backward_ref.face_id = face_q as i32;
		}

		pair_up(edge_pos, |mut e| {
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
	instance_id_offset: u32,
	write_tri2face: bool,
) {
	//(struct DuplicateHalfedges is inlined here)
	for idx in 0..halfedge_a.len() {
		if !whole_halfedge_a[idx] {
			continue;
		}

		let mut start_vert = halfedge_a.start[idx];
		let mut end_vert = halfedge_a.start[next_halfedge(idx)];
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
		let pair = halfedge_a.pair[idx] as usize;
		let face_left_a = idx / 3;
		let new_face = face_pq2r[face_left_a] as usize;
		let face_right_a = pair / 3;
		let face_right = face_pq2r[face_right_a] as usize;
		// Negative inclusion means the halfedges are reversed, which means our
		// reference is now to the endVert instead of the startVert, which is one
		// position advanced CCW.
		let mut forward_rel = tri_rel_a[face_left_a];
		forward_rel.instance_id += instance_id_offset;
		let mut backward_rel = tri_rel_a[face_right_a];
		backward_rel.instance_id += instance_id_offset;

		if write_tri2face {
			forward_rel.face_id = face_left_a as i32;
			backward_rel.face_id = face_right_a as i32;
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

pub fn create_properties(
	halfedge_r: &mut Halfedges,
	tri_rel_r: &mut [TriRelation],
	vert_pos_r: &[Point3<f64>],
	instance_rel_r: &[InstanceRelation],
	instance_id_offset_q: u32,
	vert_pos_p: Rc<Vec<Point3<f64>>>,
	halfedge_p: Rc<Halfedges>,
	tri_rel_p: Rc<Vec<TriRelation>>,
	prop_p: Rc<Properties>,
	vert_pos_q: Rc<Vec<Point3<f64>>>,
	halfedge_q: Rc<Halfedges>,
	tri_rel_q: Rc<Vec<TriRelation>>,
	prop_q: Rc<Properties>,
	invert_q: bool,
	epsilon: f64,
) -> Vec<f64> {
	let prop_stride_r = prop_p.stride.max(prop_q.stride);
	if prop_stride_r == 0 {
		return vec![];
	}

	let num_vert_p = vert_pos_p.len();
	let num_vert_q = vert_pos_q.len();
	let num_tri_r = halfedge_r.num_tri();
	let bary = barycentric(
		halfedge_r,
		tri_rel_r,
		instance_id_offset_q,
		vert_pos_p,
		vert_pos_q,
		vert_pos_r,
		&halfedge_p,
		&halfedge_q,
		epsilon,
	);

	let id_miss_prop = vert_pos_r.len() as i32;
	let mut prop_idx: Vec<Vec<(Vector3<i32>, i32)>> = vec![Vec::new(); vert_pos_r.len() + 1];
	let prop_vert_p = if prop_p.stride == 0 {
		num_vert_p
	} else {
		prop_p.data.len() / prop_p.stride
	};
	let prop_vert_q = if prop_q.stride == 0 {
		num_vert_q
	} else {
		prop_q.data.len() / prop_q.stride
	};
	let mut prop_miss_idx = [vec![-1; prop_vert_q], vec![-1; prop_vert_p]];

	let mut properties = Vec::with_capacity(vert_pos_r.len() * prop_stride_r);
	let mut idx = 0;

	for tri in 0..num_tri_r {
		// Skip collapsed triangles
		if halfedge_r.start[3 * tri] < 0 {
			continue;
		}

		let tri_rel = &mut tri_rel_r[tri];
		//append_x_edges wrote a triangle id instead of a face id (write_tri2face
		//was true), purely for create_properties to consume and overwrite
		let tri_id = tri_rel.face_id as usize;
		let pq = tri_rel.instance_id < instance_id_offset_q;
		tri_rel.face_id = if pq {
			tri_rel_p[tri_id].face_id
		} else {
			tri_rel_q[tri_id].face_id
		};
		let old_prop_stride = (if pq { prop_p.stride } else { prop_q.stride }) as i32;
		let properties_pq = if pq { &prop_p.data } else { &prop_q.data };
		let halfedge_pq = if pq { &halfedge_p } else { &halfedge_q };

		// For Subtract, Q's triangles are flipped in the result, so Q's
		// world-frame vertex normals (slot 0..2 when hasNormals) need a sign
		// flip to point outward from the result's solid (into the cavity).
		// Check is per-source-triangle, not whole input - inQ may be a mixed
		// Boolean result.
		let negate_normals = !pq
			&& invert_q
			&& old_prop_stride >= 3
			&& tri_has_normals(
				&instance_rel_r[instance_id_offset_q as usize..],
				tri_rel_q[tri_id],
			);

		for i in 0..3 {
			let vert = halfedge_r.start[3 * tri + i];
			let uvw = &bary[3 * tri + i];

			let mut key = Vector4::new(pq as i32, id_miss_prop, -1, -1);
			if old_prop_stride > 0 {
				let mut edge = -2;
				for j in 0..3 {
					if uvw[j as usize] == 1.0 {
						// On a retained vert, the propVert must also match
						key[2] = halfedge_pq.prop[3 * tri_id + (j as usize)];
						edge = -1;
						break;
					}

					if uvw[j as usize] == 0.0 {
						edge = j
					};
				}

				if edge >= 0 {
					// On an edge, both propVerts must match
					let p0 = halfedge_pq.prop[3 * tri_id + next3_usize(edge as usize)];
					let p1 = halfedge_pq.prop[3 * tri_id + prev3_usize(edge as usize)];
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
					halfedge_r.prop[3 * tri + i] = *entry;
					continue;
				}

				*entry = idx;
			} else {
				let bin = &mut prop_idx[key.y as usize];
				let mut b_found = false;
				for b in bin.iter() {
					if b.0 == Vector3::new(key.x, key.z, key.w) {
						b_found = true;
						halfedge_r.prop[3 * tri + i] = b.1;
						break;
					}
				}

				if b_found {
					continue;
				}
				bin.push((Vector3::new(key.x, key.z, key.w), idx));
			}

			halfedge_r.prop[3 * tri + i] = idx;
			idx += 1;
			for p in 0..prop_stride_r {
				let p = p as i32;

				if p < old_prop_stride {
					let mut old_props = Vector3::default();
					for j in 0..3 {
						old_props[j] = properties_pq
							[(old_prop_stride * halfedge_pq.prop[3 * tri_id + j] + p) as usize];
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

fn barycentric(
	halfedge_r: &Halfedges,
	tri_rel_r: &[TriRelation],
	instance_id_offset_q: u32,
	vert_pos_p: Rc<Vec<Point3<f64>>>,
	vert_pos_q: Rc<Vec<Point3<f64>>>,
	vert_pos_r: &[Point3<f64>],
	halfedge_p: &Halfedges,
	halfedge_q: &Halfedges,
	epsilon: f64,
) -> Vec<Vector3<f64>> {
	(0..halfedge_r.num_tri())
		.flat_map(|tri| {
			let ref_pq = tri_rel_r[tri];
			if halfedge_r.start[3 * tri] < 0 {
				return [Vector3::default(); 3];
			}

			let tri_pq = ref_pq.face_id as usize;
			let pq = ref_pq.instance_id < instance_id_offset_q;
			let vert_pos = if pq { &vert_pos_p } else { &vert_pos_q };
			let halfedge = if pq { halfedge_p } else { halfedge_q };

			let mut tri_pos = Matrix3::default();
			for j in 0..3 {
				*tri_pos.column_mut(j) = *vert_pos[halfedge.start[3 * tri_pq + j] as usize].deref();
			}

			array::from_fn::<_, 3, _>(|i| {
				let vert = halfedge_r.start[3 * tri + i] as usize;
				get_barycentric(vert_pos_r[vert], tri_pos, epsilon)
			})
		})
		.collect()
}
