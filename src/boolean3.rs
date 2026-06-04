use crate::collider::{Recorder, SimpleRecorder};
use crate::common::{AABB, AABBOverlap, OpType};
use crate::disjoint_sets::DisjointSets;
use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{Halfedges, interpolate, shadows, with_sign};
use crate::utils::{next3_i32, permute};
use core::f64;
use nalgebra::{Point3, Vector2, Vector4};
use std::collections::HashSet;
use std::mem;
use std::ops::DerefMut;

/**
 * The notation in these files is abbreviated due to the complexity of the
 * functions involved. The key is that the input manifolds are P and Q, while
 * the output is R, and these letters in both upper and lower case refer to
 * these objects. Operations are based on dimensionality: vert: 0, edge: 1,
 * face: 2, solid: 3. X denotes a winding-number type quantity from the source
 * paper of this algorithm, while S is closely related but includes only the
 * subset of X values which "shadow" (are on the correct side of).
 *
 * Nearly everything here are sparse arrays, where for instance each pair in
 * p2q1 refers to a face index of P interacting with a halfedge index of Q.
 * Adjacent arrays like x21 refer to the values of X corresponding to each
 * sparse index pair.
 *
 * Note many functions are designed to work symmetrically, for instance for both
 * p2q1 and p1q2. Inside of these functions P and Q are marked as though the
 * function is forwards, but it may include a Boolean "reverse" that indicates P
 * and Q have been swapped.
 */

fn intersect(
	a_l: &Point3<f64>,
	a_r: &Point3<f64>,
	b_l: &Point3<f64>,
	b_r: &Point3<f64>,
) -> Vector4<f64> {
	let dyl = b_l.y - a_l.y;
	let dyr = b_r.y - a_r.y;
	debug_assert!(dyl * dyr <= 0.0, "Boolean manifold error: no intersection");
	let use_l = dyl.abs() < dyr.abs();
	let dx = a_r.x - a_l.x;
	let mut lambda = (if use_l { dyl } else { dyr }) / (dyl - dyr);
	if !lambda.is_finite() {
		lambda = 0.0;
	}
	let mut xyzz = Vector4::default();
	xyzz.x = lambda * dx + (if use_l { a_l.x } else { a_r.x });
	let a_dy = a_r.y - a_l.y;
	let b_dy = b_r.y - b_l.y;
	let use_a = a_dy.abs() < b_dy.abs();
	xyzz.y = lambda * (if use_a { a_dy } else { b_dy })
		+ (if use_l {
			if use_a { a_l.y } else { b_l.y }
		} else {
			if use_a { a_r.y } else { b_r.y }
		});
	xyzz.z = lambda * (a_r.z - a_l.z) + (if use_l { a_l.z } else { a_r.z });
	xyzz.w = lambda * (b_r.z - b_l.z) + (if use_l { b_l.z } else { b_r.z });
	return xyzz;
}

#[derive(Default, Copy, Clone)]
struct FaceEdge {
	edge: i32,
	start: i32,
	end: i32,
	is_forward: bool,
}

#[inline]
fn load_face_edges(halfedges: &Halfedges, tri: i32) -> [FaceEdge; 3] {
	let mut edge = [FaceEdge::default(); 3];
	for i in 0..3 {
		let halfedge = 3 * tri + i;
		let start = halfedges.start(halfedge);
		let end = halfedges.start(3 * tri + next3_i32(i));
		if start < end {
			edge[i as usize] = FaceEdge {
				edge: halfedge,
				start,
				end,
				is_forward: true,
			};
		} else {
			edge[i as usize] = FaceEdge {
				edge: halfedges.pair(halfedge),
				start: end,
				end: start,
				is_forward: false,
			};
		}
	}

	edge
}

#[inline]
fn shadow01<const EXPAND_P: bool, const FORWARD: bool>(
	a0: i32,
	b1: i32,
	b1s: i32,
	b1e: i32,
	in_a: &MeshBoolImpl,
	in_b: &MeshBoolImpl,
) -> (i32, Vector2<f64>) {
	let a0x = in_a.vert_pos[a0 as usize].x;
	let b1sx = in_b.vert_pos[b1s as usize].x;
	let b1ex = in_b.vert_pos[b1e as usize].x;
	let a0xp = in_a.vert_normal[a0 as usize].x;
	let b1sxp = in_b.vert_normal[b1s as usize].x;
	let b1exp = in_b.vert_normal[b1e as usize].x;
	let mut s01 = if FORWARD {
		shadows(a0x, b1ex, with_sign(EXPAND_P, a0xp) - b1exp) as i32
			- shadows(a0x, b1sx, with_sign(EXPAND_P, a0xp) - b1sxp) as i32
	} else {
		shadows(b1sx, a0x, with_sign(EXPAND_P, b1sxp) - a0xp) as i32
			- shadows(b1ex, a0x, with_sign(EXPAND_P, b1exp) - a0xp) as i32
	};

	let mut yz01 = Vector2::from_element(f64::NAN);

	if s01 != 0 {
		yz01 = interpolate(
			in_b.vert_pos[b1s as usize],
			in_b.vert_pos[b1e as usize],
			in_a.vert_pos[a0 as usize].x,
		);
		let b1pair = in_b.halfedge.pair(b1);
		let dir = in_b.face_normal[(b1 / 3) as usize].y + in_b.face_normal[(b1pair / 3) as usize].y;
		if FORWARD {
			if !shadows(in_a.vert_pos[a0 as usize].y, yz01[0], -dir) {
				s01 = 0;
			}
		} else {
			if !shadows(
				yz01[0],
				in_a.vert_pos[a0 as usize].y,
				with_sign(EXPAND_P, dir),
			) {
				s01 = 0;
			}
		}
	}

	(s01, yz01)
}

struct Kernel11<'a, const EXPAND_P: bool> {
	in_p: &'a MeshBoolImpl,
	in_q: &'a MeshBoolImpl,
}

impl<'a, const EXPAND_P: bool> Kernel11<'a, EXPAND_P> {
	fn call(
		&self,
		p1: i32,
		p1s: i32,
		p1e: i32,
		q1: i32,
		q1s: i32,
		q1e: i32,
	) -> (i32, Vector4<f64>) {
		let xyzz11;
		let mut s11 = 0;

		// For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut p_rl = [Point3::<f64>::default(); 2];
		let mut q_rl = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;

		let p0 = [p1s, p1e];
		for i in 0..p0.len() {
			let (s01, yz01) = shadow01::<EXPAND_P, true>(p0[i], q1, q1s, q1e, self.in_p, self.in_q);
			// If the value is NaN, then these do not overlap.
			if yz01[0].is_finite() {
				s11 += s01 * (if i == 0 { -1 } else { 1 });
				if k < 2 && (k == 0 || (s01 != 0) != shadows_var) {
					shadows_var = s01 != 0;
					p_rl[k] = self.in_p.vert_pos[p0[i] as usize];
					q_rl[k] = Point3::new(p_rl[k].x, yz01.x, yz01.y);
					k += 1;
				}
			}
		}

		let q0 = [q1s, q1e];
		for i in 0..q0.len() {
			let (s10, yz10) =
				shadow01::<EXPAND_P, false>(q0[i], p1, p1s, p1e, self.in_q, self.in_p);
			// If the value is NaN, then these do not overlap.
			if yz10[0].is_finite() {
				s11 += s10 * (if i == 0 { -1 } else { 1 });
				if k < 2 && (k == 0 || (s10 != 0) != shadows_var) {
					shadows_var = s10 != 0;
					q_rl[k] = self.in_q.vert_pos[q0[i] as usize];
					p_rl[k] = Point3::new(q_rl[k].x, yz10.x, yz10.y);
					k += 1;
				}
			}
		}

		if s11 == 0
		//no intersection
		{
			xyzz11 = Vector4::from_element(f64::NAN);
		} else {
			debug_assert!(k == 2, "Boolean manifold error: s11");
			xyzz11 = intersect(&p_rl[0], &p_rl[1], &q_rl[0], &q_rl[1]);

			let p1pair = self.in_p.halfedge.pair(p1);
			let dir_p = self.in_p.face_normal[(p1 / 3) as usize].z
				+ self.in_p.face_normal[(p1pair / 3) as usize].z;
			let q1pair = self.in_q.halfedge.pair(q1);
			let dir_q = self.in_q.face_normal[(q1 / 3) as usize].z
				+ self.in_q.face_normal[(q1pair / 3) as usize].z;
			if !shadows(xyzz11.z, xyzz11.w, with_sign(EXPAND_P, dir_p) - dir_q) {
				s11 = 0;
			}
		}

		(s11, xyzz11)
	}
}

struct Kernel02<'a, const EXPAND_P: bool, const FORWARD: bool> {
	in_a: &'a MeshBoolImpl,
	in_b: &'a MeshBoolImpl,
}

impl<'a, const EXPAND_P: bool, const FORWARD: bool> Kernel02<'a, EXPAND_P, FORWARD> {
	fn call(&self, a0: i32, b2: i32) -> (i32, f64) {
		let edge_b = load_face_edges(&self.in_b.halfedge, b2);
		self.call_with_edge(a0, b2, &edge_b)
	}

	fn call_with_edge(&self, a0: i32, b2: i32, edge_b: &[FaceEdge; 3]) -> (i32, f64) {
		let mut s02 = 0;
		let z02;

		// For yzzLR[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut yzz_rl = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;

		for i in 0..3 {
			let syz01 = shadow01::<EXPAND_P, FORWARD>(
				a0,
				edge_b[i].edge,
				edge_b[i].start,
				edge_b[i].end,
				self.in_a,
				self.in_b,
			);
			let s01 = syz01.0;
			let yz01 = syz01.1;
			// If the value is NaN, then these do not overlap.
			if yz01[0].is_finite() {
				s02 += s01
					* (if FORWARD == edge_b[i].is_forward {
						-1
					} else {
						1
					});
				if k < 2 && (k == 0 || (s01 != 0) != shadows_var) {
					shadows_var = s01 != 0;
					yzz_rl[k] = Point3::new(yz01[0], yz01[1], yz01[1]);
					k += 1;
				}
			}
		}

		if s02 == 0
		//no intersection
		{
			z02 = f64::NAN;
		} else {
			debug_assert!(k == 2, "Boolean manifold error: s02");
			let vert_pos_a = self.in_a.vert_pos[a0 as usize];
			z02 = interpolate(yzz_rl[0], yzz_rl[1], vert_pos_a.y)[1];
			if FORWARD {
				if !shadows(vert_pos_a.z, z02, -self.in_b.face_normal[b2 as usize].z) {
					s02 = 0;
				}
			} else {
				if !shadows(
					z02,
					vert_pos_a.z,
					with_sign(EXPAND_P, self.in_b.face_normal[b2 as usize].z),
				) {
					s02 = 0;
				}
			}
		}

		(s02, z02)
	}
}

struct Kernel12<'a, const EXPAND_P: bool, const FORWARD: bool> {
	in_a: &'a MeshBoolImpl,
	in_b: &'a MeshBoolImpl,
	k02: Kernel02<'a, EXPAND_P, FORWARD>,
	k11: Kernel11<'a, EXPAND_P>,
}

impl<'a, const EXPAND_P: bool, const FORWARD: bool> Kernel12<'a, EXPAND_P, FORWARD> {
	fn call(&self, a1: i32, b2: i32) -> (i32, Point3<f64>) {
		let mut x12 = 0;
		let mut v12 = Point3::new(f64::NAN, f64::NAN, f64::NAN);

		// For xzy_lr-[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut xzy_lr0 = [Point3::<f64>::default(); 2];
		let mut xzy_lr1 = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;

		let edge_a_start = self.in_a.halfedge.start(a1);
		let edge_a_end = self.in_a.halfedge.end(a1);
		let edge_b = load_face_edges(&self.in_b.halfedge, b2);

		for vert_a in [edge_a_start, edge_a_end] {
			let (s, z) = self.k02.call_with_edge(vert_a, b2, &edge_b);
			if z.is_finite() {
				x12 += s
					* (if (vert_a == edge_a_start) == FORWARD {
						1
					} else {
						-1
					});
				if k < 2 && (k == 0 || (s != 0) != shadows_var) {
					shadows_var = s != 0;
					xzy_lr0[k] = self.in_a.vert_pos[vert_a as usize];
					let switcheroo = xzy_lr0[k].deref_mut();
					mem::swap(&mut switcheroo.y, &mut switcheroo.z);
					xzy_lr1[k] = xzy_lr0[k];
					xzy_lr1[k][1] = z;
					k += 1;
				}
			}
		}

		for i in 0..3 {
			let (s, xyzz) = if FORWARD {
				self.k11.call(
					a1,
					edge_a_start,
					edge_a_end,
					edge_b[i].edge,
					edge_b[i].start,
					edge_b[i].end,
				)
			} else {
				self.k11.call(
					edge_b[i].edge,
					edge_b[i].start,
					edge_b[i].end,
					a1,
					edge_a_start,
					edge_a_end,
				)
			};
			if xyzz[0].is_finite() {
				x12 -= s * (if edge_b[i].is_forward { 1 } else { -1 });
				if k < 2 && (k == 0 || (s != 0) != shadows_var) {
					shadows_var = s != 0;
					xzy_lr0[k][0] = xyzz.x;
					xzy_lr0[k][1] = xyzz.z;
					xzy_lr0[k][2] = xyzz.y;
					xzy_lr1[k] = xzy_lr0[k];
					xzy_lr1[k][1] = xyzz.w;
					if !FORWARD {
						mem::swap(&mut xzy_lr0[k][1], &mut xzy_lr1[k][1]);
					}
					k += 1;
				}
			}
		}

		if x12 == 0
		//no intersection
		{
			v12 = Point3::new(f64::NAN, f64::NAN, f64::NAN);
		} else {
			debug_assert!(k == 2, "Boolean manifold error: v12");
			let xzyy = intersect(&xzy_lr0[0], &xzy_lr0[1], &xzy_lr1[0], &xzy_lr1[1]);
			v12.x = xzyy[0];
			v12.y = xzyy[2];
			v12.z = xzyy[1];
		}

		(x12, v12)
	}
}

// In forward mode, stores the intersections of edges of P with faces of Q.
// In reverse mode, stores the intersections of faces of P with edges of Q.
// In reverse, p1q2 -> p2q1, x12 -> x21, v12 -> v21.
#[derive(Default)]
pub struct Intersections {
	pub p1q2: Vec<[i32; 2]>,
	pub x12: Vec<i32>,
	pub v12: Vec<Point3<f64>>,
}

pub struct Kernel12Recorder<'a, const EXPAND_P: bool, const FORWARD: bool> {
	k12: &'a Kernel12<'a, EXPAND_P, FORWARD>,
	local_store: Intersections,
}

impl<'a, const EXPAND_P: bool, const FORWARD: bool> Recorder
	for Kernel12Recorder<'a, EXPAND_P, FORWARD>
{
	fn record(&mut self, query_idx: i32, leaf_idx: i32) {
		let tmp = &mut self.local_store;
		let (x12, v12) = self.k12.call(query_idx, leaf_idx);
		if v12[0].is_finite() {
			if FORWARD {
				tmp.p1q2.push([query_idx, leaf_idx]);
			} else {
				tmp.p1q2.push([leaf_idx, query_idx]);
			}

			tmp.x12.push(x12);
			tmp.v12.push(v12);
		}
	}
}

fn intersect12_impl<const EXPAND_P: bool, const FORWARD: bool>(
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
) -> Intersections {
	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	// a: 1 (edge), b: 2 (face)
	let a = if FORWARD { in_p } else { in_q };
	let b = if FORWARD { in_q } else { in_p };

	let k02 = Kernel02::<EXPAND_P, FORWARD> { in_a: a, in_b: b };
	let k11 = Kernel11::<EXPAND_P> { in_p, in_q };

	let k12 = Kernel12::<EXPAND_P, FORWARD> {
		in_a: a,
		in_b: b,
		k02,
		k11,
	};
	let mut recorder = Kernel12Recorder::<EXPAND_P, FORWARD> {
		k12: &k12,
		local_store: Intersections::default(),
	};
	let f = |i| {
		let start = a.halfedge.start(i);
		let end = a.halfedge.end(i);
		if start < end {
			AABB::new(a.vert_pos[start as usize], a.vert_pos[end as usize])
		} else {
			AABB::default()
		}
	};

	b.collider
		.collisions_from_fn::<false, _>(&mut recorder, f, a.halfedge.len(), true);

	let mut result = recorder.local_store;
	let p1q2 = &mut result.p1q2;
	// sort p1q2 according to edges
	let mut i12: Vec<_> = (0..p1q2.len()).collect();

	let index = if FORWARD { 0 } else { 1 };
	i12.sort_by_key(|&i| (p1q2[i][index], p1q2[i][1 - index]));
	permute(p1q2, &i12);
	permute(&mut result.x12, &i12);
	permute(&mut result.v12, &i12);
	result
}

fn intersect12<const FORWARD: bool>(
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	expand_p: bool,
) -> Intersections {
	if expand_p {
		intersect12_impl::<true, FORWARD>(in_p, in_q)
	} else {
		intersect12_impl::<false, FORWARD>(in_p, in_q)
	}
}

fn winding03_impl<const EXPAND_P: bool, const FORWARD: bool>(
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	p1q2: &[[i32; 2]],
) -> Vec<i32> {
	// a: 0 (vert), b: 2 (face)
	let a = if FORWARD { in_p } else { in_q };
	let b = if FORWARD { in_q } else { in_p };
	let index = if FORWARD { 0 } else { 1 };

	// Invariant: every ctx-passing parallel op is followed by IsCancelled to
	// keep partial output from feeding unconditional downstream consumers.
	let u_a = DisjointSets::new(a.vert_pos.len());
	for edge in 0..a.halfedge.len() as i32 {
		let start = a.halfedge.start(edge);
		let end = a.halfedge.end(edge);
		if start >= end {
			continue;
		}
		// check if the edge is broken
		let it = p1q2.partition_point(|collision_pair| collision_pair[index] < edge);
		if it == p1q2.len() || p1q2[it][index] != edge {
			u_a.unite(start as usize, end as usize);
		}
	}

	// find components, the hope is the number of components should be small
	let mut components = HashSet::new();
	for v in 0..a.vert_pos.len() {
		components.insert(u_a.find(v));
	}

	let verts: Vec<_> = components.into_iter().map(|c| c as i32).collect();

	let mut w03 = vec![0; a.num_vert()];
	let k02 = Kernel02::<EXPAND_P, FORWARD> { in_a: a, in_b: b };
	let mut recorderf = |query_idx: i32, leaf_idx: i32| {
		let (s02, z02) = k02.call(verts[query_idx as usize], leaf_idx);
		if z02.is_finite() {
			// note that i is distinct on each thread, and verts contains unique
			// elements, so this does not require atomics
			w03[verts[query_idx as usize] as usize] += s02 * (if FORWARD { 1 } else { -1 });
		}
	};
	let mut recorder = SimpleRecorder::new(&mut recorderf);
	let f = |i| a.vert_pos[verts[i as usize] as usize];
	b.collider
		.collisions_from_fn::<false, _>(&mut recorder, f, verts.len(), true);
	// flood fill
	for i in 0..w03.len() {
		let root = u_a.find(i);
		if root == i {
			continue;
		}
		w03[i] = w03[root];
	}

	w03
}

fn winding03<const FORWARD: bool>(
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	p1q2: &[[i32; 2]],
	expand_p: bool,
) -> Vec<i32> {
	if expand_p {
		winding03_impl::<true, FORWARD>(in_p, in_q, p1q2)
	} else {
		winding03_impl::<false, FORWARD>(in_p, in_q, p1q2)
	}
}

pub struct Boolean3<'a> {
	pub in_p: &'a MeshBoolImpl,
	pub in_q: &'a MeshBoolImpl,
	pub expand_p: bool,
	pub xv12: Intersections,
	pub xv21: Intersections,
	pub w03: Vec<i32>,
	pub w30: Vec<i32>,
	pub valid: bool,
}

impl<'a> Boolean3<'a> {
	pub fn new(in_p: &'a MeshBoolImpl, in_q: &'a MeshBoolImpl, op: OpType) -> Self {
		let expand_p = op == OpType::Add;

		// Symbolic perturbation:
		// Union -> expand inP, expand inQ
		// Difference, Intersection -> contract inP, expand inQ
		// Technically Intersection should contract inQ, but doing it this way makes
		// Split faster and any suboptimal cases seem pretty rare.
		const INT_MAX_SZ: usize = i32::MAX as usize;

		if in_p.is_empty() || in_q.is_empty() || !in_p.bbox.does_overlap(&in_q.bbox) {
			//No overlap, early out
			return Boolean3 {
				in_p,
				in_q,
				expand_p,
				xv12: Intersections::default(),
				xv21: Intersections::default(),
				w03: vec![0; in_p.num_vert()],
				w30: vec![0; in_q.num_vert()],
				valid: true,
			};
		}

		// Level 3
		// Build up the intersection of the edges and triangles, keeping only those
		// that intersect, and record the direction the edge is passing through the
		// triangle.
		let xv12 = intersect12::<true>(in_p, in_q, expand_p);
		let xv21 = intersect12::<false>(in_p, in_q, expand_p);

		if xv12.x12.len() > INT_MAX_SZ || xv21.x12.len() > INT_MAX_SZ {
			return Boolean3 {
				in_p,
				in_q,
				expand_p,
				xv12,
				xv21,
				w03: Vec::default(),
				w30: Vec::default(),
				valid: false,
			};
		}

		// Compute winding numbers of all vertices using flood fill
		// Vertices on the same connected component have the same winding number
		let w03 = winding03::<true>(in_p, in_q, &xv12.p1q2, expand_p);
		let w30 = winding03::<false>(in_p, in_q, &xv21.p1q2, expand_p);

		Boolean3 {
			in_p,
			in_q,
			expand_p,
			xv12,
			xv21,
			w03,
			w30,
			valid: true,
		}
	}
}
