use crate::halfedge::Halfedges;
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::disjoint_sets::DisjointSets;
use crate::util::math::next3_usize;
use core::f64;
use nalgebra::{Point3, Vector2, Vector3, Vector4};
use rustc_hash::FxHashSet;
use std::mem;
use std::ops::DerefMut;
use std::rc::Rc;

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

/// In forward mode, stores the intersections of edges of P with faces of Q.
/// In reverse mode, stores the intersections of faces of P with edges of Q.
/// In reverse, p1q2 -> p2q1, x12 -> x21, v12 -> v21.
#[derive(Default, Debug)]
pub struct Intersections {
	pub p1q2: Vec<[i32; 2]>,
	pub x12: Vec<i8>,
	pub v12: Vec<Point3<f64>>,
}

pub fn intersect12<const FORWARD: bool>(
	vert_pos_p: &[Point3<f64>],
	halfedge_p: &Halfedges,
	tri_normal_p: &[Vector3<f64>],
	collider_p: &BVHCollider,
	vert_normal_p: &[Vector3<f64>],
	vert_pos_q: &[Point3<f64>],
	halfedge_q: &Halfedges,
	tri_normal_q: &[Vector3<f64>],
	collider_q: &BVHCollider,
	vert_normal_q: &[Vector3<f64>],
	expand_p: bool,
) -> Intersections {
	if expand_p {
		intersect12_impl::<true, FORWARD>(
			vert_pos_p,
			halfedge_p,
			tri_normal_p,
			collider_p,
			vert_normal_p,
			vert_pos_q,
			halfedge_q,
			tri_normal_q,
			collider_q,
			vert_normal_q,
		)
	} else {
		intersect12_impl::<false, FORWARD>(
			vert_pos_p,
			halfedge_p,
			tri_normal_p,
			collider_p,
			vert_normal_p,
			vert_pos_q,
			halfedge_q,
			tri_normal_q,
			collider_q,
			vert_normal_q,
		)
	}
}

fn intersect12_impl<const EXPAND_P: bool, const FORWARD: bool>(
	vert_pos_p: &[Point3<f64>],
	halfedge_p: &Halfedges,
	tri_normal_p: &[Vector3<f64>],
	collider_p: &BVHCollider,
	vert_normal_p: &[Vector3<f64>],
	vert_pos_q: &[Point3<f64>],
	halfedge_q: &Halfedges,
	tri_normal_q: &[Vector3<f64>],
	collider_q: &BVHCollider,
	vert_normal_q: &[Vector3<f64>],
) -> Intersections {
	// a: 1 (edge), b: 2 (face)
	let (vert_pos_a, halfedge_a, vert_normal_a) = if FORWARD {
		(vert_pos_p, halfedge_p, vert_normal_p)
	} else {
		(vert_pos_q, halfedge_q, vert_normal_q)
	};
	let (vert_pos_b, halfedge_b, tri_normal_b, collider_b, vert_normal_b) = if FORWARD {
		(
			vert_pos_q,
			halfedge_q,
			tri_normal_q,
			collider_q,
			vert_normal_q,
		)
	} else {
		(
			vert_pos_p,
			halfedge_p,
			tri_normal_p,
			collider_p,
			vert_normal_p,
		)
	};

	let k02 = Kernel02::<EXPAND_P, FORWARD> {
		vert_pos_a,
		vert_normal_a,
		vert_pos_b,
		halfedge_b,
		tri_normal_b,
		vert_normal_b,
	};
	let k11 = Kernel11::<EXPAND_P> {
		vert_pos_p,
		halfedge_pair_p: &halfedge_p.pair,
		tri_normal_p,
		vert_normal_p,
		vert_pos_q,
		halfedge_pair_q: &halfedge_q.pair,
		tri_normal_q,
		vert_normal_q,
	};

	let k12 = Kernel12::<EXPAND_P, FORWARD> {
		vert_pos_a,
		halfedge_a,
		halfedge_b,
		k02,
		k11,
	};
	let f = |i| {
		let start = halfedge_a.start[i] as usize;
		let end = halfedge_a.end(i) as usize;
		if start < end {
			Box3D::new(vert_pos_a[start], vert_pos_a[end])
		} else {
			Box3D::empty()
		}
	};

	let mut xv12 = Intersections::default();
	collider_b.collisions_from_fn::<false, _>(
		|query_idx, leaf_idx| {
			let (x12, v12) = k12.call(query_idx, leaf_idx);
			if v12[0].is_finite() {
				if FORWARD {
					xv12.p1q2.push([query_idx as i32, leaf_idx as i32]);
				} else {
					xv12.p1q2.push([leaf_idx as i32, query_idx as i32]);
				}

				xv12.x12.push(x12);
				xv12.v12.push(v12);
			}
		},
		f,
		halfedge_a.len(),
		true,
	);

	xv12
}

struct Kernel12<'a, const EXPAND_P: bool, const FORWARD: bool> {
	vert_pos_a: &'a [Point3<f64>],
	halfedge_a: &'a Halfedges,
	halfedge_b: &'a Halfedges,
	k02: Kernel02<'a, EXPAND_P, FORWARD>,
	k11: Kernel11<'a, EXPAND_P>,
}

impl<'a, const EXPAND_P: bool, const FORWARD: bool> Kernel12<'a, EXPAND_P, FORWARD> {
	fn call(&self, a1: usize, b2: usize) -> (i8, Point3<f64>) {
		let mut x12: i8 = 0;
		let mut v12 = Point3::new(f64::NAN, f64::NAN, f64::NAN);

		// For xzy_lr-[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut xzy_lr0 = [Point3::<f64>::default(); 2];
		let mut xzy_lr1 = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;

		let edge_a_start = self.halfedge_a.start[a1] as usize;
		let edge_a_end = self.halfedge_a.end(a1) as usize;
		let edge_b = load_face_edges(self.halfedge_b, b2);

		for vert_a in [edge_a_start, edge_a_end] {
			let (s, z) = self.k02.call_with_edge(vert_a, b2, &edge_b);
			if z.is_finite() {
				x12 += (s
					* (if (vert_a == edge_a_start) == FORWARD {
						1
					} else {
						-1
					})) as i8;
				if k < 2 && (k == 0 || (s != 0) != shadows_var) {
					shadows_var = s != 0;
					xzy_lr0[k] = self.vert_pos_a[vert_a];
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

struct Kernel11<'a, const EXPAND_P: bool> {
	vert_pos_p: &'a [Point3<f64>],
	halfedge_pair_p: &'a [i32],
	tri_normal_p: &'a [Vector3<f64>],
	vert_normal_p: &'a [Vector3<f64>],
	vert_pos_q: &'a [Point3<f64>],
	halfedge_pair_q: &'a [i32],
	tri_normal_q: &'a [Vector3<f64>],
	vert_normal_q: &'a [Vector3<f64>],
}

impl<'a, const EXPAND_P: bool> Kernel11<'a, EXPAND_P> {
	fn call(
		&self,
		p1: usize,
		p1s: usize,
		p1e: usize,
		q1: usize,
		q1s: usize,
		q1e: usize,
	) -> (i8, Vector4<f64>) {
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
			let (s01, yz01) = shadow01::<EXPAND_P, true>(
				p0[i],
				q1,
				q1s,
				q1e,
				self.vert_pos_p,
				self.vert_normal_p,
				self.vert_pos_q,
				self.halfedge_pair_q,
				self.tri_normal_q,
				self.vert_normal_q,
			);
			// If the value is NaN, then these do not overlap.
			if yz01[0].is_finite() {
				s11 += s01 * (if i == 0 { -1 } else { 1 });
				if k < 2 && (k == 0 || (s01 != 0) != shadows_var) {
					shadows_var = s01 != 0;
					p_rl[k] = self.vert_pos_p[p0[i]];
					q_rl[k] = Point3::new(p_rl[k].x, yz01.x, yz01.y);
					k += 1;
				}
			}
		}

		let q0 = [q1s, q1e];
		for i in 0..q0.len() {
			let (s10, yz10) = shadow01::<EXPAND_P, false>(
				q0[i],
				p1,
				p1s,
				p1e,
				self.vert_pos_q,
				self.vert_normal_q,
				self.vert_pos_p,
				self.halfedge_pair_p,
				self.tri_normal_p,
				self.vert_normal_p,
			);
			// If the value is NaN, then these do not overlap.
			if yz10[0].is_finite() {
				s11 += s10 * (if i == 0 { -1 } else { 1 });
				if k < 2 && (k == 0 || (s10 != 0) != shadows_var) {
					shadows_var = s10 != 0;
					q_rl[k] = self.vert_pos_q[q0[i]];
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

			let p1pair = self.halfedge_pair_p[p1] as usize;
			let dir_p = self.tri_normal_p[p1 / 3].z + self.tri_normal_p[p1pair / 3].z;
			let q1pair = self.halfedge_pair_q[q1] as usize;
			let dir_q = self.tri_normal_q[q1 / 3].z + self.tri_normal_q[q1pair / 3].z;
			if !shadows(xyzz11.z, xyzz11.w, with_sign(EXPAND_P, dir_p) - dir_q) {
				s11 = 0;
			}
		}

		(s11, xyzz11)
	}
}

pub fn winding03<const FORWARD: bool>(
	vert_pos_a: &[Point3<f64>],
	halfedge_a: &Halfedges,
	vert_normal_a: &[Vector3<f64>],
	vert_pos_b: &[Point3<f64>],
	halfedge_b: &Halfedges,
	tri_normal_b: &[Vector3<f64>],
	collider_b: Rc<BVHCollider>,
	vert_normal_b: &[Vector3<f64>],
	p1q2: &[[i32; 2]],
	expand_p: bool,
) -> Vec<i32> {
	if expand_p {
		winding03_impl::<true, FORWARD>(
			vert_pos_a,
			halfedge_a,
			vert_normal_a,
			vert_pos_b,
			halfedge_b,
			tri_normal_b,
			collider_b,
			vert_normal_b,
			p1q2,
		)
	} else {
		winding03_impl::<false, FORWARD>(
			vert_pos_a,
			halfedge_a,
			vert_normal_a,
			vert_pos_b,
			halfedge_b,
			tri_normal_b,
			collider_b,
			vert_normal_b,
			p1q2,
		)
	}
}

fn winding03_impl<const EXPAND_P: bool, const FORWARD: bool>(
	vert_pos_a: &[Point3<f64>],
	halfedge_a: &Halfedges,
	vert_normal_a: &[Vector3<f64>],
	vert_pos_b: &[Point3<f64>],
	halfedge_b: &Halfedges,
	tri_normal_b: &[Vector3<f64>],
	collider_b: Rc<BVHCollider>,
	vert_normal_b: &[Vector3<f64>],
	p1q2: &[[i32; 2]],
) -> Vec<i32> {
	// a: 0 (vert), b: 2 (face)
	let index = if FORWARD { 0 } else { 1 };

	let mut u_a = DisjointSets::new(vert_pos_a.len());
	for edge in 0..halfedge_a.len() {
		let start = halfedge_a.start[edge] as usize;
		let end = halfedge_a.end(edge) as usize;
		if start >= end {
			continue;
		}
		// check if the edge is broken
		let it = p1q2.partition_point(|collision_pair| (collision_pair[index] as usize) < edge);
		if it == p1q2.len() || p1q2[it][index] as usize != edge {
			u_a.unite(start, end);
		}
	}

	// find components, the hope is the number of components should be small
	let mut components = FxHashSet::default();
	for v in 0..vert_pos_a.len() {
		components.insert(u_a.find(v));
	}

	let verts = Vec::from_iter(components.into_iter().map(|c| c as i32));

	let k02 = Kernel02::<EXPAND_P, FORWARD> {
		vert_pos_a,
		vert_normal_a,
		vert_pos_b: vert_pos_b,
		halfedge_b: halfedge_b,
		tri_normal_b: tri_normal_b,
		vert_normal_b,
	};

	let mut w03 = vec![0; vert_pos_a.len()];
	let f = |i| vert_pos_a[verts[i] as usize];
	collider_b.collisions_from_fn::<false, _>(
		|query_idx, leaf_idx| {
			let (s02, z02) = k02.call(verts[query_idx] as usize, leaf_idx);
			if z02.is_finite() {
				// note that i is distinct on each thread, and verts contains unique
				// elements, so this does not require atomics
				w03[verts[query_idx] as usize] += s02 * (if FORWARD { 1 } else { -1 });
			}
		},
		f,
		verts.len(),
		true,
	);

	drop(collider_b);

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

struct Kernel02<'a, const EXPAND_P: bool, const FORWARD: bool> {
	vert_pos_a: &'a [Point3<f64>],
	vert_normal_a: &'a [Vector3<f64>],
	vert_pos_b: &'a [Point3<f64>],
	halfedge_b: &'a Halfedges,
	tri_normal_b: &'a [Vector3<f64>],
	vert_normal_b: &'a [Vector3<f64>],
}

impl<'a, const EXPAND_P: bool, const FORWARD: bool> Kernel02<'a, EXPAND_P, FORWARD> {
	fn call(&self, a0: usize, b2: usize) -> (i32, f64) {
		let edge_b = load_face_edges(self.halfedge_b, b2);
		self.call_with_edge(a0, b2, &edge_b)
	}

	fn call_with_edge(&self, a0: usize, b2: usize, edge_b: &[FaceEdge; 3]) -> (i32, f64) {
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
				self.vert_pos_a,
				self.vert_normal_a,
				self.vert_pos_b,
				&self.halfedge_b.pair,
				self.tri_normal_b,
				self.vert_normal_b,
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
			let vert_pos_a = self.vert_pos_a[a0];
			z02 = interpolate(yzz_rl[0], yzz_rl[1], vert_pos_a.y)[1];
			if FORWARD {
				if !shadows(vert_pos_a.z, z02, -self.tri_normal_b[b2].z) {
					s02 = 0;
				}
			} else {
				if !shadows(
					z02,
					vert_pos_a.z,
					with_sign(EXPAND_P, self.tri_normal_b[b2].z),
				) {
					s02 = 0;
				}
			}
		}

		(s02.into(), z02)
	}
}

#[inline(always)]
fn shadow01<const EXPAND_P: bool, const FORWARD: bool>(
	a0: usize,
	b1: usize,
	b1s: usize,
	b1e: usize,
	vert_pos_a: &[Point3<f64>],
	vert_normal_a: &[Vector3<f64>],
	vert_pos_b: &[Point3<f64>],
	halfedge_pair_b: &[i32],
	tri_normal_b: &[Vector3<f64>],
	vert_normal_b: &[Vector3<f64>],
) -> (i8, Vector2<f64>) {
	let a0x = vert_pos_a[a0].x;
	let b1sx = vert_pos_b[b1s].x;
	let b1ex = vert_pos_b[b1e].x;
	let a0xp = vert_normal_a[a0].x;
	let b1sxp = vert_normal_b[b1s].x;
	let b1exp = vert_normal_b[b1e].x;
	let mut s01 = if FORWARD {
		shadows(a0x, b1ex, with_sign(EXPAND_P, a0xp) - b1exp) as i8
			- shadows(a0x, b1sx, with_sign(EXPAND_P, a0xp) - b1sxp) as i8
	} else {
		shadows(b1sx, a0x, with_sign(EXPAND_P, b1sxp) - a0xp) as i8
			- shadows(b1ex, a0x, with_sign(EXPAND_P, b1exp) - a0xp) as i8
	};

	let mut yz01 = Vector2::from_element(f64::NAN);

	if s01 != 0 {
		yz01 = interpolate(vert_pos_b[b1s], vert_pos_b[b1e], vert_pos_a[a0].x);
		let b1pair = halfedge_pair_b[b1] as usize;
		let dir = tri_normal_b[b1 / 3].y + tri_normal_b[b1pair / 3].y;
		if FORWARD {
			if !shadows(vert_pos_a[a0].y, yz01[0], -dir) {
				s01 = 0;
			}
		} else {
			if !shadows(yz01[0], vert_pos_a[a0].y, with_sign(EXPAND_P, dir)) {
				s01 = 0;
			}
		}
	}

	(s01, yz01)
}

#[derive(Default, Copy, Clone)]
struct FaceEdge {
	edge: usize,
	start: usize,
	end: usize,
	is_forward: bool,
}

#[inline(always)]
fn load_face_edges(halfedges: &Halfedges, tri: usize) -> [FaceEdge; 3] {
	let mut edge = [FaceEdge::default(); 3];
	for i in 0..3 {
		let halfedge = 3 * tri + i;
		let start = halfedges.start[halfedge] as usize;
		let end = halfedges.start[3 * tri + next3_usize(i)] as usize;
		if start < end {
			edge[i] = FaceEdge {
				edge: halfedge,
				start,
				end,
				is_forward: true,
			};
		} else {
			edge[i] = FaceEdge {
				edge: halfedges.pair[halfedge] as usize,
				start: end,
				end: start,
				is_forward: false,
			};
		}
	}

	edge
}

///Intersect two projected segments aL-aR and bL-bR. The segments are ordered
///over the same x interval, and their y gaps must bracket zero. The returned
///value is (x, y, a.z, b.z) at the crossing.
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

///Interpolate the (y, z) of segment aL-aR at the given x. The choice of
///(x - aL) vs (x - aR) is the smaller in magnitude, which keeps FP error
///low near either endpoint. Domain check via DEBUG_ASSERT.
#[inline(always)]
fn interpolate(a_l: Point3<f64>, a_r: Point3<f64>, x: f64) -> Vector2<f64> {
	let dx_l = x - a_l.x;
	let dx_r = x - a_r.x;
	debug_assert!(dx_l * dx_r <= 0.0, "Boolean manifold error: not in domain");

	let use_l = dx_l.abs() < dx_r.abs();
	let d_lr = a_r - a_l;
	let lambda = (if use_l { dx_l } else { dx_r }) / d_lr.x;
	if !lambda.is_finite() || !d_lr.y.is_finite() || !d_lr.z.is_finite() {
		return Vector2::new(a_l.y, a_l.z);
	}

	let mut yz = Vector2::default();
	yz[0] = lambda * d_lr.y + (if use_l { a_l.y } else { a_r.y });
	yz[1] = lambda * d_lr.z + (if use_l { a_l.z } else { a_r.z });
	return yz;
}

///`p < q` with symbolic perturbation: when `p == q` exactly, `dir < 0`
///acts as the tiebreaker. Used to give consistent strict-ordering answers
///regardless of which side of an FP equality we land on.
#[inline(always)]
fn shadows(p: f64, q: f64, dir: f64) -> bool {
	if p == q { dir < 0.0 } else { p < q }
}

///Symbolic perturbation primitives shared by Boolean3 and Boolean2.
///Carefully designed to minimize FP rounding error and eliminate it at edge
///cases.
#[inline(always)]
fn with_sign(pos: bool, v: f64) -> f64 {
	if pos { v } else { -v }
}
