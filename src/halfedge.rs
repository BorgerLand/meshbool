use crate::util::math::{atomic_add, next3_usize};
use crate::util::vec_ext;
use nalgebra::Vector3;
use std::array;

///Temporary or value-style halfedge record. Persistent Manifold storage uses
///Halfedges below, which derives endVert from the next halfedge in each face.
#[derive(Clone, Copy, Debug, Default)]
pub struct Halfedge {
	pub start_vert: i32,
	pub end_vert: i32,
	pub paired_halfedge: i32,
	pub prop_vert: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Halfedges {
	pub start: Vec<i32>, //index into vert_pos
	pub pair: Vec<i32>,  //index into halfedge.start
	pub prop: Vec<i32>,  //index into properties.data
}

impl Halfedges {
	///Create the halfedge_ data structure from a list of triangles. If the optional
	///triVert array is missing, it's assumed that triProp is identical to triVert.
	pub fn from_tri_indices(
		vert_count: usize,
		tri_vert: Vec<Vector3<i32>>,
		tri_prop: Option<Vec<Vector3<i32>>>,
	) -> Self {
		let num_tri = tri_vert.len();
		let num_halfedge = 3 * num_tri;
		let vert_count = vert_count as i32;

		//PrepHalfedges start
		let (halfedge, mut ids) = if vert_count < (1 << 18) {
			// For small vertex count, it is faster to just do sorting
			let mut edge = unsafe { vec_ext::uninit(num_halfedge) };
			let halfedge = prep_halfedges(tri_vert, tri_prop, |e, v0, v1| {
				edge[e] = (if v0 < v1 { 1 } else { 0 }) << 63
					| (v0.min(v1) as u64) << 32
					| (v0.max(v1) as u64);
			});

			let mut ids: Vec<i32> = (0..num_halfedge as i32).collect();
			ids.sort_unstable_by_key(|&i| edge[i as usize]);
			(halfedge, ids)
		} else {
			// For larger vertex count, we separate the ids into slices for halfedges
			// with the same smaller vertex.
			// We first copy them there (as HalfedgePairData), and then do sorting
			// locally for each slice.
			// This helps with memory locality, and is faster for larger meshes.

			let mut offsets = vec![0; (vert_count * 2) as usize];
			let halfedge = prep_halfedges(tri_vert, tri_prop, |_, v0, v1| {
				let offset = if v0 > v1 { 0 } else { vert_count };
				atomic_add(&mut offsets[(v0.min(v1) + offset) as usize], 1);
			});

			vec_ext::exclusive_scan_in_place(&mut offsets, 0);

			let mut entries = unsafe { vec_ext::uninit(num_halfedge) };
			for tri in 0..num_tri {
				let tri = tri as i32;
				for i in 0..3 {
					let e = 3 * tri + i;
					let e_usize = e as usize;
					let v0 = halfedge[e_usize].start_vert;
					let v1 = halfedge[e_usize].end_vert;
					let offset = if v0 > v1 { 0 } else { vert_count };
					let start = v0.min(v1);
					let index = atomic_add(&mut offsets[(start + offset) as usize], 1) as usize;
					entries[index] = HalfedgePairData {
						large_vert: v0.max(v1),
						tri,
						edge_index: e,
					};
				}
			}

			let mut ids = unsafe { vec_ext::uninit(num_halfedge) };
			for v in 0..offsets.len() {
				let start = if v == 0 { 0 } else { offsets[v - 1] };
				let end = offsets[v];
				for i in start..end {
					ids[i as usize] = i;
				}

				ids[start as usize..end as usize].sort_unstable_by_key(|&i| {
					let entry = &entries[i as usize];
					(entry.large_vert, entry.tri)
				});

				for i in start..end {
					let i = i as usize;
					ids[i] = entries[ids[i] as usize].edge_index;
				}
			}

			(halfedge, ids)
		};

		//PrepHalfedges end

		// Mark opposed triangles for removal - this may strand unreferenced verts
		// which are removed later by self.remove_unreferenced_verts() and self.finish().
		let num_edge = num_halfedge / 2;
		let mut removed = vec![false; num_halfedge];

		let mut consecutive_start = 0;
		for i in 0..num_edge {
			let pair0 = ids[i] as usize;
			let h0 = halfedge[pair0];
			let mut k = num_edge + consecutive_start;
			loop {
				let pair1 = ids[k] as usize;
				let h1 = halfedge[pair1];
				if h0.start_vert != h1.end_vert || h0.end_vert != h1.start_vert {
					break;
				}
				if !removed[pair1]
					&& halfedge[next_halfedge(pair0)].end_vert
						== halfedge[next_halfedge(pair1)].end_vert
				{
					removed[pair0] = true;
					removed[pair1] = true;
					if i + num_edge != k {
						// Reorder so that remaining edges pair up, while preserving relative
						// order between the edges (triangle id order)
						// cannot directly use move and move_backward because we need to keep
						// removed halfedges in-place
						let dir = i + num_edge < k;
						let mut a = k;
						let mut b = if dir { k + 1 } else { k - 1 };
						let is_removed = |x: usize, ids: &[i32]| removed[ids[x] as usize];
						let in_range = |a: usize| {
							if dir {
								a >= i + num_edge
							} else {
								a <= i + num_edge
							}
						};
						loop {
							loop {
								if dir {
									a -= 1;
								} else {
									a += 1;
								}
								if !(in_range(a) && is_removed(a, &ids)) {
									break;
								}
							}
							if !in_range(a) {
								break;
							}
							loop {
								if dir {
									b -= 1;
								} else {
									b += 1;
								}
								if !(is_removed(b, &ids) && b != k) {
									break;
								}
							}
							ids[b] = ids[a];
						}
						ids[i + num_edge] = pair1 as i32;
					}
					break;
				}

				k += 1;
				if k >= num_edge * 2 {
					break;
				}
			}

			if i + 1 == num_edge {
				continue;
			}
			let h1 = halfedge[ids[i + 1] as usize];
			if h1.start_vert == h0.start_vert && h1.end_vert == h0.end_vert {
				continue;
			}

			consecutive_start = i + 1;
		}

		let mut out = unsafe { Self::uninit(num_halfedge) };
		for i in 0..num_edge {
			let pair0_i32 = ids[i];
			let pair1_i32 = ids[i + num_edge];
			let pair0_usize = pair0_i32 as usize;
			let pair1_usize = pair1_i32 as usize;
			if !removed[pair0_usize] {
				out.start[pair0_usize] = halfedge[pair0_usize].start_vert;
				out.prop[pair0_usize] = halfedge[pair0_usize].prop_vert;
				out.pair[pair0_usize] = pair1_i32;
				out.start[pair1_usize] = halfedge[pair1_usize].start_vert;
				out.prop[pair1_usize] = halfedge[pair1_usize].prop_vert;
				out.pair[pair1_usize] = pair0_i32;
			} else {
				out.start[pair0_usize] = -1;
				out.prop[pair0_usize] = 0;
				out.pair[pair0_usize] = -1;
				out.start[pair1_usize] = -1;
				out.prop[pair1_usize] = 0;
				out.pair[pair1_usize] = -1;
			}
		}

		#[cfg(feature = "test_thoroughly")]
		for edge in 0..num_halfedge {
			let next = next_halfedge(edge);
			if !removed[edge] && !removed[next] {
				debug_assert!(
					halfedge[edge].end_vert == halfedge[next].start_vert,
					"CreateHalfedges requires triangle-ordered edges!"
				);
			}
		}

		out
	}
}

#[derive(Copy, Clone)]
struct CreateHalfedge {
	start_vert: i32,
	end_vert: i32,
	prop_vert: i32,
}

fn prep_halfedges(
	tri_vert: Vec<Vector3<i32>>,
	tri_prop: Option<Vec<Vector3<i32>>>,
	f: impl FnMut(usize, i32, i32),
) -> Vec<CreateHalfedge> {
	if let Some(tri_prop) = tri_prop {
		prep_halfedges_impl::<true>(tri_vert, tri_prop, f)
	} else {
		prep_halfedges_impl::<false>(tri_vert, Vec::new(), f)
	}
}

fn prep_halfedges_impl<const HAS_PROP: bool>(
	tri_vert: Vec<Vector3<i32>>,
	tri_prop: Vec<Vector3<i32>>,
	mut f: impl FnMut(usize, i32, i32),
) -> Vec<CreateHalfedge> {
	(0..tri_vert.len())
		.flat_map(|tri| {
			let verts = tri_vert[tri];
			let props = if HAS_PROP {
				tri_prop[tri]
			} else {
				Vector3::default()
			};

			array::from_fn::<_, 3, _>(|i| {
				let j = next3_usize(i);
				let e = 3 * tri + i;
				let v0 = verts[i];
				let v1 = verts[j];
				debug_assert!(v0 != v1, "topological degeneracy");
				f(e, v0, v1);

				CreateHalfedge {
					start_vert: v0,
					end_vert: v1,
					prop_vert: if HAS_PROP { props[i] } else { verts[i] },
				}
			})
		})
		.collect()
}

struct HalfedgePairData {
	large_vert: i32,
	tri: i32,
	edge_index: i32,
}

impl Halfedges {
	pub unsafe fn uninit(size: usize) -> Self {
		unsafe {
			Self {
				start: vec_ext::uninit(size),
				pair: vec_ext::uninit(size),
				prop: vec_ext::uninit(size),
			}
		}
	}

	pub fn len(&self) -> usize {
		self.start.len()
	}

	pub fn num_tri(&self) -> usize {
		self.len() / 3
	}

	pub fn num_edge(&self) -> usize {
		self.len() / 2
	}

	pub fn end(&self, idx: usize) -> i32 {
		self.start[next_halfedge(idx)]
	}

	pub fn set_end(&mut self, idx: usize, vert: i32) {
		self.start[next_halfedge(idx)] = vert;
	}

	pub fn is_forward(&self, idx: usize) -> bool {
		self.start[idx] < self.end(idx)
	}

	pub fn valid(&self, idx: usize) -> bool {
		self.pair[idx] >= 0
	}

	pub fn tri(&self, idx: usize) -> usize {
		idx / 3
	}

	pub fn prop_end(&self, idx: usize) -> i32 {
		self.prop[next_halfedge(idx)]
	}

	pub fn get(&self, idx: usize) -> Halfedge {
		Halfedge {
			start_vert: self.start[idx],
			end_vert: self.end(idx),
			paired_halfedge: self.pair[idx],
			prop_vert: self.prop[idx],
		}
	}

	pub fn set(&mut self, idx: usize, start_vert: i32, paired_halfedge: i32, prop_vert: i32) {
		self.start[idx] = start_vert;
		self.pair[idx] = paired_halfedge;
		self.prop[idx] = prop_vert;
	}

	pub fn push(&mut self, start_vert: i32, paired_halfedge: i32, prop_vert: i32) {
		self.start.push(start_vert);
		self.pair.push(paired_halfedge);
		self.prop.push(prop_vert);
	}

	#[inline(always)]
	pub fn for_vert(&self, halfedge: usize, mut func: impl FnMut(usize)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.pair[current] as usize);
			func(current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline(always)]
	pub fn for_vert_mut(&mut self, halfedge: usize, mut func: impl FnMut(&mut Self, usize)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.pair[current] as usize);
			func(self, current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline(always)]
	pub fn for_vert_fn<T>(
		&self,
		halfedge: usize,
		mut transform: impl FnMut(usize) -> T,
		mut binary_op: impl FnMut(usize, &T, &mut T),
	) {
		let mut here = transform(halfedge);
		let mut current = halfedge;
		loop {
			let next_halfedge = next_halfedge(self.pair[current] as usize);
			let mut next = transform(next_halfedge);
			binary_op(current, &here, &mut next);
			here = next;
			current = next_halfedge;
			if current == halfedge {
				break;
			}
		}
	}

	pub fn pair_up(&mut self, edge0: usize, edge1: usize) {
		self.pair[edge0] = edge1 as i32;
		self.pair[edge1] = edge0 as i32;
	}

	/// Traverses CW around startEdge.endVert from startEdge to endEdge
	/// (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
	/// to vert instead.
	pub fn update_vert(&mut self, vert: i32, start_edge: usize, end_edge: usize) {
		let mut current = start_edge;
		while current != end_edge {
			self.set_end(current, vert);
			current = next_halfedge(current);
			self.start[current] = vert;
			current = self.pair[current] as usize;
			debug_assert!(current != start_edge, "infinite loop in decimator!");
		}
	}

	///Returns true if this manifold is in fact an oriented even manifold and all of
	///the data structures are consistent.
	pub fn is_manifold(&self) -> bool {
		if self.len() == 0 {
			return true;
		}
		if self.len() % 3 != 0 {
			return false;
		}
		let check = CheckHalfedges { halfedges: self };
		(0..self.len()).all(|edge| check.call(edge))
	}

	///Returns true if this manifold is in fact an oriented 2-manifold and all of
	///the data structures are consistent.
	#[cfg(feature = "test_thoroughly")]
	pub fn is_2_manifold(&self) -> bool {
		if self.len() == 0 {
			return true;
		}
		if !self.is_manifold() {
			return false;
		}

		let mut halfedge = self.to_data();
		halfedge.sort_unstable_by_key(|edge| (edge.start_vert, edge.end_vert));

		(0..(2 * self.num_edge() - 1)).all(|edge| {
			let h = halfedge[edge];
			if h.start_vert == -1 && h.end_vert == -1 && h.paired_halfedge == -1 {
				return true;
			}

			h.start_vert != halfedge[edge + 1].start_vert
				|| h.end_vert != halfedge[edge + 1].end_vert
		})
	}

	#[cfg(feature = "test_thoroughly")]
	fn to_data(&self) -> Vec<Halfedge> {
		(0..self.len()).map(|idx| self.get(idx)).collect()
	}
}

#[inline(always)]
pub fn next_halfedge(current: usize) -> usize {
	if current % 3 == 2 {
		current - 2
	} else {
		current + 1
	}
}

#[inline(always)]
pub fn prev_halfedge(current: usize) -> usize {
	if current % 3 == 0 {
		current + 2
	} else {
		current - 1
	}
}

struct CheckHalfedges<'a> {
	halfedges: &'a Halfedges,
}

impl<'a> CheckHalfedges<'a> {
	fn call(&self, edge: usize) -> bool {
		let start = self.halfedges.start[edge];
		let end = self.halfedges.end(edge);
		let pair = self.halfedges.pair[edge];
		if start == -1 && end == -1 && pair == -1 {
			return true;
		}
		if self.halfedges.start[next_halfedge(edge)] == -1
			|| self.halfedges.start[next_halfedge(next_halfedge(edge))] == -1
		{
			return false;
		}
		if pair == -1 {
			return false;
		}

		let mut good = true;
		let pair = pair as usize;
		good &= self.halfedges.pair[pair] == edge as i32;
		good &= start != end;
		good &= start == self.halfedges.end(pair);
		good &= end == self.halfedges.start[pair];
		good
	}
}
