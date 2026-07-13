use crate::util::math::{atomic_add, next3_usize};
use crate::util::vec_ext;
use nalgebra::Vector3;

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
	start: Vec<i32>,
	paired: Vec<i32>,
	prop_vert: Vec<i32>,
}

impl Halfedges {
	///Create the halfedge_ data structure from a list of triangles. If the optional
	///prop2vert array is missing, it's assumed these triangles are are pointing to
	///both vert and propVert indices. If prop2vert is present, the triangles are
	///assumed to be pointing to propVert indices only. The prop2vert array is used
	///to map the propVert indices to vert indices.
	pub fn from_tri_indices(
		vert_count: usize,
		tri_vert: Vec<Vector3<i32>>,
		tri_prop: Option<Vec<Vector3<i32>>>,
	) -> Self {
		let num_tri = tri_vert.len();
		let has_prop = tri_prop.is_some();
		let num_halfedge = 3 * num_tri;
		let mut halfedge = unsafe { vec_ext::uninit(num_halfedge) };

		let vert_count = vert_count as i32;

		//PrepHalfedges start
		let mut ids = {
			let ids = if vert_count < (1 << 18) {
				// For small vertex count, it is faster to just do sorting
				let mut edge: Vec<u64> = unsafe { vec_ext::uninit(num_halfedge) };
				let mut set_edge = |e: usize, v0: i32, v1: i32| {
					edge[e] = (if v0 < v1 { 1 } else { 0 }) << 63
						| (v0.min(v1) as u64) << 32
						| (v0.max(v1) as u64);
				};

				let mut job = PrepHalfedges {
					halfedges: &mut halfedge,
					tri_vert,
					tri_prop: tri_prop.unwrap_or(Vec::new()),
					f: &mut set_edge,
				};

				if has_prop {
					for i in 0..num_tri {
						job.call::<true>(i);
					}
				} else {
					for i in 0..num_tri {
						job.call::<false>(i);
					}
				}

				drop(job);
				let mut ids: Vec<i32> = (0..num_halfedge as i32).collect();
				ids.sort_by_key(|&i| edge[i as usize]);
				ids
			} else {
				// For larger vertex count, we separate the ids into slices for halfedges
				// with the same smaller vertex.
				// We first copy them there (as HalfedgePairData), and then do sorting
				// locally for each slice.
				// This helps with memory locality, and is faster for larger meshes.
				let mut entries = unsafe { vec_ext::uninit(num_halfedge) };
				let mut offsets: Vec<i32> = vec![0; (vert_count * 2) as usize];
				let mut set_offset = |_, v0: i32, v1: i32| {
					let offset = if v0 > v1 { 0 } else { vert_count };
					atomic_add(&mut offsets[(v0.min(v1) + offset) as usize], 1);
				};

				let mut job = PrepHalfedges {
					halfedges: &mut halfedge,
					tri_vert,
					tri_prop: tri_prop.unwrap_or(Vec::new()),
					f: &mut set_offset,
				};

				if has_prop {
					for i in 0..num_tri {
						job.call::<true>(i);
					}
				} else {
					for i in 0..num_tri {
						job.call::<false>(i);
					}
				}

				drop(job);
				vec_ext::exclusive_scan_in_place(&mut offsets, 0);

				for tri in 0..num_tri {
					let tri = tri as i32;
					for i in 0..3 {
						let e = 3 * tri + i;
						let e_usize = e as usize;
						let v0 = halfedge[e_usize].start_vert;
						let v1 = halfedge[e_usize].end_vert;
						let offset = if v0 > v1 { 0 } else { vert_count };
						let start = v0.min(v1);
						let index = atomic_add(&mut offsets[(start + offset) as usize], 1);
						entries[index as usize] = HalfedgePairData {
							large_vert: v0.max(v1),
							tri,
							edge_index: e,
						};
					}
				}

				let mut ids: Vec<i32> = unsafe { vec_ext::uninit(num_halfedge) };
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

				ids
			};

			ids
		};

		//PrepHalfedges end

		// Mark opposed triangles for removal - this may strand unreferenced verts
		// which are removed later by self.remove_unreferenced_verts() and self.finish().
		let num_edge = (num_halfedge / 2) as i32;
		let mut removed = vec![false; num_halfedge];

		let mut consecutive_start = 0;
		for i in 0..num_edge {
			let pair0 = ids[i as usize];
			let h0 = halfedge[pair0 as usize];
			let mut k = num_edge + consecutive_start;
			loop {
				let pair1 = ids[k as usize];
				let h1 = halfedge[pair1 as usize];
				if h0.start_vert != h1.end_vert || h0.end_vert != h1.start_vert {
					break;
				}
				if !removed[pair1 as usize]
					&& halfedge[next_halfedge(pair0) as usize].end_vert
						== halfedge[next_halfedge(pair1) as usize].end_vert
				{
					removed[pair0 as usize] = true;
					removed[pair1 as usize] = true;
					if i + num_edge != k {
						// Reorder so that remaining edges pair up, while preserving relative
						// order between the edges (triangle id order)
						// cannot directly use move and move_backward because we need to keep
						// removed halfedges in-place
						let dir = if i + num_edge < k { 1 } else { -1 };
						let mut a = k;
						let mut b = k + dir;
						let is_removed =
							|x: i32, ids: &mut [i32]| removed[ids[x as usize] as usize];
						let in_range = |a: i32| {
							if dir > 0 {
								a >= i + num_edge
							} else {
								a <= i + num_edge
							}
						};
						loop {
							loop {
								a -= dir;
								if !(in_range(a) && is_removed(a, &mut ids)) {
									break;
								}
							}
							if !in_range(a) {
								break;
							}
							loop {
								b -= dir;
								if !(is_removed(b, &mut ids) && b != k) {
									break;
								}
							}
							ids[b as usize] = ids[a as usize];
						}
						ids[(i + num_edge) as usize] = pair1;
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
			let h1 = halfedge[ids[(i + 1) as usize] as usize];
			if h1.start_vert == h0.start_vert && h1.end_vert == h0.end_vert {
				continue;
			}

			consecutive_start = i + 1;
		}

		let mut out = unsafe { Self::uninit(num_halfedge) };
		for i in 0..num_edge {
			let pair0 = ids[i as usize];
			let pair1 = ids[(i + num_edge) as usize];
			if !removed[pair0 as usize] {
				out.set_start(pair0, halfedge[pair0 as usize].start_vert);
				out.set_prop(pair0, halfedge[pair0 as usize].prop_vert);
				out.set_pair(pair0, pair1);
				out.set_start(pair1, halfedge[pair1 as usize].start_vert);
				out.set_prop(pair1, halfedge[pair1 as usize].prop_vert);
				out.set_pair(pair1, pair0);
			} else {
				out.set_start(pair0, -1);
				out.set_prop(pair0, 0);
				out.set_pair(pair0, -1);
				out.set_start(pair1, -1);
				out.set_prop(pair1, 0);
				out.set_pair(pair1, -1);
			}
		}

		#[cfg(feature = "test_thoroughly")]
		for edge in 0..num_halfedge {
			let next = next_halfedge(edge as i32) as usize;
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

struct HalfedgePairData {
	large_vert: i32,
	tri: i32,
	edge_index: i32,
}

#[derive(Copy, Clone)]
struct CreateHalfedge {
	start_vert: i32,
	end_vert: i32,
	prop_vert: i32,
}

struct PrepHalfedges<'a, F: FnMut(usize, i32, i32)> {
	halfedges: &'a mut Vec<CreateHalfedge>,
	tri_vert: Vec<Vector3<i32>>,
	tri_prop: Vec<Vector3<i32>>,
	f: &'a mut F,
}

impl<'a, F: FnMut(usize, i32, i32)> PrepHalfedges<'a, F> {
	fn call<const HAS_PROP: bool>(&mut self, tri: usize) {
		let verts = self.tri_vert[tri];
		let props = if HAS_PROP {
			self.tri_prop[tri]
		} else {
			//should compile out
			Vector3::default()
		};

		for i in 0..3 {
			let j = next3_usize(i);
			let e = 3 * tri + i;
			let v0 = verts[i];
			let v1 = verts[j];
			debug_assert!(v0 != v1, "topological degeneracy");
			self.halfedges[e as usize] = CreateHalfedge {
				start_vert: v0,
				end_vert: v1,
				prop_vert: if HAS_PROP { props[i as usize] } else { 0 },
			};

			(self.f)(e, v0, v1);
		}
	}
}

impl Halfedges {
	pub unsafe fn uninit(size: usize) -> Self {
		unsafe {
			Self {
				start: vec_ext::uninit(size),
				paired: vec_ext::uninit(size),
				prop_vert: vec_ext::uninit(size),
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

	pub fn start(&self, idx: i32) -> i32 {
		self.start[idx as usize]
	}

	pub fn end(&self, idx: i32) -> i32 {
		self.start[next_halfedge(idx) as usize]
	}

	pub fn pair(&self, idx: i32) -> i32 {
		self.paired[idx as usize]
	}

	pub fn prop(&self, idx: i32) -> i32 {
		self.prop_vert[idx as usize]
	}

	pub fn set_start(&mut self, idx: i32, vert: i32) {
		self.start[idx as usize] = vert;
	}

	pub fn set_end(&mut self, idx: i32, vert: i32) {
		self.start[next_halfedge(idx) as usize] = vert;
	}

	pub fn set_pair(&mut self, idx: i32, pair: i32) {
		self.paired[idx as usize] = pair;
	}

	pub fn set_prop(&mut self, idx: i32, prop: i32) {
		self.prop_vert[idx as usize] = prop;
	}

	pub fn is_forward(&self, idx: i32) -> bool {
		self.start(idx) < self.end(idx)
	}

	pub fn get(&self, idx: i32) -> Halfedge {
		Halfedge {
			start_vert: self.start(idx),
			end_vert: self.end(idx),
			paired_halfedge: self.pair(idx),
			prop_vert: self.prop(idx),
		}
	}

	pub fn set(&mut self, idx: i32, start_vert: i32, paired_halfedge: i32, prop_vert: i32) {
		self.set_start(idx, start_vert);
		self.set_pair(idx, paired_halfedge);
		self.set_prop(idx, prop_vert);
	}

	pub fn push(&mut self, start_vert: i32, paired_halfedge: i32, prop_vert: i32) {
		self.start.push(start_vert);
		self.paired.push(paired_halfedge);
		self.prop_vert.push(prop_vert);
	}

	#[inline(always)]
	pub fn for_vert(&self, halfedge: i32, mut func: impl FnMut(i32)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.pair(current));
			func(current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline(always)]
	pub fn for_vert_mut(&mut self, halfedge: i32, mut func: impl FnMut(&mut Self, i32)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.pair(current));
			func(self, current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline(always)]
	pub fn for_vert_fn<T>(
		&self,
		halfedge: i32,
		mut transform: impl FnMut(i32) -> T,
		mut binary_op: impl FnMut(i32, &T, &mut T),
	) {
		let mut here: T = transform(halfedge);
		let mut current: i32 = halfedge;
		loop {
			let next_halfedge: i32 = next_halfedge(self.pair(current));
			let mut next: T = transform(next_halfedge);
			binary_op(current, &here, &mut next);
			here = next;
			current = next_halfedge;
			if current == halfedge {
				break;
			}
		}
	}

	pub fn pair_up(&mut self, edge0: i32, edge1: i32) {
		self.set_pair(edge0, edge1);
		self.set_pair(edge1, edge0);
	}

	//use when stride increases from 0->more than 0
	pub fn init_prop(&mut self) {
		self.prop_vert = self.start.clone();
	}

	/// Traverses CW around startEdge.endVert from startEdge to endEdge
	/// (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
	/// to vert instead.
	pub fn update_vert(&mut self, vert: i32, start_edge: i32, end_edge: i32) {
		let mut current = start_edge;
		while current != end_edge {
			self.set_end(current, vert);
			current = next_halfedge(current);
			self.set_start(current, vert);
			current = self.pair(current);
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
		(0..self.len()).all(|edge| check.call(edge as i32))
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
		halfedge.sort_by_key(|edge| (edge.start_vert, edge.end_vert));

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
		let mut data = unsafe { vec_ext::uninit(self.len()) };
		for idx in 0..self.len() {
			data[idx] = self.get(idx as i32);
		}
		data
	}
}

#[inline(always)]
pub fn next_halfedge(current: i32) -> i32 {
	current + (if current % 3 == 2 { -2 } else { 1 })
}

struct CheckHalfedges<'a> {
	halfedges: &'a Halfedges,
}

impl<'a> CheckHalfedges<'a> {
	fn call(&self, edge: i32) -> bool {
		let start = self.halfedges.start(edge);
		let end = self.halfedges.end(edge);
		let pair = self.halfedges.pair(edge);
		if start == -1 && end == -1 && pair == -1 {
			return true;
		}
		if self.halfedges.start(next_halfedge(edge)) == -1
			|| self.halfedges.start(next_halfedge(next_halfedge(edge))) == -1
		{
			return false;
		}
		if pair == -1 {
			return false;
		}

		let mut good = true;
		good &= self.halfedges.pair(pair) == edge;
		good &= start != end;
		good &= start == self.halfedges.end(pair);
		good &= end == self.halfedges.start(pair);
		good
	}
}
