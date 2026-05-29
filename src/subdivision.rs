use std::collections::HashMap;
use std::sync::Mutex;

use crate::MeshBoolImpl;
use crate::common::lerp;
use crate::meshboolimpl::BaryIndices;
use crate::parallel::exclusive_scan_iter;
use crate::shared::{Barycentric, TmpEdge, TriRef, create_tmp_edges, next_halfedge};
use crate::utils::next3_i32;
use crate::vec::vec_resize_nofill;
use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3, Vector4};

use std::sync::LazyLock;

static PARTITION_CACHE: LazyLock<Mutex<HashMap<Vector4<i32>, Partition>>> =
	LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Default, Clone)]
pub struct Partition {
	// The cached partitions don't have idx - it's added to the copy returned
	// from GetPartition that contains the mapping of the input divisions into the
	// sorted divisions that are uniquely cached.
	pub idx: Vector4<i32>,
	pub sorted_divisions: Vector4<i32>,
	pub vert_bary: Vec<Vector4<f64>>,
	pub tri_vert: Vec<Vector3<i32>>,
}

impl Partition {
	pub fn interior_offset(&self) -> i32 {
		return self.sorted_divisions[0]
			+ self.sorted_divisions[1]
			+ self.sorted_divisions[2]
			+ self.sorted_divisions[3];
	}

	pub fn num_interior(&self) -> i32 {
		return self.vert_bary.len() as i32 - self.interior_offset();
	}

	pub fn get_partition(divisions: Vector4<i32>) -> Self {
		if divisions[0] == 0 {
			return Self::default(); // skip wrong side of quad
		}

		let mut sorted_div: Vector4<i32> = divisions;
		let mut tri_idx = Vector4::new(0i32, 1i32, 2i32, 3i32);
		if divisions[3] == 0 {
			// triangle
			if sorted_div[2] > sorted_div[1] {
				sorted_div.as_mut_slice().swap(2, 1);
				tri_idx.as_mut_slice().swap(2, 1);
			}
			if sorted_div[1] > sorted_div[0] {
				sorted_div.as_mut_slice().swap(1, 0);
				tri_idx.as_mut_slice().swap(1, 0);
				if sorted_div[2] > sorted_div[1] {
					sorted_div.as_mut_slice().swap(2, 1);
					tri_idx.as_mut_slice().swap(2, 1);
				}
			}
		} else {
			// quad
			let mut min_idx: i32 = 0;
			let mut min: i32 = divisions[min_idx as usize];
			let mut next: i32 = divisions[1];
			for i in 1..4 {
				let n = divisions[(i + 1) % 4];
				if divisions[i] < min || (divisions[i] == min && n < next) {
					min_idx = i as i32;
					min = divisions[i];
					next = n;
				}
			}
			// Backwards (mirrored) quads get a separate cache key for now for
			// simplicity, so there is no reversal necessary for quads when
			// re-indexing.
			let tmp: Vector4<i32> = sorted_div;
			for i in 0..4 {
				tri_idx[i] = (i as i32 + min_idx) % 4;
				sorted_div[i] = tmp[tri_idx[i] as usize];
			}
		}

		let mut partition: Self = Self::get_cached_partition(sorted_div);
		partition.idx = tri_idx;

		return partition;
	}

	pub fn reindex(
		&self,
		tri_verts: Vector4<i32>,
		edge_offsets: Vector4<i32>,
		mut edge_fwd: Vector4<bool>,
		interior_offset: i32,
	) -> Vec<Vector3<i32>> {
		let mut new_verts: Vec<i32> = Vec::with_capacity(self.vert_bary.len());
		let mut tri_idx = self.idx;
		let mut out_tri = Vector4::new(0i32, 1, 2, 3);
		if tri_verts[3] < 0 && self.idx[1] != next3_i32(self.idx[0]) {
			tri_idx = Vector4::new(self.idx[2], self.idx[0], self.idx[1], self.idx[3]);
			edge_fwd.iter_mut().for_each(|b| *b = !*b);
			out_tri.as_mut_slice().swap(0, 1);
		}
		for i in 0..4 {
			if tri_verts[tri_idx[i] as usize] >= 0 {
				new_verts.push(tri_verts[tri_idx[i] as usize]);
			}
		}
		for i in 0..4 {
			let n = self.sorted_divisions[i] - 1;
			let mut offset = edge_offsets[self.idx[i] as usize]
				+ (if edge_fwd[self.idx[i] as usize] {
					0
				} else {
					n - 1
				});
			for _ in 0..n {
				new_verts.push(offset);
				offset += if edge_fwd[self.idx[i] as usize] {
					1
				} else {
					-1
				};
			}
		}

		let offset = interior_offset - new_verts.len() as i32;
		let old: usize = new_verts.len();
		unsafe { vec_resize_nofill(&mut new_verts, self.vert_bary.len()) };
		for (i, v) in new_verts[old..].iter_mut().enumerate() {
			*v = old as i32 + offset + i as i32;
		}

		let num_tri = self.tri_vert.len() as i32;
		let mut new_tri_vert: Vec<Vector3<i32>> = vec![Default::default(); num_tri as usize];
		new_tri_vert
			.iter_mut()
			.enumerate()
			.for_each(|(tri_idx, tri)| {
				for j in 0..3 {
					tri[out_tri[j] as usize] = new_verts[self.tri_vert[tri_idx][j] as usize];
				}
			});
		return new_tri_vert;
	}

	// This triangulation is purely topological - it depends only on the number of
	// divisions of the three sides of the triangle. This allows them to be cached
	// and reused for similar triangles. The shape of the final surface is defined
	// by the tangents and the barycentric coordinates of the new verts. For
	// triangles, the input must be sorted: n[0] >= n[1] >= n[2] > 0.
	fn get_cached_partition(n: Vector4<i32>) -> Self {
		{
			let lock = PARTITION_CACHE.lock().unwrap();
			if let Some(val) = lock.get(&n) {
				return val.clone();
			}
		}
		let mut partition = Self::default();
		partition.sorted_divisions = n;
		if n[3] > 0 {
			// quad
			partition.vert_bary.push(Vector4::new(1.0, 0.0, 0.0, 0.0));
			partition.vert_bary.push(Vector4::new(0.0, 1.0, 0.0, 0.0));
			partition.vert_bary.push(Vector4::new(0.0, 0.0, 1.0, 0.0));
			partition.vert_bary.push(Vector4::new(0.0, 0.0, 0.0, 1.0));
			let mut edge_offsets: Vector4<i32> = Default::default();
			edge_offsets[0] = 4;
			for i in 0..4 {
				if i > 0 {
					edge_offsets[i] = edge_offsets[i - 1] + n[i - 1] - 1;
				}
				let next_bary: Vector4<f64> = partition.vert_bary[(i + 1) % 4];
				for j in 1..n[i] {
					partition
						.vert_bary
						.push(partition.vert_bary[i].lerp(&next_bary, j as f64 / n[i] as f64));
				}
			}
			Self::partition_quad(
				&mut partition.tri_vert,
				&mut partition.vert_bary,
				Vector4::new(0, 1, 2, 3),
				edge_offsets,
				n - Vector4::repeat(1i32),
				Vector4::new(true, true, true, true),
			);
		} else {
			// tri
			partition.vert_bary.push(Vector4::new(1.0, 0.0, 0.0, 0.0));
			partition.vert_bary.push(Vector4::new(0.0, 1.0, 0.0, 0.0));
			partition.vert_bary.push(Vector4::new(0.0, 0.0, 1.0, 0.0));
			for i in 0..3 {
				let next_bary: Vector4<f64> = partition.vert_bary[(i + 1) % 3];
				for j in 1..n[i] {
					partition
						.vert_bary
						.push(partition.vert_bary[i].lerp(&next_bary, j as f64 / n[i] as f64));
				}
			}
			let edge_offsets = Vector3::new(3i32, 3 + n[0] - 1, 3 + n[0] - 1 + n[1] - 1);

			let f: f64 = (n[2] * n[2] + n[0] * n[0]) as f64;
			if n[1] == 1 {
				if n[0] == 1 {
					partition.tri_vert.push(Vector3::new(0, 1, 2));
				} else {
					Self::partition_fan(
						&mut partition.tri_vert,
						Vector3::new(0, 1, 2),
						n[0] - 1,
						edge_offsets[0],
					);
				}
			} else if n[1] as f64 * n[1] as f64 > f - 2.0f64.sqrt() * n[0] as f64 * n[2] as f64 {
				// acute-ish
				partition
					.tri_vert
					.push(Vector3::new(edge_offsets[1] - 1, 1, edge_offsets[1]));
				Self::partition_quad(
					&mut partition.tri_vert,
					&mut partition.vert_bary,
					Vector4::new(edge_offsets[1] - 1, edge_offsets[1], 2, 0),
					Vector4::new(-1, edge_offsets[1] + 1, edge_offsets[2], edge_offsets[0]),
					Vector4::new(0, n[1] - 2, n[2] - 1, n[0] - 2),
					Vector4::new(true, true, true, true),
				);
			} else {
				// obtuse -> spit into two acute
				// portion of n[0] under n[2]
				let ns: i32 = (n[0] - 2)
					.min(f64::round((f - (n[1] * n[1]) as f64) / (2 * n[0]) as f64) as i32);
				// height from n[0]: nh <= n[2]
				let nh: i32 = 1.0f64.max(((n[2] * n[2] - ns * ns) as f64).sqrt().round()) as i32;

				let h_offset: i32 = partition.vert_bary.len() as i32;
				let middle_bary: Vector4<f64> =
					partition.vert_bary[(edge_offsets[0] + ns - 1) as usize];
				for j in 1..nh {
					partition
						.vert_bary
						.push(partition.vert_bary[2].lerp(&middle_bary, j as f64 / nh as f64));
				}

				partition
					.tri_vert
					.push(Vector3::new(edge_offsets[1] - 1, 1, edge_offsets[1]));
				Self::partition_quad(
					&mut partition.tri_vert,
					&mut partition.vert_bary,
					Vector4::new(
						edge_offsets[1] - 1,
						edge_offsets[1],
						2,
						edge_offsets[0] + ns - 1,
					),
					Vector4::new(-1, edge_offsets[1] + 1, h_offset, edge_offsets[0] + ns),
					Vector4::new(0, n[1] - 2, nh - 1, n[0] - ns - 2),
					Vector4::new(true, true, true, true),
				);

				if n[2] == 1 {
					Self::partition_fan(
						&mut partition.tri_vert,
						Vector3::new(0, edge_offsets[0] + ns - 1, 2),
						ns - 1,
						edge_offsets[0],
					);
				} else {
					if ns == 1 {
						partition
							.tri_vert
							.push(Vector3::new(h_offset, 2, edge_offsets[2]));
						Self::partition_quad(
							&mut partition.tri_vert,
							&mut partition.vert_bary,
							Vector4::new(h_offset, edge_offsets[2], 0, edge_offsets[0]),
							Vector4::new(-1, edge_offsets[2] + 1, -1, h_offset + nh - 2),
							Vector4::new(0, n[2] - 2, ns - 1, nh - 2),
							Vector4::new(true, true, true, false),
						);
					} else {
						partition
							.tri_vert
							.push(Vector3::new(h_offset - 1, 0, edge_offsets[0]));
						Self::partition_quad(
							&mut partition.tri_vert,
							&mut partition.vert_bary,
							Vector4::new(
								h_offset - 1,
								edge_offsets[0],
								edge_offsets[0] + ns - 1,
								2,
							),
							Vector4::new(
								-1,
								edge_offsets[0] + 1,
								h_offset + nh - 2,
								edge_offsets[2],
							),
							Vector4::new(0, ns - 2, nh - 1, n[2] - 2),
							Vector4::new(true, true, false, true),
						);
					}
				}
			}
		}

		let mut lock = PARTITION_CACHE.lock().unwrap();
		lock.insert(n, partition.clone());
		return partition;
	}

	// Side 0 has added edges while sides 1 and 2 do not. Fan spreads from vert 2.
	fn partition_fan(
		tri_vert: &mut Vec<Vector3<i32>>,
		corner_verts: Vector3<i32>,
		added: i32,
		edge_offset: i32,
	) {
		let mut last: i32 = corner_verts[0];
		for i in 0..added {
			let next = edge_offset + i;
			tri_vert.push(Vector3::new(last, next, corner_verts[2]));
			last = next;
		}
		tri_vert.push(Vector3::new(last, corner_verts[1], corner_verts[2]));
	}

	// Partitions are parallel to the first edge unless two consecutive edgeAdded
	// are zero, in which case a terminal triangulation is performed.
	fn partition_quad(
		tri_vert: &mut Vec<Vector3<i32>>,
		vert_bary: &mut Vec<Vector4<f64>>,
		corner_verts: Vector4<i32>,
		edge_offsets: Vector4<i32>,
		edge_added: Vector4<i32>,
		edge_fwd: Vector4<bool>,
	) {
		let get_edge_vert = |edge: i32, idx: i32| {
			edge_offsets[edge as usize] + (if edge_fwd[edge as usize] { 1 } else { -1 }) * idx
		};

		debug_assert!(edge_added.ge(&Vector4::repeat(0i32)), "negative divisions!");

		let mut corner: i32 = -1;
		let mut last: i32 = 3;
		let mut max_edge: i32 = -1;
		for i in 0..4 {
			if corner == -1 && edge_added[i] == 0 && edge_added[last as usize] == 0 {
				corner = i as i32;
			}
			if edge_added[i] > 0 {
				max_edge = if max_edge == -1 { i as i32 } else { -2 };
			}
			last = i as i32;
		}
		if corner >= 0 {
			// terminate
			if max_edge >= 0 {
				let mut edge: Vector4<i32> =
					Vector4::new(0i32, 1, 2, 3) + Vector4::repeat(max_edge);
				edge.iter_mut().for_each(|v| *v %= 4);
				let middle = edge_added[max_edge as usize] / 2;
				tri_vert.push(Vector3::new(
					corner_verts[edge[2] as usize],
					corner_verts[edge[3] as usize],
					get_edge_vert(max_edge, middle),
				));
				let mut last: i32 = corner_verts[edge[0] as usize];
				for i in 0..=middle {
					let next = get_edge_vert(max_edge, i);
					tri_vert.push(Vector3::new(corner_verts[edge[3] as usize], last, next));
					last = next;
				}
				last = corner_verts[edge[1] as usize];
				for i in (middle..=(edge_added[max_edge as usize] - 1)).rev() {
					let next = get_edge_vert(max_edge, i);
					tri_vert.push(Vector3::new(corner_verts[edge[2] as usize], next, last));
					last = next;
				}
			} else {
				let mut side_vert = corner_verts[0]; // initial value is unused
				for j in [1, 2] {
					let side = (corner + j) % 4;
					if j == 2 && edge_added[side as usize] > 0 {
						tri_vert.push(Vector3::new(
							corner_verts[side as usize],
							get_edge_vert(side, 0),
							side_vert,
						));
					} else {
						side_vert = corner_verts[side as usize];
					}
					for i in 0..edge_added[side as usize] {
						let next_vert = get_edge_vert(side, i);
						tri_vert.push(Vector3::new(
							corner_verts[corner as usize],
							side_vert,
							next_vert,
						));
						side_vert = next_vert;
					}
					if j == 2 || edge_added[side as usize] == 0 {
						tri_vert.push(Vector3::new(
							corner_verts[corner as usize],
							side_vert,
							corner_verts[(corner + j + 1) as usize % 4],
						));
					}
				}
			}
			return;
		}
		// recursively partition
		let partitions = 1 + edge_added[1].min(edge_added[3]);
		let mut new_corner_verts = Vector4::new(corner_verts[1], -1i32, -1, corner_verts[0]);
		let mut new_edge_offsets = Vector4::new(
			edge_offsets[1],
			-1i32,
			get_edge_vert(3, edge_added[3] + 1),
			edge_offsets[0],
		);
		let mut new_edge_added = Vector4::new(0i32, -1, 0, edge_added[0]);
		let mut new_edge_fwd = Vector4::new(edge_fwd[1], true, edge_fwd[3], edge_fwd[0]);

		for i in 1..partitions {
			let corner_offset1 = (edge_added[1] * i) / partitions;
			let corner_offset3 = edge_added[3] - 1 - (edge_added[3] * i) / partitions;
			let next_offset1 = get_edge_vert(1, corner_offset1 + 1);
			let next_offset3 = get_edge_vert(3, corner_offset3 + 1);
			let added = lerp(
				edge_added[0] as f64,
				edge_added[2] as f64,
				i as f64 / partitions as f64,
			)
			.round() as i32;

			new_corner_verts[1] = get_edge_vert(1, corner_offset1);
			new_corner_verts[2] = get_edge_vert(3, corner_offset3);
			new_edge_added[0] = (next_offset1 - new_edge_offsets[0]).abs() - 1;
			new_edge_added[1] = added;
			new_edge_added[2] = (next_offset3 - new_edge_offsets[2]).abs() - 1;
			new_edge_offsets[1] = vert_bary.len() as i32;
			new_edge_offsets[2] = next_offset3;

			for j in 0..added {
				vert_bary.push(vert_bary[new_corner_verts[1] as usize].lerp(
					&vert_bary[new_corner_verts[2] as usize],
					(j + 1) as f64 / (added + 1) as f64,
				));
			}

			Self::partition_quad(
				tri_vert,
				vert_bary,
				new_corner_verts,
				new_edge_offsets,
				new_edge_added,
				new_edge_fwd,
			);

			new_corner_verts[0] = new_corner_verts[1];
			new_corner_verts[3] = new_corner_verts[2];
			new_edge_added[3] = new_edge_added[1];
			new_edge_offsets[0] = next_offset1;
			new_edge_offsets[3] = new_edge_offsets[1] + new_edge_added[1] - 1;
			new_edge_fwd[3] = false;
		}

		new_corner_verts[1] = corner_verts[2];
		new_corner_verts[2] = corner_verts[3];
		new_edge_offsets[1] = edge_offsets[2];
		new_edge_added[0] = edge_added[1] - (new_edge_offsets[0] - edge_offsets[1]).abs();
		new_edge_added[1] = edge_added[2];
		new_edge_added[2] = (new_edge_offsets[2] - edge_offsets[3]).abs() - 1;
		new_edge_offsets[2] = edge_offsets[3];
		new_edge_fwd[1] = edge_fwd[2];

		Self::partition_quad(
			tri_vert,
			vert_bary,
			new_corner_verts,
			new_edge_offsets,
			new_edge_added,
			new_edge_fwd,
		);
	}
}

impl MeshBoolImpl {
	///Returns the tri side index (0-2) connected to the other side of this quad if
	///this tri is part of a quad, or -1 otherwise.
	fn get_neighbor(&self, tri: i32) -> i32 {
		let mut neighbor = -1;
		for i in 0..3 {
			if self.is_marked_inside_quad(3 * tri + i) {
				neighbor = if neighbor == -1 { i } else { -2 };
			}
		}
		return neighbor;
	}

	///For the given triangle index, returns either the three halfedge indices of
	///that triangle and halfedges[3] = -1, or if the triangle is part of a quad, it
	///returns those four indices. If the triangle is part of a quad and is not the
	///lower of the two triangle indices, it returns all -1s.
	pub fn get_halfedges(&self, tri: i32) -> Vector4<i32> {
		let mut halfedges = Vector4::repeat(-1i32);
		for i in 0..3 {
			halfedges[i as usize] = 3 * tri + i as i32;
		}
		let neighbor = self.get_neighbor(tri);
		if neighbor >= 0 {
			// quad
			let pair = self.halfedge[(3 * tri + neighbor) as usize].paired_halfedge;
			if pair / 3 < tri {
				return Vector4::repeat(-1i32); // only process lower tri index
			}
			// The order here matters to keep small quads split the way they started, or
			// else it can create a 4-manifold edge.
			halfedges[2] = next_halfedge(halfedges[neighbor as usize]);
			halfedges[3] = next_halfedge(halfedges[2]);
			halfedges[0] = next_halfedge(pair);
			halfedges[1] = next_halfedge(halfedges[0]);
		}
		return halfedges;
	}

	///Returns the BaryIndices, which gives the tri and indices (0-3), such that
	///GetHalfedges(val.tri)[val.start4] points back to this halfedge, and val.end4
	///will point to the next one. This function handles this for both triangles and
	///quads. Returns {-1, -1, -1} if the edge is the interior of a quad.
	fn get_indices(&self, halfedge: i32) -> BaryIndices {
		let mut tri = halfedge / 3;
		let mut idx = halfedge % 3;
		let neighbor = self.get_neighbor(tri);
		if idx == neighbor {
			return BaryIndices::new(-1, -1, -1);
		}

		if neighbor < 0 {
			// tri
			return BaryIndices::new(tri, idx, next3_i32(idx));
		} else {
			// quad
			let pair = self.halfedge[(3 * tri + neighbor) as usize].paired_halfedge;
			if pair / 3 < tri {
				tri = pair / 3;
				idx = if next3_i32(neighbor) == idx { 0 } else { 1 };
			} else {
				idx = if next3_i32(neighbor) == idx { 2 } else { 3 };
			}
			return BaryIndices::new(tri, idx, (idx + 1) % 4);
		}
	}

	///Retained verts are part of several triangles, and it doesn't matter which one
	///the vertBary refers to. Here, whichever is last will win and it's done on the
	///CPU for simplicity for now. Using AtomicCAS on .tri should work for a GPU
	///version if desired.
	fn fill_retained_verts(&mut self, vert_bary: &mut [Barycentric]) {
		let num_tri = self.halfedge.len() / 3;
		for tri in 0..num_tri {
			for i in 0..3 {
				let indices: BaryIndices = self.get_indices((3 * tri + i) as i32);
				if indices.start4 < 0 {
					continue; // skip quad interiors
				}
				let mut uvw = Vector4::repeat(0.0f64);
				uvw[indices.start4 as usize] = 1.0;
				vert_bary[self.halfedge[3 * tri + i].start_vert as usize] =
					Barycentric::new(indices.tri, uvw);
			}
		}
	}

	///Split each edge into n pieces as defined by calling the edgeDivisions
	///function, and sub-triangulate each triangle accordingly. This function
	///doesn't run Finish(), as that is expensive and it'll need to be run after
	///the new vertices have moved, which is a likely scenario after refinement
	///(smoothing).
	pub fn subdivide<F: Fn(Vector3<f64>, Vector4<f64>, Vector4<f64>) -> i32 + Send + Sync>(
		&mut self,
		edge_divisions: F,
		keep_interior: bool,
	) -> Vec<Barycentric> {
		let edges: Vec<TmpEdge> = create_tmp_edges(&self.halfedge);
		let num_vert = self.num_vert();
		let num_edge = edges.len();
		let num_tri = self.num_tri();
		let mut half2edge: Vec<i32> = vec![0; 2 * num_edge];
		for edge in 0..num_edge {
			let idx = edges[edge].halfedge_idx as usize;
			half2edge[idx] = edge as i32;
			half2edge[self.halfedge[idx].paired_halfedge as usize] = edge as i32;
		}

		let face_halfedges: Vec<Vector4<i32>>;
		face_halfedges = (0..num_tri)
			.into_iter()
			.map(|tri| self.get_halfedges(tri as i32))
			.collect();

		let mut edge_added: Vec<i32>;
		edge_added = (0..num_edge)
			.into_iter()
			.map(|i| {
				let edge: TmpEdge = edges[i].clone();
				let h_idx = edge.halfedge_idx;
				if self.is_marked_inside_quad(h_idx) {
					0
				} else {
					let vec: Vector3<f64> =
						self.vert_pos[edge.first as usize] - self.vert_pos[edge.second as usize];
					// let tangent0: Vector4<f64> = if self.halfedge_tangent.empty() {
					// 	Vector4::repeat(0.0)
					// } else {
					// 	self.halfedge_tangent[hIdx as usize]
					// };
					//
					// let tangent1: Vector4<f64> = if self.halfedge_tangent.empty() {
					// 	Vector4::repeat(0.0)
					// } else {
					// 	self.halfedge_tangent[self.halfedge[hIdx as usize].paired_halfedge]
					// };
					edge_divisions(vec, Vector4::repeat(0.0), Vector4::repeat(0.0))
				}
			})
			.collect();

		if keep_interior {
			// Triangles where the greatest number of divisions exceeds the sum of the
			// other two sides will be triangulated as a strip, since if the sub-edges
			// were all equal length it would be degenerate. This leads to poor results
			// with RefineToTolerance, so we avoid this case by adding some extra
			// divisions to the short sides so that the triangulation has some thickness
			// and creates more interior facets.
			let mut tmp: Vec<i32>;
			tmp = (0..num_edge)
				.into_iter()
				.map(|i| {
					let edge: TmpEdge = edges[i].clone();
					let h_idx = edge.halfedge_idx as usize;
					if self.is_marked_inside_quad(h_idx as i32) {
						edge_added[i]
					} else {
						let this_added = edge_added[i];
						let added = |mut h_idx: usize| {
							let mut longest = 0;
							let mut total = 0;
							for _ in 0..3 {
								let added = edge_added[half2edge[h_idx] as usize];
								longest = longest.max(added);
								total += added;
								h_idx = next_halfedge(h_idx as i32) as usize;
								if self.is_marked_inside_quad(h_idx as i32) {
									// No extra on quads
									longest = 0;
									total = 1;
									break;
								}
							}
							let min_extra: i32 = (longest as f64 * 0.2 + 1.0) as i32;
							let extra: i32 =
								(2.0 * longest as f64 + min_extra as f64 - total as f64) as i32;

							if extra > 0 {
								(extra * (longest - this_added)) / longest
							} else {
								0
							}
						};

						edge_added[i]
							+ added(h_idx).max(added(self.halfedge[h_idx].paired_halfedge as usize))
					}
				})
				.collect();
			core::mem::swap(&mut edge_added, &mut tmp);
		}

		let mut edge_offset: Vec<i32> = vec![0; num_edge];
		exclusive_scan_iter(
			edge_added.iter().copied(),
			&mut edge_offset,
			num_vert as i32,
		);

		let mut vert_bary: Vec<Barycentric> = vec![
			Default::default();
			(edge_offset.last().unwrap() + edge_added.last().unwrap())
				as usize
		];
		let total_edge_added = vert_bary.len() - num_vert;
		self.fill_retained_verts(&mut vert_bary);
		for i in 0..num_edge {
			let n = edge_added[i];
			let offset = edge_offset[i];

			let indices: BaryIndices = self.get_indices(edges[i].halfedge_idx);
			if indices.tri < 0 {
				continue; // inside quad
			}
			let frac = 1.0 / (n + 1) as f64;

			for i in 0..n {
				let mut uvw = Vector4::repeat(0.0f64);
				uvw[indices.end4 as usize] = (i + 1) as f64 * frac;
				uvw[indices.start4 as usize] = 1.0 - uvw[indices.end4 as usize];
				vert_bary[(offset + i) as usize].uvw = uvw;
				vert_bary[(offset + i) as usize].tri = indices.tri;
			}
		}

		let sub_tris: Vec<Partition>;
		sub_tris = (0..num_tri)
			.into_iter()
			.map(|tri| {
				let halfedges = face_halfedges[tri];
				let mut divisions = Vector4::repeat(0i32);
				for i in 0..4 {
					if halfedges[i] >= 0 {
						divisions[i] = edge_added[half2edge[halfedges[i] as usize] as usize] + 1;
					}
				}
				Partition::get_partition(divisions)
			})
			.collect();

		let mut tri_offset: Vec<i32> = vec![0; num_tri];
		exclusive_scan_iter(
			sub_tris.iter().map(|part| part.tri_vert.len() as i32),
			&mut tri_offset,
			0,
		);

		let mut interior_offset: Vec<i32> = vec![0; num_tri];
		exclusive_scan_iter(
			sub_tris.iter().map(|part| part.num_interior() as i32),
			&mut interior_offset,
			vert_bary.len() as i32,
		);

		let mut tri_verts: Vec<Vector3<i32>> = vec![
			Vector3::default();
			*tri_offset.last().unwrap() as usize
				+ sub_tris.last().unwrap().tri_vert.len()
		];
		vert_bary.resize(
			(interior_offset.last().unwrap() + sub_tris.last().unwrap().num_interior()) as usize,
			Default::default(),
		);
		let mut tri_ref: Vec<TriRef> = vec![Default::default(); tri_verts.len()];
		for tri in 0..num_tri {
			let halfedges: Vector4<i32> = face_halfedges[tri];
			if halfedges[0] < 0 {
				continue;
			}
			let mut tri3: Vector4<i32> = Default::default();
			let mut edge_offsets: Vector4<i32> = Default::default();
			let mut edge_fwd = Vector4::repeat(false);
			for i in 0..4 {
				if halfedges[i] < 0 {
					tri3[i] = -1;
					continue;
				}
				let halfedge = &self.halfedge[halfedges[i] as usize];
				tri3[i] = halfedge.start_vert;
				edge_offsets[i] = edge_offset[half2edge[halfedges[i] as usize] as usize];
				edge_fwd[i] = halfedge.is_forward();
			}

			let new_tris: Vec<Vector3<i32>> =
				sub_tris[tri].reindex(tri3, edge_offsets, edge_fwd, interior_offset[tri]);

			tri_verts[tri_offset[tri] as usize..tri_offset[tri] as usize + new_tris.len()]
				.copy_from_slice(&new_tris);
			let start = tri_offset[tri] as usize;
			tri_ref[start..start + new_tris.len()].fill(self.mesh_relation.tri_ref[tri]);

			let idx: Vector4<i32> = sub_tris[tri].idx;
			let v_idx: Vector4<i32> = if halfedges[3] >= 0 || idx[1] == next3_i32(idx[0]) {
				idx
			} else {
				Vector4::new(idx[2], idx[0], idx[1], idx[3])
			};
			let mut r_idx: Vector4<i32> = Default::default();
			for i in 0..4 {
				r_idx[v_idx[i] as usize] = i as i32;
			}

			let sub_bary = &sub_tris[tri].vert_bary;
			sub_bary[sub_tris[tri].interior_offset() as usize..]
				.iter()
				.zip(vert_bary[interior_offset[tri] as usize..].iter_mut())
				.for_each(|(bary_in, bary_out)| {
					*bary_out = Barycentric::new(
						tri as i32,
						Vector4::new(
							bary_in[r_idx[0] as usize],
							bary_in[r_idx[1] as usize],
							bary_in[r_idx[2] as usize],
							bary_in[r_idx[3] as usize],
						),
					);
				});
		}
		self.mesh_relation.tri_ref = tri_ref;

		let new_vert_pos: Vec<Vector3<f64>>;
		new_vert_pos = (0..vert_bary.len())
			.into_iter()
			.map(|vert| {
				let bary: Barycentric = vert_bary[vert].clone();
				let halfedges: Vector4<i32> = face_halfedges[bary.tri as usize];
				if halfedges[3] < 0 {
					let mut tri_pos: Matrix3<f64> = Default::default();
					for i in 0..3 {
						tri_pos.set_column(
							i,
							&self.vert_pos
								[self.halfedge[halfedges[i] as usize].start_vert as usize]
								.coords,
						);
					}
					tri_pos * bary.uvw.xyz()
				} else {
					let mut quad_pos: Matrix3x4<f64> = Default::default();
					for i in 0..4 {
						quad_pos.set_column(
							i,
							&self.vert_pos
								[self.halfedge[halfedges[i] as usize].start_vert as usize]
								.coords,
						);
					}
					quad_pos * bary.uvw
				}
			})
			.collect();
		self.vert_pos = new_vert_pos.into_iter().map(|v| Point3::from(v)).collect();

		self.face_normal.clear();

		if self.num_prop > 0 {
			let num_prop_vert: i32 = self.num_prop_vert() as i32;
			let added_verts: i32 = self.num_vert() as i32 - num_vert as i32;
			let prop_offset: i32 = num_prop_vert - num_vert as i32;
			// duplicate the prop verts along all new edges even though this is
			// unnecessary for edges that share the same prop verts. The duplicates will
			// be removed by CompactProps.
			let mut prop: Vec<f64> = vec![
				0.0;
				(self.num_prop * (num_prop_vert + added_verts + total_edge_added as i32))
					as usize
			];

			// copy retained prop verts
			prop[0..self.properties.len()].copy_from_slice(&self.properties);

			// copy interior prop verts and forward edge prop verts
			for i in 0..added_verts {
				let vert: i32 = num_prop_vert + i;
				let bary: Barycentric = vert_bary[num_vert + i as usize].clone();
				let halfedges: Vector4<i32> = face_halfedges[bary.tri as usize];
				let num_prop = self.num_prop() as i32;

				for p in 0..num_prop {
					if halfedges[3] < 0 {
						let mut tri_prop: Vector3<f64> = Default::default();
						for i in 0..3 {
							tri_prop[i] = self.properties[(self.halfedge[3 * bary.tri as usize + i]
								.prop_vert * num_prop + p)
								as usize];
						}
						prop[(vert * num_prop + p) as usize] = tri_prop.dot(&bary.uvw.xyz());
					} else {
						let mut quad_prop: Vector4<f64> = Default::default();
						for i in 0..4 {
							quad_prop[i] = self.properties[(self.halfedge[halfedges[i] as usize]
								.prop_vert * num_prop + p)
								as usize];
						}
						prop[(vert * num_prop + p) as usize] = quad_prop.dot(&bary.uvw);
					}
				}
			}

			// copy backward edge prop verts, some of which will be unreferenced
			// duplicates.
			for i in 0..num_edge {
				let n: i32 = edge_added[i];
				let offset: i32 = edge_offset[i] + prop_offset + added_verts;
				let num_prop: i32 = self.num_prop() as i32;

				let frac: f64 = 1.0 / (n + 1) as f64;
				let halfedge_idx: i32 =
					self.halfedge[edges[i].halfedge_idx as usize].paired_halfedge;
				let prop0: i32 = self.halfedge[halfedge_idx as usize].prop_vert;
				let prop1: i32 = self.halfedge[next_halfedge(halfedge_idx) as usize].prop_vert;
				for i in 0..n {
					for p in 0..num_prop {
						prop[((offset + i) * num_prop + p) as usize] = lerp(
							self.properties[(prop0 * num_prop + p) as usize],
							self.properties[(prop1 * num_prop + p) as usize],
							(i + 1) as f64 * frac,
						);
					}
				}
			}

			let mut tri_prop: Vec<Vector3<i32>> = vec![Default::default(); tri_verts.len()];
			for tri in 0..num_tri {
				let halfedges: Vector4<i32> = face_halfedges[tri];
				if halfedges[0] < 0 {
					continue;
				}

				let mut tri3: Vector4<i32> = Default::default();
				let mut edge_offsets: Vector4<i32> = Default::default();
				let mut edge_fwd = Vector4::repeat(true);
				for i in 0..4 {
					if halfedges[i] < 0 {
						tri3[i] = -1;
						continue;
					}
					let halfedge = &self.halfedge[halfedges[i] as usize];
					tri3[i] = halfedge.prop_vert;
					edge_offsets[i] = edge_offset[half2edge[halfedges[i] as usize] as usize];
					if !halfedge.is_forward() {
						if self.halfedge[halfedge.paired_halfedge as usize].prop_vert
							!= self.halfedge[next_halfedge(halfedges[i]) as usize].prop_vert
							|| self.halfedge[next_halfedge(halfedge.paired_halfedge) as usize]
								.prop_vert != halfedge.prop_vert
						{
							// if the edge doesn't match, point to the backward edge
							// propverts.
							edge_offsets[i] += added_verts;
						} else {
							edge_fwd[i] = false;
						}
					}
				}

				let new_tris: Vec<Vector3<i32>> = sub_tris[tri].reindex(
					tri3,
					edge_offsets + Vector4::repeat(prop_offset),
					edge_fwd,
					interior_offset[tri] + prop_offset,
				);
				tri_prop[tri_offset[tri] as usize..tri_offset[tri] as usize + new_tris.len()]
					.copy_from_slice(&new_tris);
			}

			self.properties = prop;
			self.create_halfedges(tri_prop, tri_verts);
		} else {
			self.create_halfedges(tri_verts, vec![]);
		}

		return vert_bary;
	}
}
