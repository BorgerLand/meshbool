use crate::MeshGLP;
use crate::collider::Collider;
use crate::collider::SimpleRecorder;
use crate::common::{AABB, LossyFrom};
use crate::disjoint_sets::DisjointSets;
use crate::meshboolimpl::MeshBoolImpl;
use crate::parallel::{gather, inclusive_scan, scatter};
use crate::shared::Halfedges;
use crate::utils::{K_PRECISION, permute};
use crate::vec::{vec_resize, vec_resize_nofill, vec_uninit};
use nalgebra::{Point3, Vector3};
use std::mem;
use std::sync::atomic::{AtomicI32, Ordering};

#[cfg(feature = "test")]
use {crate::shared::Halfedge, crate::test::get_intermediate_checks};

const K_NO_CODE: u32 = 0xFFFFFFFF;

fn morton_code(position: Point3<f64>, bbox: AABB) -> u32 {
	// Unreferenced vertices are marked NaN, and this will sort them to the end
	// (the Morton code only uses the first 30 of 32 bits).
	if position.x.is_nan() {
		K_NO_CODE
	} else {
		Collider::morton_code(position, bbox)
	}
}

struct ReindexFace<'a> {
	halfedge: &'a mut Halfedges,
	old_halfedge: &'a Halfedges,
	face_new2old: &'a [i32],
	face_old2new: &'a [i32],
}

impl ReindexFace<'_> {
	fn call(&mut self, new_face: i32) {
		let old_face = self.face_new2old[new_face as usize];
		for i in 0..3 {
			let old_edge = 3 * old_face + i;
			let mut edge = self.old_halfedge.get(old_edge);
			let paired_face = edge.paired_halfedge / 3;
			let offset = edge.paired_halfedge - 3 * paired_face;
			edge.paired_halfedge = 3 * self.face_old2new[paired_face as usize] + offset;
			let new_edge = 3 * new_face + i;
			self.halfedge.set(
				new_edge,
				edge.start_vert,
				edge.paired_halfedge,
				edge.prop_vert,
			);
		}
	}
}

fn merge_mesh_glp<Precision, I>(mesh: &mut MeshGLP<Precision, I>) -> bool
where
	Precision: LossyFrom<f64> + Copy,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	u64: LossyFrom<I>,
	i32: LossyFrom<I>,
	f64: LossyFrom<Precision>,
{
	let mut open_edges: Vec<(i32, i32)> = vec![];

	let mut merge: Vec<i32> = (0..i32::lossy_from(mesh.num_vert())).collect();
	for i in 0..mesh.merge_from_vert.len() {
		merge[usize::lossy_from(mesh.merge_from_vert[i])] = i32::lossy_from(mesh.merge_to_vert[i]);
	}

	let num_vert = usize::lossy_from(mesh.num_vert());
	let num_tri = usize::lossy_from(mesh.num_tri());
	let next: [i32; 3] = [1, 2, 0];
	for tri in 0..num_tri {
		for i in [0, 1, 2] {
			let mut edge = (
				merge[usize::lossy_from(mesh.tri_verts[3 * tri + next[i] as usize])],
				merge[usize::lossy_from(mesh.tri_verts[3 * tri + i])],
			);
			let it = open_edges.iter().position(|p| *p == edge);
			if it.is_none() {
				core::mem::swap(&mut edge.0, &mut edge.1);
				open_edges.push(edge);
			} else {
				open_edges.remove(it.unwrap());
			}
		}
	}
	if open_edges.is_empty() {
		return false;
	}

	let num_open_vert = open_edges.len();
	let mut open_verts: Vec<i32> = vec![0; num_open_vert];
	let mut i = 0;
	for edge in open_edges.iter() {
		let vert: i32 = edge.0;
		open_verts[i] = vert;
		i += 1;
	}

	let vert_prop_d: Vec<Precision> = mesh.vert_properties.clone();
	let mut b_box: AABB = Default::default();
	for i in [0, 1, 2] {
		let min_max = vert_prop_d[i..vert_prop_d.len()]
			.iter()
			.cloned()
			.step_by(usize::lossy_from(mesh.num_prop))
			.map(|f| (f64::lossy_from(f), f64::lossy_from(f)))
			.reduce(|acc, b| (acc.0.min(b.0), acc.1.max(b.1)))
			.unwrap_or((core::f64::INFINITY, core::f64::NEG_INFINITY));
		b_box.min[i] = min_max.0;
		b_box.max[i] = min_max.1;
	}

	// TODO: if Precision == f32
	let tolerance: f64 = f64::lossy_from(mesh.tolerance).max(
		(if true {
			core::f32::EPSILON as f64
		} else {
			K_PRECISION
		}) * b_box.scale(),
	);

	// let mut policy = autoPolicy(numOpenVert, 1e5);
	let mut vert_box: Vec<AABB> = vec![Default::default(); num_open_vert];
	let mut vert_morton: Vec<u32> = vec![0; num_open_vert];

	(0..num_open_vert).for_each(|i| {
		let vert: i32 = open_verts[i];

		let center: Vector3<f64> = Vector3::new(
			f64::lossy_from(mesh.vert_properties[usize::lossy_from(mesh.num_prop) * vert as usize]),
			f64::lossy_from(
				mesh.vert_properties[usize::lossy_from(mesh.num_prop) * vert as usize + 1],
			),
			f64::lossy_from(
				mesh.vert_properties[usize::lossy_from(mesh.num_prop) * vert as usize + 2],
			),
		);

		vert_box[i].min = center.into();
		vert_box[i].min.iter_mut().for_each(|v| {
			*v -= tolerance / 2.0;
		});
		vert_box[i].max = center.into();
		vert_box[i].max.iter_mut().for_each(|v| {
			*v += tolerance / 2.0;
		});

		vert_morton[i] = morton_code(center.into(), b_box);
	});

	let mut vert_new2old: Vec<_> = (0..num_open_vert as i32).into_iter().collect();
	vert_new2old.sort_by_key(|&i| vert_morton[i as usize]);

	permute(&mut vert_morton, &vert_new2old);
	permute(&mut vert_box, &vert_new2old);
	permute(&mut open_verts, &vert_new2old);

	let collider = Collider::new(&vert_box, &vert_morton);
	let uf = DisjointSets::new(num_vert);

	let mut f = |a: i32, b: i32| {
		uf.unite(
			open_verts[a as usize] as usize,
			open_verts[b as usize] as usize,
		);
	};

	let mut recorder = SimpleRecorder::new(&mut f);
	collider.collisions_from_slice::<true, _>(&mut recorder, &vert_box, false);

	for i in 0..mesh.merge_from_vert.len() {
		uf.unite(
			usize::lossy_from(mesh.merge_from_vert[i]),
			usize::lossy_from(mesh.merge_to_vert[i]),
		);
	}

	mesh.merge_to_vert.clear();
	mesh.merge_from_vert.clear();
	for v in 0..num_vert {
		let merge_to = uf.find(v);
		if merge_to != v {
			mesh.merge_from_vert.push(I::lossy_from(v));
			mesh.merge_to_vert.push(I::lossy_from(merge_to));
		}
	}

	return true;
}

impl MeshBoolImpl {
	///Once halfedge_ has been filled in, this function can be called to create the
	///rest of the internal data structures. This function also removes the verts
	///and halfedges flagged for removal (NaN verts and -1 halfedges).
	pub fn sort_geometry(&mut self) {
		if self.halfedge.len() == 0 {
			self.collider = Collider::default();
			return;
		}

		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.

		self.sort_verts();
		let mut face_box: Vec<AABB> = Vec::default();
		let mut face_morton: Vec<u32> = Vec::default();
		self.get_face_box_morton(&mut face_box, &mut face_morton);
		self.sort_faces(&mut face_box, &mut face_morton);
		if self.halfedge.len() == 0 {
			self.collider = Collider::default();
			return;
		}
		self.collider = Collider::new(&face_box, &face_morton);
		self.bbox = self.collider.get_bounding_box();
		self.compact_props();

		debug_assert!(
			self.halfedge.len() % 6 == 0,
			"Not an even number of faces after sorting faces!"
		);

		#[cfg(feature = "test")]
		if get_intermediate_checks() {
			let max_or_minus = |a: i32, b: i32| {
				if a.min(b) < 0 { -1 } else { a.max(b) }
			};
			let mut face = 0;
			let mut extrema = Halfedge::default();
			for i in 0..self.halfedge.len() as i32 {
				let start = if self.halfedge.is_forward(i) {
					self.halfedge.start(i)
				} else {
					self.halfedge.end(i)
				};
				let end = if self.halfedge.is_forward(i) {
					self.halfedge.end(i)
				} else {
					self.halfedge.start(i)
				};
				extrema.start_vert = extrema.start_vert.min(start);
				extrema.end_vert = extrema.end_vert.min(end);
				extrema.paired_halfedge =
					max_or_minus(extrema.paired_halfedge, self.halfedge.pair(i));
				face = max_or_minus(face, i / 3);
			}
			debug_assert!(extrema.start_vert >= 0, "Vertex index is negative!");
			debug_assert!(
				extrema.end_vert < self.num_vert() as i32,
				"Vertex index exceeds number of verts!"
			);
			debug_assert!(extrema.paired_halfedge >= 0, "Halfedge index is negative!");
			debug_assert!(
				extrema.paired_halfedge < 2 * self.num_edge() as i32,
				"Halfedge index exceeds number of halfedges!"
			);
			debug_assert!(face >= 0, "Face index is negative!");
			debug_assert!(
				face < self.num_tri() as i32,
				"Face index exceeds number of faces!"
			);
		}

		debug_assert!(
			self.mesh_relation.tri_ref.len() == self.num_tri()
				|| self.mesh_relation.tri_ref.len() == 0,
			"Mesh Relation doesn't fit!"
		);

		debug_assert!(self.is_2_manifold(), "mesh is not 2-manifold!");
	}

	///Sorts the vertices according to their Morton code.
	fn sort_verts(&mut self) {
		let num_vert = self.num_vert();
		let mut vert_morton: Vec<u32> = unsafe { vec_uninit(num_vert) };
		for vert in 0..num_vert {
			vert_morton[vert] = morton_code(self.vert_pos[vert], self.bbox);
		}

		let mut vert_new2old: Vec<_> = (0..num_vert as i32).collect();
		vert_new2old.sort_by_key(|&i| vert_morton[i as usize]);

		self.reindex_verts(&vert_new2old, num_vert);

		// Verts were flagged for removal with NaNs and assigned kNoCode to sort
		// them to the end, which allows them to be removed.
		let new_num_vert =
			vert_new2old.partition_point(|&vert| vert_morton[vert as usize] < K_NO_CODE);

		vec_resize(&mut vert_new2old, new_num_vert, 0);
		permute(&mut self.vert_pos, &vert_new2old);

		if self.vert_normal.len() == num_vert {
			permute(&mut self.vert_normal, &vert_new2old);
		}
	}

	///Updates the halfedges to point to new vert indices based on a mapping,
	///vertNew2Old. This may be a subset, so the total number of original verts is
	///also given.
	pub fn reindex_verts(&mut self, vert_new2old: &[i32], old_num_vert: usize) {
		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.
		let mut vert_old2new: Vec<i32> = unsafe { vec_uninit(old_num_vert) };
		scatter(0..self.num_vert() as i32, vert_new2old, &mut vert_old2new);
		let has_prop = self.num_prop() > 0;
		for idx in 0..self.halfedge.len() as i32 {
			let start_vert = self.halfedge.start(idx);
			if start_vert < 0 {
				continue;
			}
			let new_start = vert_old2new[start_vert as usize];
			self.halfedge.set_start(idx, new_start);
			if !has_prop {
				self.halfedge.set_prop(idx, new_start);
			}
		}
	}

	fn compact_props(&mut self) {
		if self.num_prop == 0 {
			return;
		}
		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.

		let num_prop = self.num_prop();
		let num_verts = self.properties.len() / num_prop;
		let mut keep = vec![0; num_verts];

		for idx in 0..self.halfedge.len() {
			let keep: &[AtomicI32] = unsafe { std::mem::transmute(keep.as_mut_slice()) };
			keep[self.halfedge.prop(idx as i32) as usize].store(1, Ordering::Relaxed);
		}

		let mut prop_old2new = vec![0_i32; num_verts + 1];
		inclusive_scan(keep.iter().cloned(), &mut prop_old2new[1..]);

		let old_prop = self.properties.clone();
		let num_verts_new = prop_old2new[num_verts];
		unsafe {
			vec_resize_nofill(&mut self.properties, num_prop * (num_verts_new as usize));
		}
		for old_idx in 0..num_verts {
			if keep[old_idx] == 0 {
				continue;
			}
			for p in 0..num_prop {
				self.properties[prop_old2new[old_idx] as usize * num_prop + p] =
					old_prop[old_idx * num_prop + p];
			}
		}

		for idx in 0..self.halfedge.len() as i32 {
			self.halfedge
				.set_prop(idx, prop_old2new[self.halfedge.prop(idx) as usize]);
		}
	}

	///Fills the faceBox and faceMorton input with the bounding boxes and Morton
	///codes of the faces, respectively. The Morton code is based on the center of
	///the bounding box.
	pub fn get_face_box_morton(&self, face_box: &mut Vec<AABB>, face_morton: &mut Vec<u32>) {
		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.
		// faceBox should be initialized
		vec_resize(face_box, self.num_tri(), AABB::default());
		unsafe {
			vec_resize_nofill(face_morton, self.num_tri());
		}
		for face in 0..self.num_tri() {
			// Removed tris are marked by all halfedges having pairedHalfedge
			// = -1, and this will sort them to the end (the Morton code only
			// uses the first 30 of 32 bits).
			if self.halfedge.pair((3 * face) as i32) < 0 {
				face_morton[face] = K_NO_CODE;
				continue;
			}

			let mut center = Point3::<f64>::new(0.0, 0.0, 0.0);

			for i in 0..3 {
				let pos = self.vert_pos[self.halfedge.start((3 * face + i) as i32) as usize];
				center += pos.coords;
				face_box[face].union_point(pos);
			}

			center /= 3.;

			face_morton[face] = morton_code(center, self.bbox);
		}
	}

	///Sorts the faces of this manifold according to their input Morton code. The
	///bounding box and Morton code arrays are also sorted accordingly.
	fn sort_faces(&mut self, face_box: &mut Vec<AABB>, face_morton: &mut Vec<u32>) {
		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.
		let mut face_new2old: Vec<_> = (0..self.num_tri() as i32).collect();
		face_new2old.sort_by_key(|&i| face_morton[i as usize]);

		// Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
		// to sort them to the end, which allows them to be removed.
		let new_num_tri =
			face_new2old.partition_point(|&face| face_morton[face as usize] < K_NO_CODE);

		vec_resize(&mut face_new2old, new_num_tri, 0);

		permute(face_morton, &face_new2old);
		permute(face_box, &face_new2old);
		self.gather_faces(&face_new2old);
	}

	///Creates the halfedge_ vector for this manifold by copying a set of faces from
	///another manifold, given by oldHalfedge. Input faceNew2Old defines the old
	///faces to gather into this.
	fn gather_faces(&mut self, face_new2old: &[i32]) {
		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.
		let num_tri = face_new2old.len();
		if self.mesh_relation.tri_ref.len() == self.num_tri() {
			permute(&mut self.mesh_relation.tri_ref, face_new2old);
		}

		if self.face_normal.len() == self.num_tri() {
			permute(&mut self.face_normal, face_new2old);
		}

		let old_halfedge = mem::take(&mut self.halfedge);
		unsafe { self.halfedge.resize_nofill(3 * num_tri) };
		let mut face_old2new = unsafe { vec_uninit(old_halfedge.len() / 3) };
		scatter(0..num_tri as i32, face_new2old, &mut face_old2new);

		let mut reindex_face = ReindexFace {
			halfedge: &mut self.halfedge,
			old_halfedge: &old_halfedge,
			face_new2old,
			face_old2new: &face_old2new,
		};
		for new_face in 0..num_tri {
			reindex_face.call(new_face as i32);
		}
	}

	pub fn gather_faces_from_old(&mut self, old: &MeshBoolImpl, face_new2old: &[i32]) {
		// Invariant: every ctx-passing parallel op is followed by IsCancelled to
		// keep partial output from feeding unconditional downstream consumers.
		let num_tri = face_new2old.len();

		unsafe {
			vec_resize_nofill(&mut self.mesh_relation.tri_ref, num_tri);
		}
		gather(
			face_new2old,
			&old.mesh_relation.tri_ref,
			&mut self.mesh_relation.tri_ref,
		);

		self.mesh_relation
			.mesh_id_transform
			.extend(&old.mesh_relation.mesh_id_transform);

		if old.num_prop() > 0 {
			self.num_prop = old.num_prop;
			self.properties = old.properties.clone();
		}

		if old.face_normal.len() == old.num_tri() {
			unsafe {
				vec_resize_nofill(&mut self.face_normal, num_tri);
			}
			gather(face_new2old, &old.face_normal, &mut self.face_normal);
		}

		let mut face_old2new = unsafe { vec_uninit(old.num_tri()) };
		scatter(0..num_tri as i32, face_new2old, &mut face_old2new);

		unsafe {
			self.halfedge.resize_nofill(3 * num_tri);
		}
		let mut reindex_face = ReindexFace {
			halfedge: &mut self.halfedge,
			// halfedge_tangent: &mut self.halfedge_tangent,
			old_halfedge: &old.halfedge,
			// old_halfedge_tangent: &old_halfedge_tangent,
			face_new2old: &face_new2old,
			face_old2new: &face_old2new,
		};
		for new_face in 0..num_tri {
			reindex_face.call(new_face as i32);
		}
	}

	pub fn reorder_halfedges(&mut self) {
		// halfedges in the same face are added in non-deterministic order, so we have
		// to reorder them for determinism

		// step 1: reorder within the same face, such that the halfedge with the
		// smallest starting vertex is placed first
		for tri in 0..(self.halfedge.len() / 3) {
			let face = [
				self.halfedge.get((tri * 3) as i32),
				self.halfedge.get((tri * 3 + 1) as i32),
				self.halfedge.get((tri * 3 + 2) as i32),
			];
			if face[0].start_vert < 0 {
				continue;
			}
			let mut index = 0;
			for i in 1..3 {
				if face[i].start_vert < face[index].start_vert {
					index = i;
				};
			}
			for i in 0..3 {
				let f = face[(index + i) % 3];
				self.halfedge.set(
					(tri * 3 + i) as i32,
					f.start_vert,
					f.paired_halfedge,
					f.prop_vert,
				);
			}
		}
		// step 2: fix paired halfedge
		'outer: for tri in 0..self.halfedge.len() / 3 {
			for i in 0..3 {
				let curr_idx = (tri * 3 + i) as i32;
				let start_vert = self.halfedge.start(curr_idx);
				if start_vert < 0 {
					continue 'outer;
				}
				let opposite_face = self.halfedge.pair(curr_idx) / 3;
				let mut index = -1;
				for j in 0..3 {
					if start_vert == self.halfedge.end(opposite_face * 3 + j) {
						index = j;
					}
				}

				self.halfedge.set_pair(curr_idx, opposite_face * 3 + index);
			}
		}
	}
}

///Updates the mergeFromVert and mergeToVert vectors in order to create a
///manifold solid. If the MeshGL is already manifold, no change will occur and
///the function will return false. Otherwise, this will merge verts along open
///edges within tolerance (the maximum of the MeshGL tolerance and the
///baseline bounding-box tolerance), keeping any from the existing merge
///vectors, and return true.
///
///There is no guarantee the result will be manifold - this is a best-effort
///helper function designed primarily to aid in the case where a manifold
///multi-material MeshGL was produced, but its merge vectors were lost due to
///a round-trip through a file format. Constructing a Manifold from the result
///will report an error status if it is not manifold.
impl<F, I> MeshGLP<F, I>
where
	F: LossyFrom<f64> + Copy,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	u64: LossyFrom<I>,
	u32: LossyFrom<I>,
	i32: LossyFrom<I>,
	f64: LossyFrom<F>,
{
	pub fn merge(&mut self) -> bool {
		merge_mesh_glp(self)
	}
}
