use crate::MeshBoolError;
use crate::collider::Collider;
use crate::common::{AABB, LossyFrom};
use crate::disjoint_sets::DisjointSets;
use crate::mesh::MeshGLP;
use crate::mesh_fixes::{FlipTris, transform_normal};
use crate::parallel::exclusive_scan_in_place;
use crate::shared::{
	Halfedges, TriRef, inverse_normal_transform, max_epsilon, next_halfedge, normal_transform,
	safe_normalize,
};
use crate::utils::{atomic_add_i32, mat3, mat4, next3_i32};
use crate::vec::{vec_resize, vec_resize_nofill, vec_uninit};
use nalgebra::{Matrix3, Matrix3x4, Point3, Vector2, Vector3, Vector4};
use std::any::TypeId;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering as AtomicOrdering};
use std::{array, f64, mem};

#[derive(Clone)]
pub struct BaryIndices {
	pub tri: i32,
	pub start4: i32,
	pub end4: i32,
}

impl BaryIndices {
	pub fn new(tri: i32, start4: i32, end4: i32) -> Self {
		Self { tri, start4, end4 }
	}
}

#[derive(Copy, Clone)]
#[allow(unused)]
pub enum Shape {
	Tetrahedron,
	Cube,
	Octahedron,
}

pub static MESH_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

///@brief This library's internal representation of an oriented, 2-manifold,
///triangle mesh - a simple boundary-representation of a solid object. Use this
///class to store and operate on solids, and use MeshGL for input and output.
///
///In addition to storing geometric data, a Manifold can also store an arbitrary
///number of vertex properties. These could be anything, e.g. normals, UV
///coordinates, colors, etc, but this library is completely agnostic. All
///properties are merely float values indexed by channel number. It is up to the
///user to associate channel numbers with meaning.
///
///Manifold allows vertex properties to be shared for efficient storage, or to
///have multiple property verts associated with a single geometric vertex,
///allowing sudden property changes, e.g. at Boolean intersections, without
///sacrificing manifoldness.
///
///Manifolds also keep track of their relationships to their inputs, via
///OriginalIDs and the faceIDs and transforms accessible through MeshGL. This
///allows object-level properties to be re-associated with the output after many
///operations, particularly useful for materials. Since separate object's
///properties are not mixed, there is no requirement that channels have
///consistent meaning between different inputs.
#[derive(Clone, Debug)]
pub struct MeshBoolImpl {
	pub bbox: AABB,
	pub epsilon: f64,
	pub tolerance: f64,
	pub num_prop: i32,
	pub status: MeshBoolError,
	pub vert_pos: Vec<Point3<f64>>,
	pub halfedge: Halfedges,
	pub properties: Vec<f64>,
	// Note that vertNormal_ is not precise due to the use of an approximated acos
	// function
	pub vert_normal: Vec<Vector3<f64>>,
	pub face_normal: Vec<Vector3<f64>>,
	pub mesh_relation: MeshRelationD,
	pub collider: Collider,
}

#[derive(Clone, Debug)]
pub struct MeshRelationD {
	/// The originalID of this Manifold if it is an original; -1 otherwise.
	pub original_id: i32,
	pub mesh_id_transform: BTreeMap<i32, Relation>,
	pub tri_ref: Vec<TriRef>,
}

impl Default for MeshRelationD {
	fn default() -> Self {
		MeshRelationD {
			original_id: -1,
			mesh_id_transform: BTreeMap::default(),
			tri_ref: Vec::default(),
		}
	}
}

#[derive(Clone, Copy, Debug)]
pub struct Relation {
	pub original_id: i32,
	pub transform: Matrix3x4<f64>,
	pub back_side: bool,
	///True when this meshID's contribution to properties_ slots 0..2 holds
	///world-frame vertex normals (set by CalculateNormals at slot 0). Carries
	///through Transforms and Booleans. Exported as runFlags bit 1.
	pub has_normals: bool,
}

impl Default for Relation {
	fn default() -> Self {
		Self {
			original_id: -1,
			transform: Matrix3x4::identity(),
			back_side: false,
			has_normals: false,
		}
	}
}

impl Relation {
	pub fn get_inverse_normal_transform(&self) -> Matrix3<f64> {
		inverse_normal_transform(&self.transform) * if self.back_side { -1.0 } else { 1.0 }
	}
}

impl MeshBoolImpl {
	pub fn is_empty(&self) -> bool {
		self.num_tri() == 0
	}

	pub fn num_vert(&self) -> usize {
		self.vert_pos.len()
	}

	pub fn num_edge(&self) -> usize {
		self.halfedge.len() / 2
	}

	pub fn num_tri(&self) -> usize {
		self.halfedge.len() / 3
	}

	pub fn num_prop(&self) -> usize {
		self.num_prop as usize
	}

	pub fn num_prop_vert(&self) -> usize {
		if self.num_prop() == 0 {
			self.num_vert()
		} else {
			self.properties.len() / self.num_prop()
		}
	}
}

#[derive(Copy, Clone)]
struct CreateHalfedge {
	start_vert: i32,
	end_vert: i32,
	prop_vert: i32,
}

#[derive(Clone, Default)]
struct HalfedgePairData {
	large_vert: i32,
	tri: i32,
	edge_index: i32,
}

struct PrepHalfedges<'a, const USE_PROP: bool, F: FnMut(i32, i32, i32)> {
	halfedges: &'a mut Vec<CreateHalfedge>,
	tri_prop: &'a Vec<Vector3<i32>>,
	tri_vert: &'a Vec<Vector3<i32>>,
	f: &'a mut F,
}

impl<'a, const USE_PROP: bool, F: FnMut(i32, i32, i32)> PrepHalfedges<'a, USE_PROP, F> {
	fn call(&mut self, tri: i32) {
		let props = self.tri_prop[tri as usize];
		for i in 0..3 {
			let j = next3_i32(i);
			let e = 3 * tri + i;
			let v0 = if USE_PROP {
				props[i as usize]
			} else {
				self.tri_vert[tri as usize][i as usize]
			};
			let v1 = if USE_PROP {
				props[j as usize]
			} else {
				self.tri_vert[tri as usize][j as usize]
			};
			debug_assert!(v0 != v1, "topological degeneracy");
			self.halfedges[e as usize] = CreateHalfedge {
				start_vert: v0,
				end_vert: v1,
				prop_vert: props[i as usize],
			};

			(self.f)(e, v0, v1);
		}
	}
}

impl MeshBoolImpl {
	///True only when every meshID carries normals at slot 0..2 - the
	///condition under which GetMeshGL(-1) can safely auto-substitute that
	///slot. A mixed Boolean output (some meshIDs with normals, some
	///without) returns false; the output MeshGL's per-run bit 1 still
	///marks the with-normals runs individually.
	pub fn all_have_normals(&self) -> bool {
		if self.mesh_relation.mesh_id_transform.is_empty() {
			return false;
		}
		for m in self.mesh_relation.mesh_id_transform.values() {
			if !m.has_normals {
				return false;
			}
		}

		true
	}

	///True iff the meshID owning `tri` has hasNormals set. Returns false when
	///the meshID isn't in meshRelation_.meshIDtransform (treat as no-normals).
	pub fn tri_has_normals(mesh_relation: &MeshRelationD, tri: i32) -> bool {
		let mesh_id = mesh_relation.tri_ref[tri as usize].mesh_id;
		let it = mesh_relation.mesh_id_transform.get(&mesh_id);
		it.map(|it| it.has_normals).unwrap_or(false)
	}

	pub fn from_meshgl<F, I>(mesh_gl: &MeshGLP<F, I>) -> Self
	where
		F: LossyFrom<f64> + Copy,
		f64: From<F>,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
	{
		let num_vert = usize::lossy_from(mesh_gl.num_vert());
		let num_tri = usize::lossy_from(mesh_gl.num_tri());

		let mut manifold = Self::default();

		if num_vert == 0 && num_tri == 0 {
			manifold.make_empty(MeshBoolError::NoError);
			return manifold;
		}

		if num_vert < 4 || num_tri < 4 {
			manifold.make_empty(MeshBoolError::NotManifold);
			return manifold;
		}

		if usize::lossy_from(mesh_gl.num_prop) < 3 {
			manifold.make_empty(MeshBoolError::MissingPositionProperties);
			return manifold;
		}

		if mesh_gl.merge_from_vert.len() != mesh_gl.merge_to_vert.len() {
			manifold.make_empty(MeshBoolError::MergeVectorsDifferentLengths);
			return manifold;
		}

		if !mesh_gl.run_transform.is_empty()
			&& 12 * mesh_gl.run_original_id.len() != mesh_gl.run_transform.len()
		{
			manifold.make_empty(MeshBoolError::TransformWrongLength);
			return manifold;
		}

		if !mesh_gl.run_original_id.is_empty()
			&& !mesh_gl.run_index.is_empty()
			&& mesh_gl.run_original_id.len() + 1 != mesh_gl.run_index.len()
			&& mesh_gl.run_original_id.len() != mesh_gl.run_index.len()
		{
			manifold.make_empty(MeshBoolError::RunIndexWrongLength);
			return manifold;
		}

		if !mesh_gl.face_id.is_empty() && mesh_gl.face_id.len() != num_tri {
			manifold.make_empty(MeshBoolError::FaceIDWrongLength);
			return manifold;
		}

		if mesh_gl
			.vert_properties
			.iter()
			.any(|v| !f64::from(*v).is_finite())
		{
			manifold.make_empty(MeshBoolError::NonFiniteVertex);
			return manifold;
		}

		if mesh_gl
			.run_transform
			.iter()
			.any(|x| !f64::from(*x).is_finite())
		{
			manifold.make_empty(MeshBoolError::InvalidConstruction);
			return manifold;
		}

		// if (!manifold::all_of(meshGL.halfedgeTangent.begin(),
		//                       meshGL.halfedgeTangent.end(),
		//                       [](Precision x) { return std::isfinite(x); })) {
		//   MakeEmpty(Error::InvalidConstruction);
		//   return;
		// }

		let mut prop2vert: Vec<i32>;
		if !mesh_gl.merge_from_vert.is_empty() {
			prop2vert = (0..num_vert as i32).collect();
			for i in 0..mesh_gl.merge_from_vert.len() {
				let from = usize::lossy_from(mesh_gl.merge_from_vert[i]);
				let to = usize::lossy_from(mesh_gl.merge_to_vert[i]);
				if from >= num_vert || to >= num_vert {
					manifold.make_empty(MeshBoolError::MergeIndexOutOfBounds);
					return manifold;
				}
				prop2vert[from] = to as i32;
			}
		} else {
			prop2vert = vec![];
		}

		let num_prop = usize::lossy_from(mesh_gl.num_prop) - 3;
		manifold.num_prop = num_prop as i32;
		unsafe { vec_resize_nofill(&mut manifold.properties, num_vert * num_prop) };
		manifold.tolerance = f64::from(mesh_gl.tolerance);
		// This will have unreferenced duplicate positions that will be removed by
		// Impl::remove_unreferenced_verts().
		unsafe { vec_resize_nofill(&mut manifold.vert_pos, num_vert) };

		for i in 0..num_vert {
			for j in [0, 1, 2] {
				manifold.vert_pos[i][j] =
					mesh_gl.vert_properties[usize::lossy_from(mesh_gl.num_prop) * i + j].into();
			}
			for j in 0..num_prop {
				manifold.properties[i * num_prop + j] =
					mesh_gl.vert_properties[usize::lossy_from(mesh_gl.num_prop) * i + 3 + j].into();
			}
		}

		// halfedgeTangent_.resize_nofill(meshGL.halfedgeTangent.len() / 4);
		// for i in 0..halfedgeTangent_.len() {
		//   for j in [0, 1, 2, 3] {
		//     halfedgeTangent_[i][j] = meshGL.halfedgeTangent[4 * i + j];
		//   }
		// }

		let mut tri_ref: Vec<TriRef> = unsafe { vec_uninit(num_tri) };

		let mut run_index: Vec<usize> = mesh_gl
			.run_index
			.iter()
			.map(|x| usize::lossy_from(*x))
			.collect();
		let run_end = mesh_gl.tri_verts.len();
		if run_index.is_empty() {
			run_index = vec![0, run_end];
		} else if run_index.len() == mesh_gl.run_original_id.len() {
			run_index.push(run_end);
		} else if run_index.len() == 1 {
			run_index.push(run_end);
		}

		let start_id = MeshBoolImpl::reserve_ids(1.max(mesh_gl.run_original_id.len()));
		let mut run_original_id = mesh_gl.run_original_id.clone();
		if run_original_id.is_empty() {
			run_original_id.push(start_id as u32);
		}
		for i in 0..run_original_id.len() {
			let mesh_id = start_id + i;
			let original_id = run_original_id[i] as i32;
			let backside = mesh_gl.backside(i);
			// Per-run hasNormals (runFlags bit 1). Defensively require numProp >= 3
			// so a caller setting the bit on a too-small MeshGL doesn't make us read
			// past the property bounds.
			let run_has_n = mesh_gl.has_normals(i) && num_prop >= 3;
			for tri in (run_index[i] / 3)..(run_index[i + 1] / 3) {
				let r = &mut tri_ref[tri as usize];
				r.mesh_id = mesh_id as i32;
				r.original_id = original_id;
				r.face_id = if mesh_gl.face_id.is_empty() {
					-1
				} else {
					usize::lossy_from(mesh_gl.face_id[tri]) as i32
				};
				r.coplanar_id = tri as i32;
			}

			if mesh_gl.run_transform.is_empty() {
				manifold
					.mesh_relation
					.mesh_id_transform
					.entry(mesh_id as i32)
					.or_insert_with(|| Relation {
						original_id,
						transform: Matrix3x4::identity(),
						back_side: backside,
						has_normals: run_has_n,
					});
			} else {
				let m: [_; 12] = array::from_fn(|j| f64::from(mesh_gl.run_transform[i * 12 + j]));
				manifold
					.mesh_relation
					.mesh_id_transform
					.entry(mesh_id as i32)
					.or_insert_with(|| Relation {
						original_id: original_id,
						transform: [
							[m[0], m[1], m[2]],
							[m[3], m[4], m[5]],
							[m[6], m[7], m[8]],
							[m[9], m[10], m[11]],
						]
						.into(),
						back_side: backside,
						has_normals: run_has_n,
					});
			}
		}

		let mut tri_prop: Vec<Vector3<i32>> = Vec::with_capacity(num_tri);
		let mut tri_vert: Vec<Vector3<i32>> = vec![];
		let needs_prop_map = num_prop > 0 && !prop2vert.is_empty();
		if needs_prop_map {
			tri_vert.reserve(num_tri)
		}
		if tri_ref.len() > 0 {
			manifold.mesh_relation.tri_ref.reserve(num_tri);
		}
		for i in 0..num_tri {
			let mut tri_p: Vector3<i32> = Vector3::default();
			let mut tri_v: Vector3<i32> = Vector3::default();
			for j in [0, 1, 2] {
				let vert = usize::lossy_from(mesh_gl.tri_verts[3 * i + j]);
				if vert >= num_vert {
					manifold.make_empty(MeshBoolError::VertexOutOfBounds);
					return manifold;
				}
				tri_p[j] = vert as i32;
				tri_v[j] = if prop2vert.is_empty() {
					vert as i32
				} else {
					prop2vert[vert as usize]
				};
			}
			if tri_v[0] != tri_v[1] && tri_v[1] != tri_v[2] && tri_v[2] != tri_v[0] {
				if needs_prop_map {
					tri_prop.push(tri_p);
					tri_vert.push(tri_v);
				} else {
					tri_prop.push(tri_v);
				}
				if tri_ref.len() > 0 {
					manifold.mesh_relation.tri_ref.push(tri_ref[i]);
				}
			}
		}

		manifold.create_halfedges(tri_prop, tri_vert);
		if !manifold.is_manifold() {
			manifold.make_empty(MeshBoolError::NotManifold);
			return manifold;
		}

		manifold.calculate_bbox();
		manifold.set_epsilon(-1.0f64, false); // TODO: if Precision == float

		// we need to split pinched verts before calculating vertex normals, because
		// the algorithm doesn't work with pinched verts
		manifold.cleanup_topology();
		manifold.dedupe_prop_verts();
		manifold.set_normals_and_coplanar();
		manifold.remove_degenerates(None);
		manifold.remove_unreferenced_verts();
		manifold.sort_geometry();

		if !manifold.is_finite() {
			manifold.make_empty(MeshBoolError::NonFiniteVertex);
			return manifold;
		}

		// A Manifold created from an input mesh is never an original - the input is
		// the original.
		manifold.mesh_relation.original_id = -1;

		manifold
	}

	#[inline]
	pub fn get_mesh_gl_impl<Precision, I>(&self, normal_idx: i32) -> MeshGLP<Precision, I>
	where
		Precision: LossyFrom<f64> + Copy + 'static,
		f64: From<Precision>,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
	{
		let num_prop = self.num_prop();
		let num_vert = self.num_prop_vert();
		let num_tri = self.num_tri();

		let is_original = self.mesh_relation.original_id >= 0;
		let update_normals = !is_original && normal_idx >= 0;

		let out_num_prop = 3 + num_prop;
		let mut tolerance = self.tolerance;
		if TypeId::of::<Precision>() == TypeId::of::<f32>() {
			tolerance = tolerance.max((f32::EPSILON as f64) * self.bbox.scale());
		}
		let mut tri_verts: Vec<I> = vec![I::lossy_from(0); 3 * num_tri];

		// Sort the triangles into runs
		let mut face_id: Vec<I> = vec![I::lossy_from(0); num_tri];
		let mut tri_new2old: Vec<_> = (0..num_tri).map(|i| i as i32).collect();
		let tri_ref = &self.mesh_relation.tri_ref;
		// Don't sort originals - keep them in order
		if !is_original {
			tri_new2old
				.sort_by_key(|&i| (tri_ref[i as usize].original_id, tri_ref[i as usize].mesh_id));
		}

		let mut run_index: Vec<I> = Vec::new();
		let mut run_original_id: Vec<u32> = Vec::new();
		let mut run_transform: Vec<Precision> = Vec::new();
		let mut run_flags: Vec<u8> = Vec::new();

		// runFlags layout: bit 0 = backSide, bit 1 = hasNormals (slot 0..2 of the
		// extra properties is world-frame vertex normals; consumers should skip
		// re-applying runTransform to those channels).
		let mut add_run = |tri, rel: Relation| {
			run_index.push(I::lossy_from(3 * tri));
			run_original_id.push(rel.original_id as u32);
			// runFlags carries hasNormals (bit 1) which we want on originals too;
			// runTransform is just metadata so skip it for originals where it would
			// always be identity.
			let flags = (rel.back_side as u8) | ((rel.has_normals as u8) << 1);
			run_flags.push(flags);
			if !is_original {
				for col in 0..4 {
					for row in 0..3 {
						run_transform.push(Precision::lossy_from(rel.transform[(row, col)]))
					}
				}
			}
		};

		let mut mesh_id_transform = self.mesh_relation.mesh_id_transform.clone();
		let mut last_id = -1;
		for tri in 0..num_tri {
			let old_tri = tri_new2old[tri];
			let tri_ref = tri_ref[old_tri as usize];
			let mesh_id = tri_ref.mesh_id;

			face_id[tri] = I::lossy_from(if tri_ref.face_id >= 0 {
				tri_ref.face_id as usize
			} else {
				tri_ref.coplanar_id as usize
			});
			for i in 0..3 {
				tri_verts[3 * tri + (i as usize)] =
					I::lossy_from(self.halfedge.start(3 * old_tri + i) as usize);
			}

			if mesh_id != last_id {
				let it = mesh_id_transform.remove(&mesh_id);
				let rel = it.unwrap_or_default();
				add_run(tri, rel);
				last_id = mesh_id;
			}
		}

		// Add runs for originals that did not contribute any faces to the output
		for pair in mesh_id_transform {
			add_run(num_tri, pair.1);
		}

		run_index.push(I::lossy_from(3 * num_tri));

		// Early return for no props
		if num_prop == 0 {
			let mut vert_properties: Vec<Precision> =
				vec![Precision::lossy_from(0.0); 3 * num_vert];
			for i in 0..num_vert {
				let v = self.vert_pos[i];
				vert_properties[3 * i] = Precision::lossy_from(v.x);
				vert_properties[3 * i + 1] = Precision::lossy_from(v.y);
				vert_properties[3 * i + 2] = Precision::lossy_from(v.z);
			}

			return MeshGLP {
				num_prop: I::lossy_from(out_num_prop),
				vert_properties,
				tri_verts,
				merge_from_vert: Vec::default(),
				merge_to_vert: Vec::default(),
				run_index,
				run_original_id,
				run_transform,
				run_flags,
				face_id,
				tolerance: Precision::lossy_from(tolerance),
			};
		}

		// Duplicate verts with different props
		let mut vert2idx: Vec<i32> = vec![-1; self.num_vert()];
		let mut vert_prop_pair: Vec<Vec<Vector2<i32>>> = vec![Vec::new(); self.num_vert()];
		let mut vert_properties: Vec<Precision> = Vec::with_capacity(num_vert * out_num_prop);

		let mut merge_from_vert: Vec<I> = Vec::new();
		let mut merge_to_vert: Vec<I> = Vec::new();

		for run in 0..run_original_id.len() {
			for tri in
				(usize::lossy_from(run_index[run]) / 3)..(usize::lossy_from(run_index[run + 1]) / 3)
			{
				for i in 0..3 {
					let prop = self.halfedge.prop(3 * tri_new2old[tri] + (i as i32));
					let vert = usize::lossy_from(tri_verts[3 * tri + i]);

					let bin = &mut vert_prop_pair[vert];
					let mut b_found = false;
					for b in bin.iter() {
						if b.x == prop {
							b_found = true;
							tri_verts[3 * tri + i] = I::lossy_from(b.y as usize);
							break;
						}
					}

					if b_found {
						continue;
					}
					let idx = vert_properties.len() / out_num_prop;
					tri_verts[3 * tri + i] = I::lossy_from(idx);
					bin.push(Vector2::new(prop, idx as i32));

					for p in 0..3 {
						vert_properties.push(Precision::lossy_from(self.vert_pos[vert][p]));
					}
					for p in 0..num_prop {
						vert_properties.push(Precision::lossy_from(
							self.properties[(prop as usize) * num_prop + p],
						));
					}

					// Normalize the requested normal slot. For runs that already carry
					// world-frame normals (hasNormals bit), just normalize; for legacy
					// callers asking to interpret a slot as normals on a run without
					// hasNormals, apply the per-run inverse-frame transform first.
					// TODO: collapse the !runHasN branch into a no-op once the explicit-
					// normalIdx parameter on GetMeshGL is removed and `updateNormals`
					// becomes implied by the hasNormals bit.
					if update_normals {
						let mut normal = Vector3::<f64>::default();
						let start = vert_properties.len() - out_num_prop;
						for i in 0..3 {
							normal[i] = f64::from(
								vert_properties[((start + 3 + i) as i32 + normal_idx) as usize],
							);
						}
						let run_has_n = !is_original && (run_flags[run] & 2) != 0;
						if !is_original && !run_has_n {
							let m: [_; 12] =
								array::from_fn(|j| f64::from(run_transform[run * 12 + j]));
							let t = Matrix3x4::from([
								[m[0], m[1], m[2]],
								[m[3], m[4], m[5]],
								[m[6], m[7], m[8]],
								[m[9], m[10], m[11]],
							]);
							normal = normal_transform(&t)
								* (if (run_flags[run] & 1) != 0 { -1.0 } else { 1.0 })
								* normal;
						}
						normal = safe_normalize(normal);
						for i in 0..3 {
							vert_properties[((start + 3 + i) as i32 + normal_idx) as usize] =
								Precision::lossy_from(normal[i]);
						}
					}

					if vert2idx[vert] == -1 {
						vert2idx[vert] = idx as i32;
					} else {
						merge_from_vert.push(I::lossy_from(idx));
						merge_to_vert.push(I::lossy_from(vert2idx[vert] as usize));
					}
				}
			}
		}

		MeshGLP {
			num_prop: I::lossy_from(out_num_prop),
			vert_properties,
			tri_verts,
			merge_from_vert,
			merge_to_vert,
			run_index,
			run_original_id,
			run_transform,
			run_flags,
			face_id,
			tolerance: Precision::lossy_from(tolerance),
		}
	}

	pub fn from_shape(shape: Shape, m: Matrix3x4<f64>) -> Self {
		let (mut vert_pos, tri_verts) = match shape {
			Shape::Tetrahedron => (
				vec![
					Point3::<f64>::new(-1.0, -1.0, 1.0),
					Point3::<f64>::new(-1.0, 1.0, -1.0),
					Point3::<f64>::new(1.0, -1.0, -1.0),
					Point3::<f64>::new(1.0, 1.0, 1.0),
				],
				vec![
					Vector3::<i32>::new(2, 0, 1),
					Vector3::<i32>::new(0, 3, 1),
					Vector3::<i32>::new(2, 3, 0),
					Vector3::<i32>::new(3, 2, 1),
				],
			),
			Shape::Cube => (
				vec![
					Point3::<f64>::new(0.0, 0.0, 0.0),
					Point3::<f64>::new(0.0, 0.0, 1.0),
					Point3::<f64>::new(0.0, 1.0, 0.0),
					Point3::<f64>::new(0.0, 1.0, 1.0),
					Point3::<f64>::new(1.0, 0.0, 0.0),
					Point3::<f64>::new(1.0, 0.0, 1.0),
					Point3::<f64>::new(1.0, 1.0, 0.0),
					Point3::<f64>::new(1.0, 1.0, 1.0),
				],
				vec![
					Vector3::<i32>::new(1, 0, 4),
					Vector3::<i32>::new(2, 4, 0),
					Vector3::<i32>::new(1, 3, 0),
					Vector3::<i32>::new(3, 1, 5),
					Vector3::<i32>::new(3, 2, 0),
					Vector3::<i32>::new(3, 7, 2),
					Vector3::<i32>::new(5, 4, 6),
					Vector3::<i32>::new(5, 1, 4),
					Vector3::<i32>::new(6, 4, 2),
					Vector3::<i32>::new(7, 6, 2),
					Vector3::<i32>::new(7, 3, 5),
					Vector3::<i32>::new(7, 5, 6),
				],
			),
			Shape::Octahedron => (
				vec![
					Point3::<f64>::new(1.0, 0.0, 0.0),
					Point3::<f64>::new(-1.0, 0.0, 0.0),
					Point3::<f64>::new(0.0, 1.0, 0.0),
					Point3::<f64>::new(0.0, -1.0, 0.0),
					Point3::<f64>::new(0.0, 0.0, 1.0),
					Point3::<f64>::new(0.0, 0.0, -1.0),
				],
				vec![
					Vector3::<i32>::new(0, 2, 4),
					Vector3::<i32>::new(1, 5, 3),
					Vector3::<i32>::new(2, 1, 4),
					Vector3::<i32>::new(3, 5, 0),
					Vector3::<i32>::new(1, 3, 4),
					Vector3::<i32>::new(0, 5, 2),
					Vector3::<i32>::new(3, 0, 4),
					Vector3::<i32>::new(2, 5, 1),
				],
			),
		};

		for v in &mut vert_pos {
			v.coords = m * v.coords.push(1.0);
		}

		let mut meshbool_impl = Self {
			vert_pos,
			..MeshBoolImpl::default()
		};

		meshbool_impl.create_halfedges(tri_verts, Vec::new());
		meshbool_impl.initialize_original();
		meshbool_impl.calculate_bbox();
		meshbool_impl.set_epsilon(-1.0, false);
		meshbool_impl.sort_geometry();
		meshbool_impl.set_normals_and_coplanar();

		meshbool_impl
	}

	pub fn remove_unreferenced_verts(&mut self) {
		let num_vert = self.num_vert();
		let keep = vec![0; num_vert];
		for edge in 0..self.halfedge.len() {
			let start_vert = self.halfedge.start(edge as i32);
			if start_vert >= 0 {
				let atomic_ref: &AtomicI32 = unsafe { mem::transmute(&keep[start_vert as usize]) };
				atomic_ref.store(1, AtomicOrdering::Relaxed);
			}
		}

		for v in 0..num_vert {
			if keep[v] == 0 {
				self.vert_pos[v] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
			}
		}
	}

	fn eager_transform_prop_normals(
		halfedge: &Halfedges,
		mesh_relation: &MeshRelationD,
		normal_transform: Matrix3<f64>,
		properties: &mut [f64],
		num_prop_vert: usize,
		stride: i32,
		offset: i32,
	) {
		// Short-circuit when no meshID carries normals. OR semantics (any has
		// it), unlike AllHaveNormals() - mixed inputs still need the per-meshID
		// iteration below to rotate the with-normals subset.
		let mut any_has_normals = false;
		for m in mesh_relation.mesh_id_transform.values() {
			if m.has_normals {
				any_has_normals = true;
				break;
			}
		}

		if !any_has_normals {
			return;
		}
		let mut prop_visited = vec![false; num_prop_vert];
		for e in 0..halfedge.len() as i32 {
			if !MeshBoolImpl::tri_has_normals(mesh_relation, e / 3) {
				continue;
			}
			let prop = halfedge.prop(e);
			if prop < 0 || prop_visited[prop as usize] {
				continue;
			}
			prop_visited[prop as usize] = true;
			let mut n = Vector3::default();
			for i in 0..3 {
				n[i as usize] = properties[((offset + prop) * stride + i) as usize];
			}
			// Re-normalize as we transform: non-orthogonal transforms (scale) and
			// barycentric interpolation upstream both leave non-unit values that
			// would otherwise compound and break downstream lighting / smoothing.
			n = safe_normalize(normal_transform * n);
			for i in 0..3 {
				properties[((offset + prop) * stride + i) as usize] = n[i as usize];
			}
		}
	}

	pub fn reserve_ids(n: usize) -> usize {
		MESH_ID_COUNTER.fetch_add(n, AtomicOrdering::Relaxed)
	}

	pub fn initialize_original(&mut self) {
		let mesh_id = MeshBoolImpl::reserve_ids(1) as i32;
		self.mesh_relation.original_id = mesh_id;
		let num_tri = self.num_tri();
		unsafe {
			vec_resize_nofill(&mut self.mesh_relation.tri_ref, num_tri);
		}
		for i in 0..num_tri {
			let tri = &mut self.mesh_relation.tri_ref[i];
			let coplanar_id = tri.coplanar_id;
			*tri = TriRef {
				mesh_id,
				original_id: mesh_id,
				face_id: -1,
				coplanar_id,
			};
		}

		// Preserve the AND-across-old-Relations state so AsOriginal keeps the
		// recording when it builds a fresh Relation. Primitives start with an
		// empty map, which AllHaveNormals() returns false for.
		let had_normals = self.all_have_normals();
		self.mesh_relation.mesh_id_transform.clear();
		self.mesh_relation
			.mesh_id_transform
			.entry(mesh_id)
			.or_insert_with(|| Relation {
				original_id: mesh_id,
				transform: Matrix3x4::identity(),
				back_side: false,
				has_normals: had_normals,
			});
	}

	pub fn set_normals_and_coplanar(&mut self) {
		let num_tri = self.num_tri();
		vec_resize(&mut self.face_normal, num_tri, Vector3::default());
		struct TriPriority {
			area2: f64,
			tri: i32,
		}
		let mut tri_priority = unsafe { vec_uninit(num_tri) };
		for tri in 0..num_tri {
			self.mesh_relation.tri_ref[tri].coplanar_id = -1;
			if self.halfedge.start((3 * tri) as i32) < 0 {
				tri_priority[tri] = TriPriority {
					area2: 0.0,
					tri: tri as i32,
				};
				continue;
			}

			let v = self.vert_pos[self.halfedge.start((3 * tri) as i32) as usize];
			let n = (self.vert_pos[self.halfedge.end(3 * (tri as i32)) as usize] - v)
				.cross(&(self.vert_pos[self.halfedge.end((3 * tri + 1) as i32) as usize] - v));
			self.face_normal[tri] = n.normalize();
			if self.face_normal[tri].x.is_nan() {
				self.face_normal[tri] = Vector3::new(0.0, 0.0, 1.0);
			}
			tri_priority[tri] = TriPriority {
				area2: n.magnitude_squared(),
				tri: tri as i32,
			};
		}

		tri_priority.sort_by(|a, b| b.area2.partial_cmp(&a.area2).unwrap_or(CmpOrdering::Equal));

		let mut interior_halfedges: Vec<i32> = Vec::default();
		for tp in &tri_priority {
			if self.mesh_relation.tri_ref[tp.tri as usize].coplanar_id >= 0 {
				continue;
			}

			self.mesh_relation.tri_ref[tp.tri as usize].coplanar_id = tp.tri;
			if self.halfedge.start(3 * tp.tri) < 0 {
				continue;
			}
			let base = self.vert_pos[self.halfedge.start(3 * tp.tri) as usize];
			let normal = self.face_normal[tp.tri as usize];
			vec_resize(&mut interior_halfedges, 3, 0);
			interior_halfedges[0] = 3 * tp.tri;
			interior_halfedges[1] = 3 * tp.tri + 1;
			interior_halfedges[2] = 3 * tp.tri + 2;
			while !interior_halfedges.is_empty() {
				let h = next_halfedge(self.halfedge.pair(*interior_halfedges.last().unwrap()));
				interior_halfedges.pop().unwrap();
				if self.mesh_relation.tri_ref[(h / 3) as usize].coplanar_id >= 0 {
					continue;
				}

				let v = self.vert_pos[self.halfedge.end(h) as usize];
				if (v - base).dot(&normal).abs() < self.tolerance {
					let tri = (h / 3) as usize;
					self.mesh_relation.tri_ref[tri].coplanar_id = tp.tri;
					self.face_normal[tri] = normal;

					if interior_halfedges.is_empty()
						|| h != self.halfedge.pair(*interior_halfedges.last().unwrap())
					{
						interior_halfedges.push(h);
					} else {
						interior_halfedges.pop().unwrap();
					}

					let h_next = next_halfedge(h);
					interior_halfedges.push(h_next);
				}
			}
		}
		self.calculate_vert_normals();
	}

	///Dereference duplicate property vertices if they are exactly floating-point
	///equal. These unreferenced properties are then removed by CompactProps.
	pub fn dedupe_prop_verts(&mut self) {
		let num_prop = self.num_prop();
		if num_prop == 0 {
			return;
		}

		let mut vert2vert: Vec<(i32, i32)> = vec![(-1, -1); self.halfedge.len()];
		for edge_idx in 0..self.halfedge.len() {
			let pair = self.halfedge.pair(edge_idx as i32);
			if pair < 0 {
				continue;
			}
			let edge_face = edge_idx / 3;
			let pair_face = pair / 3;

			if self.mesh_relation.tri_ref[edge_face].mesh_id
				!= self.mesh_relation.tri_ref[pair_face as usize].mesh_id
			{
				continue;
			}

			let prop0 = self.halfedge.prop(edge_idx as i32);
			let prop1 = self.halfedge.prop(next_halfedge(pair));
			let mut prop_equal = true;
			for p in 0..num_prop {
				if self.properties[num_prop * prop0 as usize + p]
					!= self.properties[num_prop * prop1 as usize + p]
				{
					prop_equal = false;
					break;
				}
			}
			if prop_equal {
				vert2vert[edge_idx] = (prop0, prop1);
			}
		}

		let mut vert_labels: Vec<i32> = vec![];
		let num_prop_vert = self.num_prop_vert();

		fn get_labels(components: &mut Vec<i32>, edges: &Vec<(i32, i32)>, num_nodes: i32) -> i32 {
			let uf = DisjointSets::new(num_nodes as u64);
			for edge in edges {
				if edge.0 == -1 || edge.1 == -1 {
					continue;
				}
				uf.unite(edge.0 as u64, edge.1 as u64);
			}

			return uf.connected_components(components) as i32;
		}

		let num_labels = get_labels(&mut vert_labels, &vert2vert, num_prop_vert as i32) as usize;

		let mut label2vert: Vec<i32> = vec![0; num_labels];
		for v in 0..num_prop_vert {
			label2vert[vert_labels[v] as usize] = v as i32;
		}

		for edge in 0..self.halfedge.len() as i32 {
			self.halfedge.set_prop(
				edge,
				label2vert[vert_labels[self.halfedge.prop(edge) as usize] as usize],
			);
		}
	}

	///Create the halfedge_ data structure from a list of triangles. If the optional
	///prop2vert array is missing, it's assumed these triangles are are pointing to
	///both vert and propVert indices. If prop2vert is present, the triangles are
	///assumed to be pointing to propVert indices only. The prop2vert array is used
	///to map the propVert indices to vert indices.
	pub fn create_halfedges(&mut self, tri_prop: Vec<Vector3<i32>>, tri_vert: Vec<Vector3<i32>>) {
		let num_tri = tri_prop.len();
		let num_halfedge = (3 * num_tri) as i32;
		let mut halfedge = unsafe { vec_uninit(num_halfedge as usize) };

		let vert_count = self.vert_pos.len() as i32;

		//PrepHalfedges start
		let mut ids = {
			let ids = if vert_count < (1 << 18) {
				// For small vertex count, it is faster to just do sorting
				let mut edge: Vec<u64> = unsafe { vec_uninit(num_halfedge as usize) };
				let mut set_edge = |e: i32, v0: i32, v1: i32| {
					edge[e as usize] = (if v0 < v1 { 1 } else { 0 }) << 63
						| (v0.min(v1) as u64) << 32
						| (v0.max(v1) as u64);
				};

				if tri_vert.is_empty() {
					let mut job = PrepHalfedges::<true, _> {
						halfedges: &mut halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_edge,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				} else {
					let mut job = PrepHalfedges::<false, _> {
						halfedges: &mut halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_edge,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				}

				let mut ids: Vec<i32> = (0..num_halfedge).collect();
				ids.sort_by_key(|&i| edge[i as usize]);
				ids
			} else {
				// For larger vertex count, we separate the ids into slices for halfedges
				// with the same smaller vertex.
				// We first copy them there (as HalfedgePairData), and then do sorting
				// locally for each slice.
				// This helps with memory locality, and is faster for larger meshes.
				let mut entries = unsafe { vec_uninit(num_halfedge as usize) };
				let mut offsets: Vec<i32> = vec![0; (vert_count * 2) as usize];
				let mut set_offset = |_e: i32, v0: i32, v1: i32| {
					let offset = if v0 > v1 { 0 } else { vert_count };
					unsafe {
						atomic_add_i32(&mut offsets[(v0.min(v1) + offset) as usize], 1);
					}
				};

				if tri_vert.is_empty() {
					let mut job = PrepHalfedges::<true, _> {
						halfedges: &mut halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_offset,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				} else {
					let mut job = PrepHalfedges::<false, _> {
						halfedges: &mut halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_offset,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				}

				exclusive_scan_in_place(&mut offsets, 0);

				for tri in 0..num_tri {
					let tri = tri as i32;
					for i in 0..3 {
						let e = 3 * tri + i;
						let e_usize = e as usize;
						let v0 = halfedge[e_usize].start_vert;
						let v1 = halfedge[e_usize].end_vert;
						let offset = if v0 > v1 { 0 } else { vert_count as i32 };
						let start = v0.min(v1);
						let index =
							unsafe { atomic_add_i32(&mut offsets[(start + offset) as usize], 1) };
						entries[index as usize] = HalfedgePairData {
							large_vert: v0.max(v1),
							tri,
							edge_index: e,
						};
					}
				}

				let mut ids: Vec<i32> = unsafe { vec_uninit(num_halfedge as usize) };
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
		let num_edge = num_halfedge / 2;
		let mut removed = vec![false; num_halfedge as usize];

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

		self.halfedge = Halfedges::default();
		unsafe {
			self.halfedge.resize_nofill(num_halfedge as usize);
		}
		for i in 0..num_edge {
			let pair0 = ids[i as usize];
			let pair1 = ids[(i + num_edge) as usize];
			if !removed[pair0 as usize] {
				self.halfedge
					.set_start(pair0, halfedge[pair0 as usize].start_vert);
				self.halfedge
					.set_prop(pair0, halfedge[pair0 as usize].prop_vert);
				self.halfedge.set_pair(pair0, pair1);
				self.halfedge
					.set_start(pair1, halfedge[pair1 as usize].start_vert);
				self.halfedge
					.set_prop(pair1, halfedge[pair1 as usize].prop_vert);
				self.halfedge.set_pair(pair1, pair0);
			} else {
				self.halfedge.set_start(pair0, -1);
				self.halfedge.set_prop(pair0, 0);
				self.halfedge.set_pair(pair0, -1);
				self.halfedge.set_start(pair1, -1);
				self.halfedge.set_prop(pair1, 0);
				self.halfedge.set_pair(pair1, -1);
			}
		}
	}

	pub fn make_empty(&mut self, status: MeshBoolError) {
		self.bbox = AABB::default();
		self.vert_pos = Vec::default();
		self.halfedge = Halfedges::default();
		self.vert_normal = Vec::default();
		self.face_normal = Vec::default();
		self.mesh_relation = MeshRelationD::default();
		self.collider = Collider::default();
		self.status = status;
	}

	pub fn warp(&mut self, mut warp_func: impl FnMut(&mut Point3<f64>)) {
		self.warp_batch(|vecs| {
			vecs.iter_mut().for_each(|v| warp_func(v));
		});
	}

	pub fn warp_batch(&mut self, mut warp_func: impl FnMut(&mut [Point3<f64>])) {
		warp_func(&mut self.vert_pos);
		self.calculate_bbox();
		if !self.is_finite() {
			self.make_empty(MeshBoolError::NonFiniteVertex);
			return;
		}
		self.set_epsilon(-1.0, false);
		self.sort_geometry();
		self.set_normals_and_coplanar();
		self.mesh_relation.original_id = -1;
	}

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(&self, transform: &Matrix3x4<f64>) -> MeshBoolImpl {
		if *transform == Matrix3x4::identity() {
			return self.clone();
		}
		let mut result = MeshBoolImpl::default();
		if self.status != MeshBoolError::NoError {
			result.status = self.status;
			return result;
		}
		if !transform.iter().fold(true, |acc, e| acc && e.is_finite()) {
			result.make_empty(MeshBoolError::NonFiniteVertex);
			return result;
		}

		result.mesh_relation = self.mesh_relation.clone();
		result.epsilon = self.epsilon;
		result.tolerance = self.tolerance;
		result.num_prop = self.num_prop;
		result.properties = self.properties.clone();
		result.bbox = self.bbox;
		result.halfedge = self.halfedge.clone();

		result.mesh_relation.original_id = -1;
		for m in &mut result.mesh_relation.mesh_id_transform {
			m.1.transform = transform * mat4(&m.1.transform);
		}

		vec_resize(&mut result.vert_pos, self.num_vert(), Point3::default());
		vec_resize(
			&mut result.face_normal,
			self.face_normal.len(),
			Vector3::default(),
		);
		vec_resize(
			&mut result.vert_normal,
			self.vert_normal.len(),
			Vector3::default(),
		);
		for i in 0..self.vert_pos.len() {
			let v = &self.vert_pos[i];
			result.vert_pos[i] = (transform * Vector4::new(v.x, v.y, v.z, 1.0)).into();
		}

		let normal_transform = normal_transform(transform);
		for i in 0..self.face_normal.len() {
			result.face_normal[i] = transform_normal(normal_transform, self.face_normal[i]);
		}
		for i in 0..self.vert_normal.len() {
			result.vert_normal[i] = transform_normal(normal_transform, self.vert_normal[i]);
		}

		if self.num_prop >= 3 {
			MeshBoolImpl::eager_transform_prop_normals(
				&self.halfedge,
				&self.mesh_relation,
				normal_transform,
				&mut result.properties,
				self.num_prop_vert(),
				self.num_prop,
				0,
			);
		}

		let invert = mat3(transform).determinant() < 0.0;
		if invert {
			for tri in 0..result.num_tri() {
				FlipTris {
					halfedge: &mut result.halfedge,
				}
				.call(tri as i32);
			}
		}

		result.calculate_bbox();
		result.epsilon *= mat3(transform).svd(false, false).singular_values[0];
		result.set_epsilon(result.epsilon, false);

		if !result.is_empty() {
			if Collider::is_axis_aligned(transform) {
				result.collider = self.collider.clone();
				result.collider.transform(*transform);
			} else {
				result.collider = self.collider.clone();
				let mut face_box = Vec::new();
				let mut face_morton = Vec::new();
				result.get_face_box_morton(&mut face_box, &mut face_morton);
				result.collider.update_boxes(&face_box);
			}
		}

		result
	}

	///This function uses the face normals to compute
	///vertex normals (angle-weighted pseudo-normals). Face normals should only be
	///calculated when needed because nearly degenerate faces will accrue rounding
	///error, while the Boolean can retain their original normal, which is more
	///accurate and can help with merging coplanar faces.
	pub fn calculate_vert_normals(&mut self) {
		let num_vert = self.num_vert();
		vec_resize(&mut self.vert_normal, num_vert, Vector3::default());

		let vert_halfedge_map: Vec<AtomicI32> = (0..self.num_vert())
			.map(|_| AtomicI32::new(i32::MAX))
			.collect();

		let atomic_min = |value, vert: i32| {
			if vert < 0 {
				return;
			}
			let mut old = i32::MAX;
			while let Err(actual) = vert_halfedge_map[vert as usize].compare_exchange(
				old,
				value,
				AtomicOrdering::SeqCst,
				AtomicOrdering::SeqCst,
			) {
				old = actual;
				if old < value {
					break;
				}
			}
		};

		for i in 0..self.halfedge.len() as i32 {
			atomic_min(i, self.halfedge.start(i));
		}

		for vert in 0..self.num_vert() {
			let first_edge = vert_halfedge_map[vert].load(AtomicOrdering::SeqCst);
			// not referenced
			if first_edge == i32::MAX {
				self.vert_normal[vert] = Vector3::from_element(0.0);
				continue;
			}

			let mut normal = Vector3::from_element(0.0);
			self.for_vert(first_edge, |edge| {
				let tri_verts = Vector3::<i32>::new(
					self.halfedge.start(edge),
					self.halfedge.end(edge),
					self.halfedge.end(next_halfedge(edge)),
				);
				let curr_edge = (self.vert_pos[tri_verts[1] as usize]
					- self.vert_pos[tri_verts[0] as usize])
					.normalize();
				let prev_edge = (self.vert_pos[tri_verts[0] as usize]
					- self.vert_pos[tri_verts[2] as usize])
					.normalize();

				// if it is not finite, this means that the triangle is degenerate, and we
				// should just exclude it from the normal calculation...
				if !curr_edge[0].is_finite() || !prev_edge[0].is_finite() {
					return;
				}
				let dot = -prev_edge.dot(&curr_edge);
				let phi = if dot >= 1.0 {
					0.0
				} else if dot <= -1.0 {
					f64::consts::PI
				} else {
					libm::acos(dot)
				};
				normal += phi * self.face_normal[(edge / 3) as usize];
			});

			self.vert_normal[vert] = safe_normalize(normal);
		}
	}

	pub fn set_epsilon(&mut self, min_epsilon: f64, use_single: bool) {
		self.epsilon = max_epsilon(min_epsilon, &self.bbox);
		let mut min_tol = self.epsilon;
		if use_single {
			min_tol = min_tol.max(f32::EPSILON as f64 * self.bbox.scale());
		}

		self.tolerance = self.tolerance.max(min_tol);
	}

	///Remaps all the contained meshIDs to new unique values to represent new
	///instances of these meshes.
	pub fn increment_mesh_ids(&mut self) {
		//in c++ this uses a custom hashtable class
		let mut mesh_id_old2new =
			HashMap::with_capacity(self.mesh_relation.mesh_id_transform.len() * 2);
		let old_transforms = mem::take(&mut self.mesh_relation.mesh_id_transform);
		let num_mesh_ids = old_transforms.len();
		let mut next_mesh_id = MeshBoolImpl::reserve_ids(num_mesh_ids) as i32;
		for pair in old_transforms {
			mesh_id_old2new.insert(pair.0, next_mesh_id);
			self.mesh_relation
				.mesh_id_transform
				.entry(next_mesh_id)
				.or_insert(pair.1);
			next_mesh_id += 1;
		}

		let num_tri = self.num_tri();
		for i in 0..num_tri {
			let tri_ref = &mut self.mesh_relation.tri_ref[i];
			tri_ref.mesh_id = *mesh_id_old2new.get(&tri_ref.mesh_id).unwrap()
		}
	}

	#[inline]
	pub fn for_vert(&self, halfedge: i32, mut func: impl FnMut(i32)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.halfedge.pair(current));
			func(current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline]
	pub fn for_vert_mut(&mut self, halfedge: i32, mut func: impl FnMut(&mut Self, i32)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.halfedge.pair(current));
			func(self, current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline]
	pub fn for_vert_fn<T>(
		&self,
		halfedge: i32,
		mut transform: impl FnMut(i32) -> T,
		mut binary_op: impl FnMut(i32, &T, &mut T),
	) {
		let mut here: T = transform(halfedge);
		let mut current: i32 = halfedge;
		loop {
			let next_halfedge: i32 = next_halfedge(self.halfedge.pair(current));
			let mut next: T = transform(next_halfedge);
			binary_op(current, &here, &mut next);
			here = next;
			current = next_halfedge;
			if current == halfedge {
				break;
			}
		}
	}
}

impl Default for MeshBoolImpl {
	fn default() -> Self {
		Self {
			bbox: AABB::default(),
			epsilon: -1.0,
			tolerance: -1.0,
			num_prop: 0,
			status: MeshBoolError::NoError,
			vert_pos: Vec::default(),
			halfedge: Halfedges::default(),
			properties: Vec::default(),
			vert_normal: Vec::default(),
			face_normal: Vec::default(),
			mesh_relation: MeshRelationD::default(),
			collider: Collider::default(),
		}
	}
}
