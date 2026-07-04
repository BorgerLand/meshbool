use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation, reserve_original_id};
use crate::meshgl::MeshGL;
use crate::postprocessing as pp;
use crate::postprocessing::sort::morton_code;
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::{BVHCollider, SimpleRecorder};
use crate::util::disjoint_sets::DisjointSets;
use crate::util::hash_table::DeterministicMap;
use crate::util::math::K_PRECISION;
use crate::util::num_convert::{LossyFrom, LossyInto};
use crate::util::vec_ext;
use crate::{MeshBool, Precision, Properties, Triangles};
use nalgebra::{Matrix3x4, Point3, Vector3};
use std::any::TypeId;
use std::{array, mem};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MeshGLError {
	NonFiniteVertex,
	InvalidConstruction,
	NotManifold,
	MissingPositionProperties,
	MergeVectorsDifferentLengths,
	TransformWrongLength,
	RunIndexWrongLength,
	FaceIDWrongLength,
	MergeIndexOutOfBounds,
	VertexOutOfBounds,
}

impl<F, I> TryFrom<&MeshGL<F, I>> for MeshBool
where
	F: LossyFrom<f64> + Copy + 'static,
	f64: From<F>,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	i32: LossyFrom<I>,
{
	type Error = MeshGLError;

	///Convert a MeshGL into a Manifold, retaining its properties and merging only
	///the positions according to the merge vectors. Will return an empty Manifold
	///and set an Error Status if the result is not an oriented 2-manifold. Will
	///collapse degenerate triangles and unnecessary vertices.
	///
	///All fields are read, making this structure suitable for a lossless round-trip
	///of data from GetMeshGL. For multi-material input, use ReserveIDs to set a
	///unique originalID for each material, and sort the materials into triangle
	///runs.
	fn try_from(mesh_gl: &MeshGL<F, I>) -> Result<Self, Self::Error> {
		let num_vert = usize::lossy_from(mesh_gl.num_vert());
		let num_tri = usize::lossy_from(mesh_gl.num_tri());
		let mut instance_relation = DeterministicMap::new();

		if num_vert == 0 && num_tri == 0 {
			return Ok(MeshBool {
				original_id: None,
				precision: Precision {
					epsilon: -1.0,
					tolerance: -1.0,
				},
				vert_pos: Vec::default(),
				tri: Triangles::default(),
				properties: Properties::default(),
				instance_relation,
				collider: BVHCollider::default(),
			});
		}

		if num_vert < 4 || num_tri < 4 {
			return Err(MeshGLError::NotManifold);
		}

		if usize::lossy_from(mesh_gl.prop_stride) < 3 {
			return Err(MeshGLError::MissingPositionProperties);
		}

		if mesh_gl.merge_from_vert.len() != mesh_gl.merge_to_vert.len() {
			return Err(MeshGLError::MergeVectorsDifferentLengths);
		}

		if !mesh_gl.run_transform.is_empty()
			&& 12 * mesh_gl.run_original_id.len() != mesh_gl.run_transform.len()
		{
			return Err(MeshGLError::TransformWrongLength);
		}

		if !mesh_gl.run_original_id.is_empty()
			&& !mesh_gl.run_index.is_empty()
			&& mesh_gl.run_original_id.len() + 1 != mesh_gl.run_index.len()
			&& mesh_gl.run_original_id.len() != mesh_gl.run_index.len()
		{
			return Err(MeshGLError::RunIndexWrongLength);
		}

		if !mesh_gl.face_id.is_empty() && mesh_gl.face_id.len() != num_tri {
			return Err(MeshGLError::FaceIDWrongLength);
		}

		if mesh_gl
			.vert_properties
			.iter()
			.any(|v| !f64::from(*v).is_finite())
		{
			return Err(MeshGLError::NonFiniteVertex);
		}

		if mesh_gl
			.run_transform
			.iter()
			.any(|x| !f64::from(*x).is_finite())
		{
			return Err(MeshGLError::InvalidConstruction);
		}

		let prop2vert = if !mesh_gl.merge_from_vert.is_empty() {
			let mut prop2vert: Vec<_> = (0..num_vert as i32).collect();
			for i in 0..mesh_gl.merge_from_vert.len() {
				let from = usize::lossy_from(mesh_gl.merge_from_vert[i]);
				let to = usize::lossy_from(mesh_gl.merge_to_vert[i]);
				if from >= num_vert || to >= num_vert {
					return Err(MeshGLError::MergeIndexOutOfBounds);
				}
				prop2vert[from] = to as i32;
			}
			prop2vert
		} else {
			vec![]
		};

		let prop_stride = usize::lossy_from(mesh_gl.prop_stride) - 3;
		let mut properties = unsafe { vec_ext::uninit(num_vert * prop_stride) };
		let tolerance = f64::from(mesh_gl.tolerance);
		// This will have unreferenced duplicate positions that will be removed by
		// Impl::remove_unreferenced_verts().
		let mut vert_pos: Vec<Point3<f64>> = unsafe { vec_ext::uninit(num_vert) };

		for i in 0..num_vert {
			for j in [0, 1, 2] {
				vert_pos[i][j] =
					mesh_gl.vert_properties[usize::lossy_from(mesh_gl.prop_stride) * i + j].into();
			}
			for j in 0..prop_stride {
				properties[i * prop_stride + j] = mesh_gl.vert_properties
					[usize::lossy_from(mesh_gl.prop_stride) * i + 3 + j]
					.into();
			}
		}

		let mut properties = Properties {
			data: properties,
			stride: prop_stride,
		};
		let mut tri_rel: Vec<TriRelation> = unsafe { vec_ext::uninit(num_tri) };

		let mut run_index = mesh_gl.run_index.clone();
		let run_end = mesh_gl.tri_verts.len();
		if run_index.is_empty() {
			run_index = vec![I::lossy_from(0), I::lossy_from(run_end)];
		} else if run_index.len() == mesh_gl.run_original_id.len() {
			run_index.push(I::lossy_from(run_end));
		} else if run_index.len() == 1 {
			run_index.push(I::lossy_from(run_end));
		}

		let mut run_original_id = mesh_gl.run_original_id.clone();
		let original_id = if run_original_id.is_empty() {
			let original_id = reserve_original_id();
			run_original_id.push(original_id as u32);
			Some(original_id)
		} else {
			None
		};

		for i in 0..run_original_id.len() {
			let instance_id = i as u32;
			let original_id = run_original_id[i];
			let back_side = mesh_gl.back_side(i);
			// Per-run hasNormals (runFlags bit 1). Defensively require numProp >= 3
			// so a caller setting the bit on a too-small MeshGL doesn't make us read
			// past the property bounds.
			let run_has_n = mesh_gl.has_normals(i) && prop_stride >= 3;
			for tri in usize::lossy_from(run_index[i]) / 3..usize::lossy_from(run_index[i + 1]) / 3
			{
				tri_rel[tri as usize] = TriRelation {
					instance_id,
					face_id: if mesh_gl.face_id.is_empty() {
						-1
					} else {
						i32::lossy_from(mesh_gl.face_id[tri])
					},
					coplanar_id: -1,
				};
			}

			if mesh_gl.run_transform.is_empty() {
				instance_relation.insert(
					instance_id,
					InstanceRelation {
						original_id,
						transform: Matrix3x4::identity(),
						back_side,
						has_normals: run_has_n,
					},
				);
			} else {
				let m: [_; 12] = array::from_fn(|j| f64::from(mesh_gl.run_transform[i * 12 + j]));
				instance_relation.insert(
					instance_id,
					InstanceRelation {
						original_id,
						transform: Matrix3x4::from_column_slice(&m),
						back_side,
						has_normals: run_has_n,
					},
				);
			}
		}

		let mut tri_vert = Vec::with_capacity(num_tri);
		let mut tri_prop =
			(prop_stride > 0 && !prop2vert.is_empty()).then(|| Vec::with_capacity(num_tri));
		for i in 0..num_tri {
			let mut tri_v = Vector3::default();
			let mut tri_p = Vector3::default();
			for j in [0, 1, 2] {
				let vert = usize::lossy_from(mesh_gl.tri_verts[3 * i + j]);
				if vert >= num_vert {
					return Err(MeshGLError::VertexOutOfBounds);
				}

				tri_v[j] = vert as i32;
				if tri_prop.is_some() {
					tri_p[j] = prop2vert[vert as usize];
				}
			}
			if tri_v[0] != tri_v[1]
				&& tri_v[1] != tri_v[2]
				&& tri_v[2] != tri_v[0]
				&& (tri_prop.is_none()
					|| tri_p[0] != tri_p[1] && tri_p[1] != tri_p[2] && tri_p[2] != tri_p[0])
			{
				tri_vert.push(tri_v);
				if let Some(tri_prop) = &mut tri_prop {
					tri_prop.push(tri_p);
				}
			}
		}

		let mut halfedge = Halfedges::from_tri_indices(vert_pos.len(), tri_vert, tri_prop);
		if !halfedge.is_manifold() {
			return Err(MeshGLError::NotManifold);
		}

		let bbox = Box3D::from_cloud(&vert_pos);
		let precision = Precision::new(bbox, tolerance, TypeId::of::<F>() == TypeId::of::<f32>());

		// we need to split pinched verts before calculating vertex normals, because
		// the algorithm doesn't work with pinched verts
		pp::split_pinched_verts(&mut halfedge, &mut vert_pos);
		pp::dedupe_prop_verts(&mut halfedge, &tri_rel, &properties);
		let tri_normal =
			pp::set_normals_and_coplanar(&mut tri_rel, &halfedge, &vert_pos, precision.tolerance);
		let mut tri = Triangles {
			halfedge,
			normal: tri_normal,
			relation: tri_rel,
		};
		pp::dedupe_edges(&mut tri, &mut vert_pos);
		pp::collapse_short_edges(
			&mut tri.halfedge,
			&mut vert_pos,
			&tri.normal,
			&tri.relation,
			prop_stride,
			precision,
			0,
		);
		pp::swap_degenerates(&mut tri, &mut vert_pos, &mut properties, precision, 0);
		pp::mark_unreferenced_verts(&mut tri.halfedge, &mut vert_pos);
		let Some(collider) =
			pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
		else {
			return Ok(MeshBool::decimated(
				original_id,
				instance_relation,
				prop_stride,
				precision,
			));
		};

		Ok(MeshBool {
			original_id,
			precision,
			vert_pos,
			properties,
			tri,
			instance_relation,
			collider,
		})
	}
}

//consuming variant
impl<F, I> TryFrom<MeshGL<F, I>> for MeshBool
where
	F: LossyFrom<f64> + Copy + 'static,
	f64: From<F>,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	i32: LossyFrom<I>,
{
	type Error = MeshGLError;

	///Convert a MeshGL into a Manifold, retaining its properties and merging only
	///the positions according to the merge vectors. Will return an empty Manifold
	///and set an Error Status if the result is not an oriented 2-manifold. Will
	///collapse degenerate triangles and unnecessary vertices.
	///
	///All fields are read, making this structure suitable for a lossless round-trip
	///of data from GetMeshGL. For multi-material input, use ReserveIDs to set a
	///unique originalID for each material, and sort the materials into triangle
	///runs.
	fn try_from(value: MeshGL<F, I>) -> Result<Self, Self::Error> {
		Self::try_from(&value)
	}
}

impl<F, I> MeshGL<F, I>
where
	F: LossyFrom<f64> + Copy + 'static,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	u64: LossyFrom<I>,
	i32: LossyFrom<I>,
	f64: LossyFrom<F>,
{
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
	pub fn merge(&mut self) -> bool
	where
		F: LossyFrom<f64> + Copy + 'static,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
		u64: LossyFrom<I>,
		i32: LossyFrom<I>,
		f64: LossyFrom<F>,
	{
		let mut open_edges = vec![]; //c++ used multiset

		let mut merge: Vec<i32> = (0..i32::lossy_from(self.num_vert())).collect();
		for i in 0..self.merge_from_vert.len() {
			merge[usize::lossy_from(self.merge_from_vert[i])] =
				i32::lossy_from(self.merge_to_vert[i]);
		}

		let num_vert = usize::lossy_from(self.num_vert());
		let num_tri = usize::lossy_from(self.num_tri());
		let next = [1, 2, 0];
		for tri in 0..num_tri {
			for i in [0, 1, 2] {
				let mut edge = (
					merge[usize::lossy_from(self.tri_verts[3 * tri + next[i] as usize])],
					merge[usize::lossy_from(self.tri_verts[3 * tri + i])],
				);
				let it = open_edges.iter().position(|&p| p == edge);
				if it.is_none() {
					mem::swap(&mut edge.0, &mut edge.1);
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
		let mut open_verts: Vec<_> = open_edges.iter().map(|&(vert, _)| vert).collect();

		let vert_prop_d = self.vert_properties.clone();
		let mut bbox = Box3D::default();
		for i in [0, 1, 2] {
			let min_max = vert_prop_d[i..vert_prop_d.len()]
				.iter()
				.cloned()
				.step_by(usize::lossy_from(self.prop_stride))
				.map(|f| (f64::lossy_from(f), f64::lossy_from(f)))
				.reduce(|acc, b| (acc.0.min(b.0), acc.1.max(b.1)))
				.unwrap_or((core::f64::INFINITY, core::f64::NEG_INFINITY));
			bbox.min[i] = min_max.0;
			bbox.max[i] = min_max.1;
		}

		let tolerance = f64::lossy_from(self.tolerance).max(
			(if TypeId::of::<F>() == TypeId::of::<f32>() {
				core::f32::EPSILON as f64
			} else {
				K_PRECISION
			}) * bbox.scale(),
		);

		// let mut policy = autoPolicy(numOpenVert, 1e5);
		let mut vert_box: Vec<Box3D> = unsafe { vec_ext::uninit(num_open_vert) };
		let mut vert_morton = unsafe { vec_ext::uninit(num_open_vert) };

		(0..num_open_vert).for_each(|i| {
			let vert = open_verts[i];

			let center: Vector3<f64> = Vector3::new(
				self.vert_properties[usize::lossy_from(self.prop_stride) * vert as usize]
					.lossy_into(),
				self.vert_properties[usize::lossy_from(self.prop_stride) * vert as usize + 1]
					.lossy_into(),
				self.vert_properties[usize::lossy_from(self.prop_stride) * vert as usize + 2]
					.lossy_into(),
			);

			vert_box[i].min = center.into();
			vert_box[i].min.iter_mut().for_each(|v| {
				*v -= tolerance / 2.0;
			});
			vert_box[i].max = center.into();
			vert_box[i].max.iter_mut().for_each(|v| {
				*v += tolerance / 2.0;
			});

			vert_morton[i] = morton_code(center.into(), bbox);
		});

		let mut vert_new2old: Vec<_> = (0..num_open_vert as i32).into_iter().collect();
		vert_new2old.sort_by_key(|&i| vert_morton[i as usize]);

		vec_ext::gather_in_place(&mut vert_morton, &vert_new2old);
		vec_ext::gather_in_place(&mut vert_box, &vert_new2old);
		vec_ext::gather_in_place(&mut open_verts, &vert_new2old);

		let collider = BVHCollider::new(&vert_box, &vert_morton);
		let uf = DisjointSets::new(num_vert);

		let mut f = |a: i32, b: i32| {
			uf.unite(
				open_verts[a as usize] as usize,
				open_verts[b as usize] as usize,
			);
		};

		let mut recorder = SimpleRecorder::new(&mut f);
		collider.collisions_from_slice::<true, _>(&mut recorder, &vert_box, false);

		for i in 0..self.merge_from_vert.len() {
			uf.unite(
				usize::lossy_from(self.merge_from_vert[i]),
				usize::lossy_from(self.merge_to_vert[i]),
			);
		}

		self.merge_to_vert = Vec::new();
		self.merge_from_vert = Vec::new();
		for v in 0..num_vert {
			let merge_to = uf.find(v);
			if merge_to != v {
				self.merge_from_vert.push(I::lossy_from(v));
				self.merge_to_vert.push(I::lossy_from(merge_to));
			}
		}

		return true;
	}
}
