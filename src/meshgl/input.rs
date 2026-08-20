use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation, reserve_original_id};
use crate::meshgl::MeshGL;
use crate::postprocessing as pp;
use crate::postprocessing::sort::morton_code;
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::disjoint_sets::DisjointSets;
use crate::util::math::K_PRECISION;
use crate::util::num_convert::LossyFrom;
use crate::util::vec_ext;
use crate::{MeshBool, Precision, Properties, TrianglesWIP};
use nalgebra::{Matrix3x4, Point3, Vector3};
use std::any::TypeId;
use std::array;
use std::rc::Rc;

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
		let requested_tolerance = f64::from(mesh_gl.tolerance);

		let gl_prop_stride = usize::lossy_from(mesh_gl.prop_stride);
		if gl_prop_stride < 3 {
			return Err(MeshGLError::MissingPositionProperties);
		}

		let original_id = mesh_gl
			.run_original_id
			.is_empty()
			.then(|| reserve_original_id());
		let prop_stride = gl_prop_stride - 3;

		let num_vert = usize::lossy_from(mesh_gl.num_vert());
		let num_tri = usize::lossy_from(mesh_gl.num_tri());

		if num_vert == 0 && num_tri == 0 {
			return Ok(MeshBool::decimated(
				original_id,
				Rc::default(),
				prop_stride,
				Precision {
					epsilon: -1.0,
					tolerance: requested_tolerance,
				},
			));
		}

		if num_vert < 4 || num_tri < 4 {
			return Err(MeshGLError::NotManifold);
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

		// This will have unreferenced duplicate positions that will be removed by
		// Impl::remove_unreferenced_verts().
		let mut vert_pos: Vec<Point3<f64>> = Vec::with_capacity(num_vert);
		let mut properties: Vec<f64> = Vec::with_capacity(num_vert * prop_stride);

		for i in 0..num_vert {
			let base = gl_prop_stride * i;
			vert_pos.push(Point3::new(
				mesh_gl.vert_properties[base].into(),
				mesh_gl.vert_properties[base + 1].into(),
				mesh_gl.vert_properties[base + 2].into(),
			));
			properties.extend(
				mesh_gl.vert_properties[base + 3..base + gl_prop_stride]
					.iter()
					.map(|&x| f64::from(x)),
			);
		}

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
			run_original_id.push(original_id);
			Some(original_id)
		} else {
			None
		};

		let mut tri_rel_unfiltered = unsafe { vec_ext::uninit(num_tri) };
		let instance_relation = Rc::new(Vec::from_iter((0..run_original_id.len()).map(|i| {
			let instance_id = i as u32;
			let original_id = run_original_id[i];
			let back_side = mesh_gl.back_side(i);
			// Per-run hasNormals (runFlags bit 1). Defensively require numProp >= 3
			// so a caller setting the bit on a too-small MeshGL doesn't make us read
			// past the property bounds.
			let run_has_n = mesh_gl.has_normals(i) && prop_stride >= 3;
			for tri in usize::lossy_from(run_index[i]) / 3..usize::lossy_from(run_index[i + 1]) / 3
			{
				tri_rel_unfiltered[tri] = TriRelation {
					instance_id,
					face_id: if mesh_gl.face_id.is_empty() {
						-1
					} else {
						i32::lossy_from(mesh_gl.face_id[tri])
					},
				};
			}

			let transform = if mesh_gl.run_transform.is_empty() {
				Matrix3x4::identity()
			} else {
				Matrix3x4::from_column_slice(&array::from_fn::<_, 12, _>(|j| {
					f64::from(mesh_gl.run_transform[i * 12 + j])
				}))
			};

			InstanceRelation {
				original_id,
				transform,
				back_side,
				has_normals: run_has_n,
				user_provided_face_id: !mesh_gl.face_id.is_empty(),
			}
		})));

		let prop2vert = (!mesh_gl.merge_from_vert.is_empty())
			.then(|| {
				let mut prop2vert = Vec::from_iter(0..num_vert as i32);
				for i in 0..mesh_gl.merge_from_vert.len() {
					let from = usize::lossy_from(mesh_gl.merge_from_vert[i]);
					let to = usize::lossy_from(mesh_gl.merge_to_vert[i]);
					if from >= num_vert || to >= num_vert {
						return Err(MeshGLError::MergeIndexOutOfBounds);
					}
					prop2vert[from] = to as i32;
				}

				Ok(prop2vert)
			})
			.transpose()?;

		let mut tri_rel = Vec::with_capacity(num_tri);
		let mut tri_vert = Vec::with_capacity(num_tri);
		let mut tri_prop =
			(prop2vert.is_some() && prop_stride > 0).then(|| Vec::with_capacity(num_tri));
		for i in 0..num_tri {
			let mut tri_v = Vector3::default();
			let mut tri_p = Vector3::default();
			for j in 0..3 {
				let vert = usize::lossy_from(mesh_gl.tri_verts[3 * i + j]);
				if vert >= num_vert {
					return Err(MeshGLError::VertexOutOfBounds);
				}

				if let Some(prop2vert) = &prop2vert {
					tri_v[j] = prop2vert[vert];
					tri_p[j] = vert as i32;
				} else {
					tri_v[j] = vert as i32;
				}
			}

			if tri_v[0] != tri_v[1] && tri_v[1] != tri_v[2] && tri_v[2] != tri_v[0] {
				tri_rel.push(tri_rel_unfiltered[i]);
				tri_vert.push(tri_v);
				if let Some(tri_prop) = &mut tri_prop {
					tri_prop.push(tri_p);
				}
			}
		}

		drop(prop2vert);
		let mut halfedge = Halfedges::from_tri_indices(vert_pos.len(), tri_vert, tri_prop);
		if !halfedge.is_manifold() {
			return Err(MeshGLError::NotManifold);
		}

		let bbox = Box3D::from_cloud(&vert_pos);
		let precision = Precision::new(
			bbox,
			requested_tolerance,
			TypeId::of::<F>() == TypeId::of::<f32>(),
		);
		let mut properties = Properties {
			data: properties,
			stride: prop_stride,
		};

		// we need to split pinched verts before calculating vertex normals, because
		// the algorithm doesn't work with pinched verts
		pp::split_pinched_verts(&mut halfedge, &mut vert_pos);
		pp::dedupe_prop_verts(&mut halfedge, &tri_rel, &properties);
		let tri_normal = pp::set_normals_and_coplanar(
			&mut tri_rel,
			&instance_relation,
			&halfedge,
			&vert_pos,
			precision.tolerance,
		);
		let mut tri = TrianglesWIP {
			halfedge,
			normal: tri_normal,
			relation: tri_rel,
		};
		pp::dedupe_edges(&mut tri, &mut vert_pos);
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
			vert_pos: Rc::new(vert_pos),
			properties: Rc::new(properties),
			tri: tri.into_rc(),
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
	f64: From<F>,
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
	pub fn merge(&mut self) -> bool {
		let mut merge: Vec<i32> = (0..i32::lossy_from(self.num_vert())).collect();
		for i in 0..self.merge_from_vert.len() {
			merge[usize::lossy_from(self.merge_from_vert[i])] =
				i32::lossy_from(self.merge_to_vert[i]);
		}

		let num_vert = usize::lossy_from(self.num_vert());
		let num_tri = usize::lossy_from(self.num_tri());
		let mut edges = Vec::with_capacity(3 * num_tri);
		let next = [1, 2, 0];
		for tri in 0..num_tri {
			for i in 0..3 {
				let first = merge[usize::lossy_from(self.tri_verts[3 * tri + i])];
				let second = merge[usize::lossy_from(self.tri_verts[3 * tri + next[i]])];
				edges.push(encode_open_edge(first, second));
			}
		}
		edges.sort_unstable();

		let mut open_verts = Vec::with_capacity(edges.len());
		// Opposing directed edges cancel in pairs. Repeated edges in the same
		// direction remain, matching the previous multiset behavior. Collapsed
		// self-edges likewise remain only when their count is odd.
		let mut begin = 0;
		while begin < edges.len() {
			let key = edges[begin] >> 1;
			let mut end = begin + 1;
			while end < edges.len() && edges[end] >> 1 == key {
				end += 1;
			}

			let low = (edges[begin] >> 32) as i32;
			let high = ((edges[begin] >> 1) & 0x7FFF_FFFF) as i32;
			if low == high {
				if (end - begin) % 2 == 1 {
					open_verts.push(low);
				}
			} else {
				let mut split = begin;
				while split < end && edges[split] & 1 == 0 {
					split += 1;
				}
				let num_forward = split - begin;
				let num_reverse = end - split;
				let num_open = num_forward.max(num_reverse) - num_forward.min(num_reverse);
				let edge = if num_forward >= num_reverse {
					edges[begin]
				} else {
					edges[end - 1]
				};
				for _ in 0..num_open {
					open_verts.push(open_edge_first(edge));
				}
			}

			begin = end;
		}

		if open_verts.is_empty() {
			return false;
		}
		// The multiset yielded first vertices in ascending order. Preserve that
		// ordering so equal Morton codes keep the same deterministic merge roots.
		open_verts.sort_unstable();

		let num_open_vert = open_verts.len();

		let vert_prop_d = self.vert_properties.clone();
		let gl_prop_stride = usize::lossy_from(self.prop_stride);
		let mut bbox = Box3D::empty();
		for i in 0..3 {
			let min_max = vert_prop_d[i..vert_prop_d.len()]
				.iter()
				.cloned()
				.step_by(gl_prop_stride)
				.map(|f| (f64::from(f), f64::from(f)))
				.reduce(|acc, b| (acc.0.min(b.0), acc.1.max(b.1)))
				.unwrap_or((core::f64::INFINITY, core::f64::NEG_INFINITY));
			bbox.min[i] = min_max.0;
			bbox.max[i] = min_max.1;
		}

		let tolerance = f64::from(self.tolerance).max(
			(if TypeId::of::<F>() == TypeId::of::<f32>() {
				core::f32::EPSILON as f64
			} else {
				K_PRECISION
			}) * bbox.scale(),
		);

		let (mut vert_box, mut vert_morton): (Vec<_>, Vec<_>) = (0..num_open_vert)
			.map(|i| {
				let vert = open_verts[i];
				let base = gl_prop_stride * vert as usize;

				let center =
					Point3::from(array::from_fn(|j| self.vert_properties[base + j].into()));

				let mut min = center;
				min.iter_mut().for_each(|v| *v -= tolerance / 2.0);

				let mut max = center;
				max.iter_mut().for_each(|v| *v += tolerance / 2.0);

				let morton = morton_code(center.into(), bbox);

				(Box3D { min, max }, morton)
			})
			.unzip();

		let mut vert_new2old = Vec::from_iter(0..num_open_vert as i32);
		vert_new2old.sort_unstable_by_key(|&i| vert_morton[i as usize]);

		vert_morton = vec_ext::gather(&vert_morton, vert_new2old.iter());
		vert_box = vec_ext::gather(&vert_box, vert_new2old.iter());
		open_verts = vec_ext::gather(&open_verts, vert_new2old.iter());

		let collider = BVHCollider::new(&vert_box, &vert_morton);
		let mut uf = DisjointSets::new(num_vert);

		collider.collisions_from_slice::<true, _>(
			|a, b| {
				uf.unite(open_verts[a] as usize, open_verts[b] as usize);
			},
			&vert_box,
			false,
		);

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

		true
	}
}

fn encode_open_edge(first: i32, second: i32) -> u64 {
	let low = first.min(second) as u64;
	let high = first.max(second) as u64;
	let direction = if first > second { 1 } else { 0 };
	// Vertex indices are non-negative ints, so 31 bits each leaves one bit for
	// direction. Keeping direction in the low bit groups opposing edges together
	// when the encoded values are sorted.
	(low << 32) | (high << 1) | direction
}

fn open_edge_first(edge: u64) -> i32 {
	let low = (edge >> 32) as i32;
	let high = ((edge >> 1) & 0x7FFF_FFFF) as i32;
	if edge & 1 == 0 { low } else { high }
}
