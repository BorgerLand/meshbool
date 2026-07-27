use crate::MeshBool;
use crate::mesh_relations::{InstanceRelation, all_instances_have_normals};
use crate::meshgl::MeshGL;
use crate::util::hash_table::DeterministicSet;
use crate::util::math::{normal_transform, safe_normalize3};
use crate::util::num_convert::LossyFrom;
use nalgebra::{Matrix3x4, Vector2, Vector3};
use std::any::TypeId;
use std::array;

impl MeshBool {
	///Returns a MeshGL that is designed
	///to easily push into a renderer, including all interleaved vertex properties
	///that may have been input. It also includes relations to all the input meshes
	///that form a part of this result and the transforms applied to each.
	///
	///@param normalIdx If this manifold has properties corresponding to normal
	///vectors, you can specify the first of the three consecutive property channels
	///forming the (x, y, z) normals, which will cause this output MeshGL to
	///automatically update these normals according to the applied transforms and
	///front/back side. normalIdx + 3 must be <= numProp, and all original meshes
	///must use the same channels for their normals. Default is -1: if
	///`CalculateNormals()` recorded normals at the standard slot, that slot is
	///used automatically; otherwise no normals are applied. If normals are
	///selected, the runTransform matrices will be removed from the output, to
	///avoid them being double-applied when round-tripped.
	///Passing a non-negative `normalIdx` is the legacy interface from before
	///`CalculateNormals` recorded the slot on the Manifold itself; prefer the
	///no-arg form after `CalculateNormals(0)`. The explicit-idx path will be
	///removed in a future release.
	#[inline(always)]
	pub fn to_meshgl<F, I>(&self, mut normal_idx: i32) -> MeshGL<F, I>
	where
		F: LossyFrom<f64> + Copy + 'static,
		f64: From<F>,
		I: LossyFrom<i32> + LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
	{
		if normal_idx < 0 && all_instances_have_normals(&self.instance_relation) {
			normal_idx = 0;
		}

		let prop_stride = self.prop_stride();
		let num_vert = self.num_prop_vert();
		let num_tri = self.num_tri();

		let is_original = self.original_id.is_some();
		let update_normals = !is_original && normal_idx >= 0;

		let out_prop_stride = 3 + prop_stride;
		let mut tolerance = self.precision.tolerance;
		if TypeId::of::<F>() == TypeId::of::<f32>() {
			tolerance =
				tolerance.max((f32::EPSILON as f64) * self.collider.get_bounding_box().scale());
		}
		let mut tri_verts = vec![I::lossy_from(0); 3 * num_tri];

		// Sort the triangles into runs
		let mut face_id = vec![I::lossy_from(0); num_tri];
		let mut tri_new2old = Vec::from_iter(0..num_tri as i32);
		let tri_rel = &self.tri.relation;
		// Don't sort originals - keep them in order
		if !is_original {
			tri_new2old.sort_unstable_by_key(|&i| {
				(
					self.instance_relation[tri_rel[i as usize].instance_id as usize].original_id,
					tri_rel[i as usize].instance_id,
				)
			});
		}

		let mut run_index = Vec::new();
		let mut run_original_id = Vec::new();
		let mut run_transform = Vec::new();
		let mut run_flags = Vec::new();

		// runFlags layout: bit 0 = backSide, bit 1 = hasNormals (slot 0..2 of the
		// extra properties is world-frame vertex normals; consumers should skip
		// re-applying runTransform to those channels).
		let mut add_run = |tri, rel: InstanceRelation| {
			run_index.push(I::lossy_from(3 * tri));
			run_original_id.push(rel.original_id);
			// runFlags carries hasNormals (bit 1) which we want on originals too;
			// runTransform is just metadata so skip it for originals where it would
			// always be identity.
			let flags = (rel.back_side as u8) | ((rel.has_normals as u8) << 1);
			run_flags.push(flags);
			if !is_original {
				for col in 0..4 {
					for row in 0..3 {
						run_transform.push(F::lossy_from(rel.transform[(row, col)]))
					}
				}
			}
		};

		let mut unhandled_instance =
			DeterministicSet::from_iter(0..self.instance_relation.len() as u32);
		let mut last_id = None;
		for tri in 0..num_tri {
			let old_tri = tri_new2old[tri] as usize;
			let tri_rel = tri_rel[old_tri];
			let instance_id = tri_rel.instance_id;

			#[cfg(feature = "test_thoroughly")]
			debug_assert!(tri_rel.face_id >= 0);

			face_id[tri] = I::lossy_from(tri_rel.face_id);
			for i in 0..3 {
				tri_verts[3 * tri + i] = I::lossy_from(self.tri.halfedge.start[3 * old_tri + i]);
			}

			if Some(instance_id) != last_id {
				unhandled_instance.remove(&instance_id);
				add_run(tri, self.instance_relation[instance_id as usize]);
				last_id = Some(instance_id);
			}
		}

		// Add runs for originals that did not contribute any faces to the output
		for instance_id in unhandled_instance {
			add_run(num_tri, self.instance_relation[instance_id as usize]);
		}

		run_index.push(I::lossy_from(3 * num_tri));

		// Early return for no props
		if prop_stride == 0 {
			let mut vert_properties = vec![F::lossy_from(0.0); 3 * num_vert];
			for i in 0..num_vert {
				let v = self.vert_pos[i];
				vert_properties[3 * i] = F::lossy_from(v.x);
				vert_properties[3 * i + 1] = F::lossy_from(v.y);
				vert_properties[3 * i + 2] = F::lossy_from(v.z);
			}

			return MeshGL {
				prop_stride: I::lossy_from(out_prop_stride),
				vert_properties,
				tri_verts,
				merge_from_vert: Vec::default(),
				merge_to_vert: Vec::default(),
				run_index,
				run_original_id,
				run_transform,
				run_flags,
				face_id,
				tolerance: F::lossy_from(tolerance),
			};
		}

		// Duplicate verts with different props
		let mut vert2idx = vec![-1; self.num_vert()];
		let mut vert_prop_pair: Vec<Vec<Vector2<i32>>> = vec![Vec::new(); self.num_vert()];
		let mut vert_properties = Vec::with_capacity(num_vert * out_prop_stride);

		let mut merge_from_vert = Vec::new();
		let mut merge_to_vert = Vec::new();

		for run in 0..run_original_id.len() {
			for tri in
				(usize::lossy_from(run_index[run]) / 3)..(usize::lossy_from(run_index[run + 1]) / 3)
			{
				for i in 0..3 {
					let prop = self.tri.halfedge.prop[3 * (tri_new2old[tri] as usize) + i];
					let vert = usize::lossy_from(tri_verts[3 * tri + i]);

					let bin = &mut vert_prop_pair[vert];
					let mut b_found = false;
					for b in bin.iter() {
						if b.x == prop {
							b_found = true;
							tri_verts[3 * tri + i] = I::lossy_from(b.y);
							break;
						}
					}

					if b_found {
						continue;
					}
					let idx = vert_properties.len() / out_prop_stride;
					tri_verts[3 * tri + i] = I::lossy_from(idx);
					bin.push(Vector2::new(prop, idx as i32));

					for p in 0..3 {
						vert_properties.push(F::lossy_from(self.vert_pos[vert][p]));
					}
					for p in 0..prop_stride {
						vert_properties.push(F::lossy_from(
							self.properties.data[(prop as usize) * prop_stride + p],
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
						let start = vert_properties.len() - out_prop_stride;
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
							normal = normal_transform(t)
								* (if (run_flags[run] & 1) != 0 { -1.0 } else { 1.0 })
								* normal;
						}
						normal = safe_normalize3(normal);
						for i in 0..3 {
							vert_properties[((start + 3 + i) as i32 + normal_idx) as usize] =
								F::lossy_from(normal[i]);
						}
					}

					if vert2idx[vert] == -1 {
						vert2idx[vert] = idx as i32;
					} else {
						merge_from_vert.push(I::lossy_from(idx));
						merge_to_vert.push(I::lossy_from(vert2idx[vert]));
					}
				}
			}
		}

		MeshGL {
			prop_stride: I::lossy_from(out_prop_stride),
			vert_properties,
			tri_verts,
			merge_from_vert,
			merge_to_vert,
			run_index,
			run_original_id,
			run_transform,
			run_flags,
			face_id,
			tolerance: F::lossy_from(tolerance),
		}
	}
}
