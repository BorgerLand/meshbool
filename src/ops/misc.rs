use crate::mesh_relations::{
	InstanceRelation, TriRelation, all_instances_have_normals, reserve_original_id,
};
use crate::postprocessing as pp;
use crate::postprocessing::sort::{gather_tris, reindex_verts};
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::disjoint_sets::DisjointSets;
use crate::util::hash_table::DeterministicMap;
use crate::util::vec_ext;
use crate::{Box3D, MeshBool, Precision, Properties, Triangles};
use nalgebra::{Matrix3x4, Point3};

impl MeshBool {
	///This removes all relations (originalID, faceID, transform) to ancestor meshes
	///and this new Manifold is marked an original. It also recreates faces
	///- these don't get joined at boundaries where originalID changes, so the
	///reset may allow triangles of flat faces to be further collapsed with
	///Simplify().
	pub fn as_original(&self) -> Self {
		let original_id = reserve_original_id();
		let mut tri_rel = vec![TriRelation::default(); self.num_tri()];
		pp::set_normals_and_coplanar(
			&mut tri_rel,
			&self.tri.halfedge,
			&self.vert_pos,
			self.precision.tolerance,
		);

		Self {
			original_id: Some(original_id),
			precision: self.precision,
			vert_pos: self.vert_pos.clone(),
			properties: self.properties.clone(),
			tri: Triangles {
				halfedge: self.tri.halfedge.clone(),
				normal: self.tri.normal.clone(),
				relation: tri_rel,
			},
			instance_relation: [(
				0_u32,
				InstanceRelation {
					original_id,
					transform: Matrix3x4::identity(),
					back_side: false,
					has_normals: all_instances_have_normals(&self.instance_relation),
				},
			)]
			.into(),
			collider: self.collider.clone(),
		}
	}

	// This operation returns a vector of Manifolds that are topologically
	// disconnected. If everything is connected, the vector is length one,
	// containing a copy of the original. It is the inverse operation of Compose().
	pub fn decompose(&self) -> Vec<Self> {
		let uf = DisjointSets::new(self.num_vert());
		for edge in 0..self.tri.halfedge.len() as i32 {
			if self.tri.halfedge.is_forward(edge) {
				uf.unite(
					self.tri.halfedge.start(edge) as usize,
					self.tri.halfedge.end(edge) as usize,
				);
			}
		}

		let (vert_label, num_components) = uf.connected_components();

		if num_components == 1 {
			return vec![self.clone()];
		}

		let num_vert = self.num_vert();
		let mut meshes: Vec<Self> = Vec::with_capacity(num_components);
		for i in 0..num_components as i32 {
			let mut vert_new2old: Vec<i32> = unsafe { vec_ext::uninit(num_vert) };
			let n_vert = vec_ext::copy_if(0..num_vert as i32, &mut vert_new2old, |v| {
				vert_label[v as usize] == i
			});
			let mut vert_pos = vec![Point3::default(); n_vert];
			vert_new2old.resize(n_vert, Default::default());
			vec_ext::gather(&vert_new2old, &self.vert_pos, &mut vert_pos);

			let mut face_new2old: Vec<i32> = Vec::with_capacity(self.num_tri());
			let halfedge = &self.tri.halfedge;
			for face in 0..self.num_tri() as i32 {
				if vert_label[halfedge.start(3 * face) as usize] == i {
					face_new2old.push(face);
				}
			}

			if face_new2old.is_empty() {
				continue;
			}

			let mut tri = gather_tris(&self.tri, &face_new2old);
			reindex_verts(&mut tri.halfedge, &vert_new2old, self.num_vert());
			let mut properties = self.properties.clone();
			let bbox = Box3D::from_cloud(&vert_pos);
			let collider =
				pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
					.unwrap();

			meshes.push(Self {
				original_id: None,
				precision: self.precision, // inherit original object's precision
				vert_pos,
				properties,
				tri,
				instance_relation: self.instance_relation.clone(),
				collider,
			});
		}
		meshes
	}

	///Return a copy of the manifold with the set tolerance value.
	///This performs mesh simplification when the tolerance value is increased.
	pub fn set_tolerance(&self, tolerance: f64) -> Self {
		self.set_tolerance_and_simplify(Some(tolerance), false)
	}

	///Return a copy of the manifold simplified to the given tolerance, but with its
	///actual tolerance value unchanged. If the tolerance is not given or is less
	///than the current tolerance, the current tolerance is used for simplification.
	///The result will contain a subset of the original verts and all surfaces will
	///have moved by less than tolerance.
	pub fn simplify(&self, tolerance: Option<f64>) -> Self {
		self.set_tolerance_and_simplify(tolerance, true)
	}

	fn set_tolerance_and_simplify(&self, tolerance: Option<f64>, mut simplify: bool) -> Self {
		let mut precision = self.precision;
		let tolerance = tolerance.unwrap_or(precision.tolerance);
		let mut tri_rel = self.tri.relation.clone();

		if tolerance > precision.tolerance {
			precision.tolerance = tolerance;
			pp::set_normals_and_coplanar(
				&mut tri_rel,
				&self.tri.halfedge,
				&self.vert_pos,
				tolerance,
			);
			simplify = true;
		} else if !simplify {
			// for reducing tolerance, we need to make sure it is still at least
			// equal to epsilon.
			precision.tolerance = precision.epsilon.max(tolerance);
		}

		if self.is_empty() {
			return Self::decimated(
				None,
				self.instance_relation.clone(),
				self.properties.stride,
				precision,
			);
		}

		let mut vert_pos = self.vert_pos.clone();
		let mut properties = self.properties.clone();
		let mut tri = Triangles {
			halfedge: self.tri.halfedge.clone(),
			normal: self.tri.normal.clone(),
			relation: tri_rel,
		};

		let collider = if simplify {
			pp::split_pinched_verts(&mut tri.halfedge, &mut vert_pos);
			pp::dedupe_edges(&mut tri, &mut vert_pos);
			pp::collapse_short_edges(
				&mut tri.halfedge,
				&mut vert_pos,
				&tri.normal,
				&tri.relation,
				self.properties.stride,
				precision,
				0,
			);
			pp::collapse_colinear_edges(
				&mut tri.halfedge,
				&mut vert_pos,
				&tri.normal,
				&tri.relation,
				properties.stride,
				precision.epsilon,
				0,
			);
			pp::swap_degenerates(&mut tri, &mut vert_pos, &mut properties, precision, 0);
			let bbox = Box3D::from_cloud(&vert_pos);
			let Some(collider) =
				pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
			else {
				return Self::decimated(
					None,
					self.instance_relation.clone(),
					properties.stride,
					precision,
				);
			};
			collider
		} else {
			self.collider.clone()
		};

		Self {
			original_id: None,
			precision,
			vert_pos,
			properties,
			tri,
			instance_relation: self.instance_relation.clone(),
			collider,
		}
	}

	//useful if the mesh should retain some info about its origins
	//but lost all its geometry along the way and early exited
	pub(crate) fn decimated(
		original_id: Option<u32>,
		instance_relation: DeterministicMap<u32, InstanceRelation>,
		prop_stride: usize,
		precision: Precision,
	) -> Self {
		Self {
			original_id,
			precision,
			vert_pos: Vec::default(),
			properties: Properties {
				data: Vec::default(),
				stride: prop_stride,
			},
			tri: Triangles::default(),
			instance_relation,
			collider: BVHCollider::default(),
		}
	}
}
