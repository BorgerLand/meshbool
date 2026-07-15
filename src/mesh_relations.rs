use crate::util::math::inverse_normal_transform;
use nalgebra::{Matrix3, Matrix3x4};
use std::sync::atomic::{AtomicU32, Ordering};

static ORIGINAL_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

///System for tracking the origin story of every triangle in the output, allowing
///rendering materials and properties to be re-applied to the output
#[derive(Copy, Clone, Debug)]
pub struct TriRelation {
	///Each time an instance of an original mesh makes an appearance somewhere in this
	///mesh, the instance receives its own LOCALLY unique (to this mesh) instance ID.
	///Not requiring global uniqueness allows faster cloning.
	pub instance_id: u32,
	/// If set as an input of MeshGL, it is passed along unchanged. This is how
	/// the user can tell us not to collapse certain edges: those that divide
	/// difference faceIDs. If not set, this is always -1.
	pub face_id: i32,
	/// Triangles with the same coplanar ID are coplanar. Starts as a canonical
	/// triangle index, but after boolean operations it may refer to a triangle
	/// that is no longer present in this mesh.
	pub coplanar_id: i32,
}

impl Default for TriRelation {
	fn default() -> Self {
		Self {
			instance_id: 0,
			face_id: -1,
			coplanar_id: -1,
		}
	}
}

impl TriRelation {
	pub fn same_face(&self, other: &TriRelation) -> bool {
		self.instance_id == other.instance_id
			&& self.coplanar_id == other.coplanar_id
			&& self.face_id == other.face_id
	}
}

#[derive(Clone, Copy, Debug)]
pub struct InstanceRelation {
	///Each mesh that is used as an input to this library, constructed either by
	///loading a MeshGL or procedurally generating a primitive shape, is considered an
	///"original" mesh and receives a GLOBALLY unique original ID. Not all meshes are
	///originals, eg. the output of a boolean operation or some other transformation
	///applied to an original to make a newly derived mesh. These derived meshes do not
	///have their own original ID; they are merely a composite of 1 or more originals.
	pub original_id: u32,
	pub transform: Matrix3x4<f64>,
	pub back_side: bool,
	///True when this meshID's contribution to properties_ slots 0..2 holds
	///world-frame vertex normals (set by CalculateNormals at slot 0). Carries
	///through Transforms and Booleans. Exported as runFlags bit 1.
	pub has_normals: bool,
}

impl InstanceRelation {
	pub fn get_inverse_normal_transform(&self) -> Matrix3<f64> {
		inverse_normal_transform(self.transform) * if self.back_side { -1.0 } else { 1.0 }
	}
}

pub fn reserve_original_id() -> u32 {
	ORIGINAL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

///True iff the meshID owning `tri` has hasNormals set. Returns false when
///the meshID isn't in meshRelation_.meshIDtransform (treat as no-normals).
pub fn tri_has_normals(instance_rel: &[InstanceRelation], tri: TriRelation) -> bool {
	instance_rel
		.get(tri.instance_id as usize)
		.map(|it| it.has_normals)
		.unwrap_or(false)
}

pub fn all_instances_have_normals(instance_rel: &Vec<InstanceRelation>) -> bool {
	instance_rel.iter().all(|instance| instance.has_normals)
}
