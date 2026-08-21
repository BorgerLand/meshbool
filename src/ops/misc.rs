use crate::mesh_relations::{
	InstanceRelation, TriRelation, all_instances_have_normals, reserve_original_id,
};
use crate::postprocessing as pp;
use crate::{Box3D, MeshBool, Precision, Properties, Triangles, TrianglesWIP};
use nalgebra::Matrix3x4;
use std::rc::Rc;

impl MeshBool {
	///This removes all relations (originalID, faceID, transform) to ancestor meshes
	///and this new Manifold is marked an original. It also recreates faces
	///- these don't get joined at boundaries where originalID changes, so the
	///reset may allow triangles of flat faces to be further collapsed with
	///Simplify().
	pub fn as_original(self) -> Self {
		drop(self.tri.normal);
		drop(self.tri.relation);

		let original_id = reserve_original_id();
		let mut tri_rel = vec![TriRelation::default(); self.tri.halfedge.num_tri()];
		let instance_rel = vec![InstanceRelation {
			original_id,
			transform: Matrix3x4::identity(),
			back_side: false,
			has_normals: all_instances_have_normals(&self.instance_relation),
			user_provided_face_id: false,
		}];

		let tri_normal = pp::set_normals_and_coplanar(
			&mut tri_rel,
			&instance_rel,
			&self.tri.halfedge,
			&self.vert_pos,
			self.precision.tolerance,
		);

		Self {
			original_id: Some(original_id),
			precision: self.precision,
			vert_pos: self.vert_pos,
			properties: self.properties,
			tri: Triangles {
				halfedge: self.tri.halfedge,
				normal: Rc::new(tri_normal),
				relation: Rc::new(tri_rel),
			},
			instance_relation: Rc::new(instance_rel),
			collider: self.collider,
		}
	}

	///Return a copy of the manifold with the set tolerance value.
	///This performs mesh simplification when the tolerance value is increased.
	pub fn set_tolerance(self, tolerance: f64) -> Self {
		self.set_tolerance_and_simplify(Some(tolerance), false)
	}

	///Return a copy of the manifold simplified to the given tolerance, but with its
	///actual tolerance value unchanged. If the tolerance is not given or is less
	///than the current tolerance, the current tolerance is used for simplification.
	///The result will contain a subset of the original verts and all surfaces will
	///have moved by less than tolerance.
	pub fn simplify(self, tolerance: Option<f64>) -> Self {
		self.set_tolerance_and_simplify(tolerance, true)
	}

	fn set_tolerance_and_simplify(self, tolerance: Option<f64>, mut simplify: bool) -> Self {
		let mut precision = self.precision;
		let requested_tolerance = tolerance.unwrap_or(precision.tolerance);
		let mut tri_rel = Rc::unwrap_or_clone(self.tri.relation);

		let tri_normal = if requested_tolerance > precision.tolerance {
			simplify = true;
			precision.tolerance = requested_tolerance;
			pp::set_normals_and_coplanar(
				&mut tri_rel,
				&self.instance_relation,
				&self.tri.halfedge,
				&self.vert_pos,
				requested_tolerance,
			)
		} else {
			if !simplify {
				// for reducing tolerance, we need to make sure it is still at least
				// equal to epsilon.
				precision.tolerance = precision.epsilon.max(requested_tolerance);
			}

			Rc::unwrap_or_clone(self.tri.normal)
		};

		if self.tri.halfedge.num_tri() == 0 {
			return Self::decimated(
				None,
				self.instance_relation,
				self.properties.stride,
				precision,
			);
		}

		let mut vert_pos = Rc::unwrap_or_clone(self.vert_pos);
		let mut properties = Rc::unwrap_or_clone(self.properties);
		let mut tri = TrianglesWIP {
			halfedge: Rc::unwrap_or_clone(self.tri.halfedge),
			normal: tri_normal,
			relation: tri_rel,
		};

		let collider = if simplify {
			pp::split_pinched_verts(&mut tri.halfedge, &mut vert_pos);
			pp::dedupe_edges(&mut tri, &mut vert_pos);
			pp::simplify_topology2(
				&mut tri,
				&mut vert_pos,
				&mut properties,
				&self.instance_relation,
				precision,
			);
			let bbox = Box3D::from_cloud(&vert_pos);
			let Some(collider) =
				pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
			else {
				return Self::decimated(None, self.instance_relation, properties.stride, precision);
			};
			collider
		} else {
			self.collider
		};

		Self {
			original_id: None,
			precision,
			vert_pos: Rc::new(vert_pos),
			properties: Rc::new(properties),
			tri: tri.into_rc(),
			instance_relation: self.instance_relation,
			collider,
		}
	}

	//useful if the mesh should retain some info about its origins
	//but lost all its geometry along the way and early exited
	pub(crate) fn decimated(
		original_id: Option<u32>,
		instance_relation: Rc<Vec<InstanceRelation>>,
		prop_stride: usize,
		precision: Precision,
	) -> Self {
		Self {
			original_id,
			precision,
			vert_pos: Rc::default(),
			properties: Rc::new(Properties {
				data: Vec::default(),
				stride: prop_stride,
			}),
			tri: Triangles::default(),
			instance_relation,
			collider: Rc::default(),
		}
	}
}
