use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation};
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::BVHCollider;
use crate::{MeshBool, Precision, Properties, Triangles};
use borger_plugin_sdk::primitive::{DeserializeOopsy, PrimitiveSerDes, usize_to_32, usize32};
use nalgebra::{Matrix3x4, Point3, Vector3};
use std::rc::Rc;

#[cfg(feature = "server")]
pub fn meshbool_ser_tx(meshbool: &MeshBool, buffer: &mut Vec<u8>) {
	meshbool.precision.epsilon.ser_tx(buffer);
	meshbool.precision.tolerance.ser_tx(buffer);

	usize_to_32(meshbool.vert_pos.len()).ser_tx(buffer);
	for vert_pos in meshbool.vert_pos.iter() {
		for cmp in vert_pos.iter() {
			cmp.ser_tx(buffer);
		}
	}

	usize_to_32(meshbool.properties.data.len()).ser_tx(buffer);
	for property in meshbool.properties.data.iter() {
		property.ser_tx(buffer);
	}

	usize_to_32(meshbool.properties.stride).ser_tx(buffer);

	let halfedge_len = meshbool.tri.halfedge.len();
	usize_to_32(halfedge_len).ser_tx(buffer);
	for start in meshbool.tri.halfedge.start.iter() {
		start.ser_tx(buffer);
	}
	for pair in meshbool.tri.halfedge.pair.iter() {
		pair.ser_tx(buffer);
	}
	for prop in meshbool.tri.halfedge.prop.iter() {
		prop.ser_tx(buffer);
	}

	for normal in meshbool.tri.normal.iter() {
		for cmp in normal.iter() {
			cmp.ser_tx(buffer);
		}
	}

	for tri_rel in meshbool.tri.relation.iter() {
		tri_rel.instance_id.ser_tx(buffer);
		tri_rel.face_id.ser_tx(buffer);
	}

	usize_to_32(meshbool.instance_relation.len()).ser_tx(buffer);
	for mesh_id_transform in meshbool.instance_relation.iter() {
		mesh_id_transform.original_id.ser_tx(buffer);

		for cmp in mesh_id_transform.transform.iter() {
			cmp.ser_tx(buffer);
		}

		mesh_id_transform.back_side.ser_tx(buffer);
		mesh_id_transform.has_normals.ser_tx(buffer);
	}
}

#[cfg(feature = "client")]
pub fn meshbool_des_rx(
	buffer: &mut impl Iterator<Item = u8>,
) -> Result<MeshBool, DeserializeOopsy> {
	let epsilon = f64::des_rx(buffer)?;
	let tolerance = f64::des_rx(buffer)?;

	let mut vert_pos = vec![Point3::default(); usize32::des_rx(buffer)? as usize];
	for vert_pos in vert_pos.iter_mut() {
		for cmp in vert_pos.iter_mut() {
			*cmp = f64::des_rx(buffer)?;
		}
	}

	let mut properties_data = vec![0.0; usize32::des_rx(buffer)? as usize];
	for property in properties_data.iter_mut() {
		*property = f64::des_rx(buffer)?;
	}

	let properties_stride = usize32::des_rx(buffer)? as usize;

	let halfedge_len = usize32::des_rx(buffer)? as usize;
	let mut halfedge = Halfedges {
		start: vec![0; halfedge_len],
		pair: vec![0; halfedge_len],
		prop: vec![0; halfedge_len],
	};
	for start in halfedge.start.iter_mut() {
		*start = i32::des_rx(buffer)?;
	}
	for pair in halfedge.pair.iter_mut() {
		*pair = i32::des_rx(buffer)?;
	}
	for prop in halfedge.prop.iter_mut() {
		*prop = i32::des_rx(buffer)?;
	}

	let mut normal = vec![Vector3::default(); halfedge.num_tri()];
	for normal in normal.iter_mut() {
		for cmp in normal.iter_mut() {
			*cmp = f64::des_rx(buffer)?;
		}
	}

	let mut relation = vec![TriRelation::default(); halfedge.num_tri()];
	for tri_rel in relation.iter_mut() {
		tri_rel.instance_id = u32::des_rx(buffer)?;
		tri_rel.face_id = i32::des_rx(buffer)?;
	}

	let instance_relation_len = usize32::des_rx(buffer)? as usize;
	let mut instance_relation = Vec::with_capacity(instance_relation_len);
	for _ in 0..instance_relation_len {
		let original_id = u32::des_rx(buffer)?;

		let mut transform = Matrix3x4::zeros();
		for cmp in transform.iter_mut() {
			*cmp = f64::des_rx(buffer)?;
		}

		let back_side = bool::des_rx(buffer)?;
		let has_normals = bool::des_rx(buffer)?;

		instance_relation.push(InstanceRelation {
			original_id,
			transform,
			back_side,
			has_normals,
			user_provided_face_id: false,
		});
	}

	let collider = recompute_collider(&vert_pos, &halfedge);

	Ok(MeshBool {
		original_id: None,
		precision: Precision { epsilon, tolerance },
		vert_pos: Rc::new(vert_pos),
		properties: Rc::new(Properties {
			data: properties_data,
			stride: properties_stride,
		}),
		tri: Triangles {
			halfedge: Rc::new(halfedge),
			normal: Rc::new(normal),
			relation: Rc::new(relation),
		},
		instance_relation: Rc::new(instance_relation),
		collider,
	})
}

pub fn meshbool_ser_rollback(meshbool: &MeshBool, buffer: &mut Vec<u8>) {
	for mesh_id_transform in meshbool.instance_relation.iter().rev() {
		mesh_id_transform.has_normals.ser_rollback(buffer);
		mesh_id_transform.back_side.ser_rollback(buffer);

		for cmp in mesh_id_transform.transform.iter().rev() {
			cmp.ser_rollback(buffer);
		}

		mesh_id_transform.original_id.ser_rollback(buffer);
	}
	usize_to_32(meshbool.instance_relation.len()).ser_rollback(buffer);

	for tri_rel in meshbool.tri.relation.iter().rev() {
		tri_rel.face_id.ser_rollback(buffer);
		tri_rel.instance_id.ser_rollback(buffer);
	}

	for normal in meshbool.tri.normal.iter().rev() {
		for cmp in normal.iter().rev() {
			cmp.ser_rollback(buffer);
		}
	}

	for prop in meshbool.tri.halfedge.prop.iter().rev() {
		prop.ser_rollback(buffer);
	}
	for pair in meshbool.tri.halfedge.pair.iter().rev() {
		pair.ser_rollback(buffer);
	}
	for start in meshbool.tri.halfedge.start.iter().rev() {
		start.ser_rollback(buffer);
	}
	let halfedge_len = meshbool.tri.halfedge.len();
	usize_to_32(halfedge_len).ser_rollback(buffer);

	usize_to_32(meshbool.properties.stride).ser_rollback(buffer);

	for property in meshbool.properties.data.iter().rev() {
		property.ser_rollback(buffer);
	}
	usize_to_32(meshbool.properties.data.len()).ser_rollback(buffer);

	for vert_pos in meshbool.vert_pos.iter().rev() {
		for cmp in vert_pos.iter().rev() {
			cmp.ser_rollback(buffer);
		}
	}
	usize_to_32(meshbool.vert_pos.len()).ser_rollback(buffer);

	meshbool.precision.tolerance.ser_rollback(buffer);
	meshbool.precision.epsilon.ser_rollback(buffer);
}

pub fn meshbool_des_rollback(buffer: &mut Vec<u8>) -> Result<MeshBool, DeserializeOopsy> {
	let epsilon = f64::des_rollback(buffer)?;
	let tolerance = f64::des_rollback(buffer)?;

	let mut vert_pos = vec![Point3::default(); usize32::des_rollback(buffer)? as usize];
	for vert_pos in vert_pos.iter_mut() {
		for cmp in vert_pos.iter_mut() {
			*cmp = f64::des_rollback(buffer)?;
		}
	}

	let mut properties_data = vec![0.0; usize32::des_rollback(buffer)? as usize];
	for property in properties_data.iter_mut() {
		*property = f64::des_rollback(buffer)?;
	}

	let properties_stride = usize32::des_rollback(buffer)? as usize;

	let halfedge_len = usize32::des_rollback(buffer)? as usize;
	let mut halfedge = Halfedges {
		start: vec![0; halfedge_len],
		pair: vec![0; halfedge_len],
		prop: vec![0; halfedge_len],
	};
	for start in halfedge.start.iter_mut() {
		*start = i32::des_rollback(buffer)?;
	}
	for pair in halfedge.pair.iter_mut() {
		*pair = i32::des_rollback(buffer)?;
	}
	for prop in halfedge.prop.iter_mut() {
		*prop = i32::des_rollback(buffer)?;
	}

	let mut normal = vec![Vector3::default(); halfedge.num_tri()];
	for normal in normal.iter_mut() {
		for cmp in normal.iter_mut() {
			*cmp = f64::des_rollback(buffer)?;
		}
	}

	let mut relation = vec![TriRelation::default(); halfedge.num_tri()];
	for tri_rel in relation.iter_mut() {
		tri_rel.instance_id = u32::des_rollback(buffer)?;
		tri_rel.face_id = i32::des_rollback(buffer)?;
	}

	let instance_relation_len = usize32::des_rollback(buffer)? as usize;
	let mut instance_relation = Vec::with_capacity(instance_relation_len);
	for _ in 0..instance_relation_len {
		let original_id = u32::des_rollback(buffer)?;

		let mut transform = Matrix3x4::zeros();
		for cmp in transform.iter_mut() {
			*cmp = f64::des_rollback(buffer)?;
		}

		let back_side = bool::des_rollback(buffer)?;
		let has_normals = bool::des_rollback(buffer)?;

		instance_relation.push(InstanceRelation {
			original_id,
			transform,
			back_side,
			has_normals,
			user_provided_face_id: false,
		});
	}

	let collider = recompute_collider(&vert_pos, &halfedge);

	Ok(MeshBool {
		original_id: None,
		precision: Precision { epsilon, tolerance },
		vert_pos: Rc::new(vert_pos),
		properties: Rc::new(Properties {
			data: properties_data,
			stride: properties_stride,
		}),
		tri: Triangles {
			halfedge: Rc::new(halfedge),
			normal: Rc::new(normal),
			relation: Rc::new(relation),
		},
		instance_relation: Rc::new(instance_relation),
		collider,
	})
}

fn recompute_collider(vert_pos: &[Point3<f64>], halfedge: &Halfedges) -> Rc<BVHCollider> {
	let (tri_box, tri_morton) =
		get_tri_box_morton(halfedge, vert_pos, Some(Box3D::from_cloud(vert_pos)));
	Rc::new(BVHCollider::new(&tri_box, &tri_morton.unwrap()))
}
