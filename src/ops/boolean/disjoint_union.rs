use crate::halfedge::Halfedges;
use crate::ops::boolean::BooleanError;
use crate::postprocessing as pp;
use crate::spatial::aabb::Box3D;
use crate::util::vec_ext;
use crate::{MeshBool, Precision, Properties, TrianglesWIP};
use std::iter;
use std::rc::Rc;

///Efficient union of a set of pairwise disjoint meshes.
pub fn boolean_disjoint_union(
	nodes: impl ExactSizeIterator<Item = MeshBool>,
) -> Result<MeshBool, BooleanError> {
	let mut prop_stride_new = 0;
	let mut bbox = Box3D::empty();
	let mut precision = Precision {
		epsilon: -1.0,
		tolerance: -1.0,
	};

	//treating nodes as an aosoa, rearrange the vecs
	//into arrays of arrays to allow early dropping
	let (vert_pos_aa, properties_aa, halfedge_aa, tri_normal_aa, tri_rel_aa, instance_rel_aa): (
		Vec<_>,
		Vec<_>,
		Vec<_>,
		Vec<_>,
		Vec<_>,
		Vec<_>,
	) = nodes
		.into_iter()
		.map(|node| {
			prop_stride_new = prop_stride_new.max(node.properties.stride);
			bbox = bbox.union_box3(node.bounding_box());

			precision.epsilon = precision.epsilon.max(node.precision.epsilon);
			precision.tolerance = precision.tolerance.max(node.precision.tolerance);

			(
				node.vert_pos,
				node.properties,
				node.tri.halfedge,
				node.tri.normal,
				node.tri.relation,
				node.instance_relation,
			)
		})
		.collect();

	let pair_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		halfedge_aa
			.iter()
			.map(|halfedge| halfedge.start.len() as i32),
		0,
	));

	let halfedge_len = *pair_offsets.last().unwrap() as usize;

	let mut pair = Vec::with_capacity(halfedge_len);
	for (pair_a, offset) in halfedge_aa
		.iter()
		.map(|halfedge| &halfedge.pair)
		.zip(pair_offsets)
	{
		pair.extend(pair_a.iter().map(|pair| pair + offset));
	}

	let start_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		vert_pos_aa.iter().map(|vert_pos| vert_pos.len() as i32),
		0,
	));

	let vert_pos_len = *start_offsets.last().unwrap() as usize;

	let mut start = Vec::with_capacity(halfedge_len);
	for (start_a, offset) in halfedge_aa
		.iter()
		.map(|halfedge| &halfedge.start)
		.zip(start_offsets)
	{
		start.extend(start_a.iter().map(|start| start + offset));
	}

	//a node carrying no properties still gets a single, zeroed property vert for
	//all of its halfedges to point at, in case another node does carry some
	let num_prop_vert = |properties: &Properties| {
		if properties.stride == 0 {
			1
		} else {
			properties.data.len() / properties.stride
		}
	};

	let mut prop_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		properties_aa
			.iter()
			.map(|properties_a| num_prop_vert(properties_a) as i32),
		0,
	));

	let num_prop_verts = prop_offsets.pop().unwrap() as usize;

	let mut prop = Vec::with_capacity(halfedge_len);
	for (i, offset) in prop_offsets.into_iter().enumerate() {
		let prop_a = &halfedge_aa[i].prop;
		let properties_a = &properties_aa[i];
		prop.extend(
			prop_a
				.iter()
				.map(|&prop| offset + if properties_a.stride > 0 { prop } else { 0 }),
		);
	}

	let halfedge = Halfedges { start, pair, prop };
	drop(halfedge_aa);

	let mut vert_pos = Vec::with_capacity(vert_pos_len);
	for vert_pos_a in vert_pos_aa {
		vert_pos.extend(vert_pos_a.iter());
	}

	let properties_data_len = prop_stride_new * num_prop_verts;
	let mut properties_data = Vec::with_capacity(properties_data_len);
	if properties_data_len > 0 {
		for properties_a in properties_aa.into_iter() {
			let prop_stride_old = properties_a.stride;
			for prop_vert in 0..num_prop_vert(&properties_a) {
				properties_data.extend(
					properties_a.data[prop_stride_old * prop_vert..][..prop_stride_old]
						.iter()
						.copied(),
				);
				properties_data.extend(iter::repeat(0.0).take(prop_stride_new - prop_stride_old));
			}
		}
	}

	let mut properties = Properties {
		data: properties_data,
		stride: prop_stride_new,
	};

	let mut tri_normal = Vec::with_capacity(halfedge.num_tri());
	for tri_normal_a in tri_normal_aa.into_iter() {
		tri_normal.extend(tri_normal_a.iter());
	}

	let instance_rel_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		instance_rel_aa
			.iter()
			.map(|instance_rel| instance_rel.len() as u32),
		0,
	));

	let instance_rel_len = *instance_rel_offsets.last().unwrap() as usize;

	let mut tri_rel = Vec::with_capacity(halfedge.num_tri());
	for (tri_rel_a, offset) in tri_rel_aa.into_iter().zip(instance_rel_offsets) {
		tri_rel.extend(tri_rel_a.iter().map(|tri_rel| {
			let mut tri_rel = *tri_rel;
			tri_rel.instance_id += offset;
			tri_rel
		}));
	}

	let mut instance_rel = Vec::with_capacity(instance_rel_len);
	for instance_rel_a in instance_rel_aa.into_iter() {
		instance_rel.extend(instance_rel_a.iter());
	}

	let mut tri = TrianglesWIP {
		halfedge,
		normal: tri_normal,
		relation: tri_rel,
	};

	let epsilon = precision.epsilon;

	pp::split_pinched_verts(&mut tri.halfedge, &mut vert_pos);
	pp::dedupe_edges(&mut tri, &mut vert_pos);
	pp::collapse_short_edges(
		&mut tri.halfedge,
		&mut vert_pos,
		&tri.normal,
		&tri.relation,
		&instance_rel,
		prop_stride_new,
		epsilon,
		precision.tolerance,
		0,
	);
	pp::swap_degenerates(
		&mut tri,
		&mut vert_pos,
		&mut properties,
		&instance_rel,
		epsilon,
		precision.tolerance,
		0,
	);

	let Some(collider) =
		pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
	else {
		return Ok(MeshBool::decimated(
			None,
			Rc::new(instance_rel),
			prop_stride_new,
			precision,
		));
	};

	Ok(MeshBool {
		original_id: None,
		precision,
		vert_pos: Rc::new(vert_pos),
		properties: Rc::new(properties),
		tri: tri.into_rc(),
		instance_relation: Rc::new(instance_rel),
		collider,
	})
}
