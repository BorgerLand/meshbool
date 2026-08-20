use crate::halfedge::Halfedges;
use crate::ops::boolean::BooleanError;
use crate::ops::boolean::expression::CSGLeaf;
use crate::ops::transform::{flip_tri_pair, flip_tri_prop, flip_tri_start};
use crate::postprocessing as pp;
use crate::spatial::aabb::Box3D;
use crate::util::math::{K_PRECISION, mat3, mul_mat3x4, normal_transform, transform_normal};
use crate::util::vec_ext;
use crate::{MeshBool, Precision, Properties, TrianglesWIP};
use nalgebra::{Matrix3x4, Point3};
use std::iter;
use std::rc::Rc;

///Efficient union of a set of pairwise disjoint meshes.
pub(in crate::ops::boolean) fn boolean_disjoint_union(
	leaves: impl ExactSizeIterator<Item = CSGLeaf>,
) -> Result<MeshBool, BooleanError> {
	let mut prop_stride_new = 0;
	let mut bbox = Box3D::empty();
	let mut precision = Precision {
		epsilon: -1.0,
		tolerance: -1.0,
	};

	//treating nodes as an aosoa, rearrange the vecs
	//into arrays of arrays to allow early dropping
	let (
		vert_pos_aa,
		properties_aa,
		halfedge_aa,
		tri_normal_aa,
		tri_rel_aa,
		instance_rel_aa,
		transform_a,
	): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = leaves
		.into_iter()
		.map(|leaf| {
			let leaf_bbox = leaf.approximate_bbox();
			let leaf_old_scale = leaf.leaf.bounding_box().scale();
			let leaf_new_scale = leaf_bbox.scale();
			let mut leaf_epsilon =
				leaf.leaf.precision.epsilon * 1.0_f64.max(leaf_new_scale / leaf_old_scale);
			leaf_epsilon = leaf_epsilon.max(K_PRECISION * leaf_new_scale);
			if !leaf_epsilon.is_finite() {
				leaf_epsilon = -1.0;
			}

			prop_stride_new = prop_stride_new.max(leaf.leaf.properties.stride);
			bbox = bbox.union_box3(leaf_bbox);
			precision.epsilon = precision.epsilon.max(leaf_epsilon);
			precision.tolerance = precision.tolerance.max(leaf.leaf.precision.tolerance);

			(
				leaf.leaf.vert_pos,
				leaf.leaf.properties,
				leaf.leaf.tri.halfedge,
				leaf.leaf.tri.normal,
				leaf.leaf.tri.relation,
				leaf.leaf.instance_relation,
				leaf.pending_transform,
			)
		})
		.collect();

	let needs_flip_tris = |transform| mat3(transform).determinant() < 0.0;

	//try to produce one vec at a time to improve caching
	//behavior and allow for more early drops

	let pair_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		halfedge_aa
			.iter()
			.map(|halfedge| halfedge.start.len() as i32),
		0,
	));

	let halfedge_len = *pair_offsets.last().unwrap() as usize;

	let mut pair = Vec::with_capacity(halfedge_len);
	for (i, (pair_a, offset)) in halfedge_aa
		.iter()
		.map(|halfedge| &halfedge.pair)
		.zip(pair_offsets)
		.enumerate()
	{
		let pair_a = pair_a
			.as_chunks::<3>()
			.0
			.iter()
			.map(|tri| tri.map(|pair| pair + offset));

		if needs_flip_tris(transform_a[i]) {
			pair.extend(pair_a.flat_map(flip_tri_pair));
		} else {
			pair.extend(pair_a.flatten());
		}
	}

	let start_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		vert_pos_aa.iter().map(|vert_pos| vert_pos.len() as i32),
		0,
	));

	let vert_pos_len = *start_offsets.last().unwrap() as usize;

	let mut start = Vec::with_capacity(halfedge_len);
	for (i, (start_a, offset)) in halfedge_aa
		.iter()
		.map(|halfedge| &halfedge.start)
		.zip(start_offsets)
		.enumerate()
	{
		let start_a = start_a
			.as_chunks::<3>()
			.0
			.iter()
			.map(|tri| tri.map(|start| start + offset));

		if needs_flip_tris(transform_a[i]) {
			start.extend(start_a.flat_map(flip_tri_start));
		} else {
			start.extend(start_a.flatten());
		}
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

	let prop_offsets = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		properties_aa
			.iter()
			.map(|properties_a| num_prop_vert(properties_a) as i32),
		0,
	));

	let num_prop_verts = *prop_offsets.last().unwrap() as usize;

	let mut prop = Vec::with_capacity(halfedge_len);
	for (i, prop_a) in halfedge_aa
		.iter()
		.map(|halfedge_a| &halfedge_a.prop)
		.enumerate()
	{
		let offset = prop_offsets[i];
		let properties_a = &properties_aa[i];
		let prop_a = prop_a
			.as_chunks::<3>()
			.0
			.iter()
			.map(|tri| tri.map(|prop| offset + if properties_a.stride > 0 { prop } else { 0 }));

		if needs_flip_tris(transform_a[i]) {
			prop.extend(prop_a.flat_map(flip_tri_prop));
		} else {
			prop.extend(prop_a.flatten());
		}
	}

	let properties_data_len = prop_stride_new * num_prop_verts;
	let mut properties = Properties {
		data: Vec::with_capacity(properties_data_len),
		stride: prop_stride_new,
	};
	if properties_data_len > 0 {
		for (i, (properties_a, prop_offset)) in
			properties_aa.into_iter().zip(prop_offsets).enumerate()
		{
			let prop_stride_old = properties_a.stride;
			let num_prop_vert = num_prop_vert(&properties_a);
			for prop_vert in 0..num_prop_vert {
				properties
					.data
					.extend(&properties_a.data[prop_stride_old * prop_vert..][..prop_stride_old]);
				properties
					.data
					.extend(iter::repeat(0.0).take(prop_stride_new - prop_stride_old));
			}

			let transform = transform_a[i];
			if prop_stride_new >= 3 && transform != Matrix3x4::identity() {
				properties.eager_transform_prop_normals(
					&halfedge_aa[i],
					&instance_rel_aa[i],
					&tri_rel_aa[i],
					normal_transform(transform),
					num_prop_vert,
					prop_offset as usize,
				);
			}
		}
	}

	let halfedge = Halfedges { start, pair, prop };
	drop(halfedge_aa);

	let mut vert_pos = Vec::with_capacity(vert_pos_len);
	for (i, vert_pos_a) in vert_pos_aa.into_iter().enumerate() {
		let iter = vert_pos_a.iter();
		let transform = transform_a[i];
		if transform == Matrix3x4::identity() {
			vert_pos.extend(iter);
		} else {
			vert_pos.extend(iter.map(|&v| Point3::from(transform * v.coords.push(1.0))));
		}
	}

	let mut tri_normal = Vec::with_capacity(halfedge.num_tri());
	for (i, tri_normal_a) in tri_normal_aa.into_iter().enumerate() {
		let iter = tri_normal_a.iter();
		let transform = transform_a[i];
		if transform == Matrix3x4::identity() {
			tri_normal.extend(iter);
		} else {
			let transform = normal_transform(transform);
			tri_normal.extend(iter.map(|&n| transform_normal(transform, n)));
		}
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
	for (instance_rel_a, transform) in instance_rel_aa.into_iter().zip(transform_a) {
		instance_rel.extend(instance_rel_a.iter().map(|rel| {
			let mut rel = *rel;
			rel.transform = mul_mat3x4(transform, rel.transform);
			rel
		}));
	}

	let mut tri = TrianglesWIP {
		halfedge,
		normal: tri_normal,
		relation: tri_rel,
	};

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
