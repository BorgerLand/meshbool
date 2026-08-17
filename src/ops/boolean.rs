use crate::ops::boolean::disjoint_union::boolean_disjoint_union;
use crate::postprocessing as pp;
use crate::spatial::aabb::Overlap;
use crate::util::vec_ext;
use crate::{Box3D, Triangles};
use crate::{MeshBool, Precision, Properties};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::rc::Rc;

#[cfg(feature = "test_thoroughly")]
use crate::test::{get_intermediate_checks, get_self_intersection_checks};

pub mod expression;

mod construct;
mod disjoint_union;
mod face2tri;
mod intersect;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BooleanError {
	ResultTooLarge,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OpType {
	Union,
	Difference,
	Intersection,
}

///	The central operation of this library: the Boolean combines two manifolds
///	into another by calculating their intersections and removing the unused
///	portions.
///	[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid)
///	inputs will produce &epsilon;-valid output. &epsilon;-invalid input may fail
///	triangulation.
///
///	These operations are optimized to produce nearly-instant results if either
///	input is empty or their bounding boxes do not overlap.
///
///	@param second The other Manifold.
///	@param op The type of operation to perform.
///
/// https://github.com/elalish/manifold/blob/master/docs/RobustBoolean.pdf
///
/// [Smith 2009] - Cambridge Technical Report 766
///
/// Towards robust inexact geometric computation
pub fn boolean(in_p: MeshBool, op: OpType, in_q: MeshBool) -> Result<MeshBool, BooleanError> {
	let overlapping = in_p
		.collider
		.get_bounding_box()
		.overlaps(in_q.collider.get_bounding_box());

	let num_tri_p = in_p.num_tri();
	let num_tri_q = in_q.num_tri();

	//handle this early exit scenario extra early before moving out of p/q
	if num_tri_p > 0 && num_tri_q > 0 && !overlapping && op == OpType::Union {
		return boolean_disjoint_union([in_p, in_q].into_iter());
	}

	//outR.epsilon_ = std::max(inP_.epsilon_, inQ_.epsilon_);
	//outR.tolerance_ = std::max(inP_.tolerance_, inQ_.tolerance_);
	let precision = Precision {
		epsilon: in_p.precision.epsilon.max(in_q.precision.epsilon),
		tolerance: in_p.precision.tolerance.max(in_q.precision.tolerance),
	};
	let epsilon = precision.epsilon;
	let tolerance = precision.tolerance;
	let prop_stride = in_p.properties.stride.max(in_q.properties.stride);
	let invert_q = op == OpType::Difference;
	let instance_id_offset_q = in_p.instance_relation.len() as u32;
	let mut instance_rel = Rc::unwrap_or_clone(in_p.instance_relation);
	instance_rel.extend(in_q.instance_relation.iter().map(|rel| {
		let mut rel = *rel;
		rel.back_side ^= invert_q;
		rel
	}));
	drop(in_q.instance_relation);
	let instance_rel = Rc::new(instance_rel);

	if num_tri_p == 0 {
		if num_tri_q > 0 && op == OpType::Union {
			//clone q
			return Ok(MeshBool {
				original_id: None,
				tri: Triangles {
					relation: Rc::new(
						in_q.tri
							.relation
							.iter()
							.map(|tri_rel| {
								let mut tri_rel = *tri_rel;
								tri_rel.instance_id += instance_id_offset_q;
								tri_rel
							})
							.collect(),
					),
					..in_q.tri
				},
				instance_relation: instance_rel,
				..in_q
			});
		}

		return Ok(MeshBool::decimated(
			None,
			instance_rel,
			prop_stride,
			precision,
		));
	} else if num_tri_q == 0 {
		if op == OpType::Intersection {
			return Ok(MeshBool::decimated(
				None,
				instance_rel,
				prop_stride,
				precision,
			));
		}

		return Ok(MeshBool {
			original_id: None,
			instance_relation: instance_rel,
			..in_p
		});
	} else if !overlapping {
		if op == OpType::Difference {
			return Ok(MeshBool {
				original_id: None,
				instance_relation: instance_rel,
				..in_p
			});
		} else if op == OpType::Intersection {
			return Ok(MeshBool::decimated(
				None,
				instance_rel,
				prop_stride,
				precision,
			));
		}
	}

	// Symbolic perturbation:
	// Union -> expand inP, expand inQ
	// Difference, Intersection -> contract inP, expand inQ
	// Technically Intersection should contract inQ, but doing it this way makes
	// Split faster and any suboptimal cases seem pretty rare.
	let expand_p = op == OpType::Union;
	const INT_MAX_SZ: usize = i32::MAX as usize;

	let vert_normal_p = MeshBool::calculate_vert_normals_internal(
		&in_p.vert_pos,
		&in_p.tri.halfedge,
		&in_p.tri.normal,
	);
	let vert_normal_q = MeshBool::calculate_vert_normals_internal(
		&in_q.vert_pos,
		&in_q.tri.halfedge,
		&in_q.tri.normal,
	);

	// Level 3
	// Build up the intersection of the edges and triangles, keeping only those
	// that intersect, and record the direction the edge is passing through the
	// triangle.
	let xv12 = intersect::intersect12::<true>(
		&in_p.vert_pos,
		&in_p.tri.halfedge,
		&in_p.tri.normal,
		&in_p.collider,
		&vert_normal_p,
		&in_q.vert_pos,
		&in_q.tri.halfedge,
		&in_q.tri.normal,
		&in_q.collider,
		&vert_normal_q,
		expand_p,
	);

	if xv12.x12.len() > INT_MAX_SZ {
		return Err(BooleanError::ResultTooLarge);
	}

	let xv21 = intersect::intersect12::<false>(
		&in_p.vert_pos,
		&in_p.tri.halfedge,
		&in_p.tri.normal,
		&in_p.collider,
		&vert_normal_p,
		&in_q.vert_pos,
		&in_q.tri.halfedge,
		&in_q.tri.normal,
		&in_q.collider,
		&vert_normal_q,
		expand_p,
	);

	if xv21.x12.len() > INT_MAX_SZ {
		return Err(BooleanError::ResultTooLarge);
	}

	// Compute winding numbers of all vertices using flood fill
	// Vertices on the same connected component have the same winding number
	//perform w30 first, with the assumption that it will cause the
	//larger of the 2 bvh colliders to drop sooner
	let w30 = intersect::winding03::<false>(
		&in_q.vert_pos,
		&in_q.tri.halfedge,
		&vert_normal_q,
		&in_p.vert_pos,
		&in_p.tri.halfedge,
		&in_p.tri.normal,
		in_p.collider,
		&vert_normal_p,
		&xv21.p1q2,
		expand_p,
	);
	let w03 = intersect::winding03::<true>(
		&in_p.vert_pos,
		&in_p.tri.halfedge,
		&vert_normal_p,
		&in_q.vert_pos,
		&in_q.tri.halfedge,
		&in_q.tri.normal,
		in_q.collider,
		&vert_normal_q,
		&xv12.p1q2,
		expand_p,
	);

	debug_assert!(
		expand_p == (op == OpType::Union),
		"Result op type not compatible with constructor op type."
	);

	drop(vert_normal_p);
	drop(vert_normal_q);

	let c1 = if op == OpType::Intersection { 0 } else { 1 };
	let c2 = if op == OpType::Union { 1 } else { 0 };
	let c3 = if op == OpType::Intersection { 1 } else { -1 };

	// Convert winding numbers to inclusion values based on operation type.
	//reuses existing allocation
	let i12 = Vec::from_iter(xv12.x12.into_iter().map(|v| c3 * v));
	let i21 = Vec::from_iter(xv21.x12.into_iter().map(|v| c3 * v));
	let i03 = Vec::from_iter(w03.into_iter().map(|v| c1 + (c3 as i32) * v));
	let i30 = Vec::from_iter(w30.into_iter().map(|v| c2 + (c3 as i32) * v));

	let v_p2r = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		i03.iter().map(|v| v.abs()),
		0,
	));
	let mut num_vert_r = *v_p2r.last().unwrap() as usize;
	let n_pv = num_vert_r;

	let v_q2r = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		i30.iter().map(|v| v.abs()),
		num_vert_r as i32,
	));
	num_vert_r = *v_q2r.last().unwrap() as usize;
	let n_qv = num_vert_r - n_pv;

	let v12_r = if i12.len() == 0 {
		Vec::new()
	} else {
		let v12_r = Vec::from_iter(vec_ext::exclusive_scan_with_total(
			i12.iter().map(|v| v.abs() as i32),
			num_vert_r as i32,
		));
		num_vert_r = *v12_r.last().unwrap() as usize;
		v12_r
	};

	let n12 = num_vert_r - n_pv - n_qv; //new verts from edgesP -> facesQ

	let v21_r = if i21.len() == 0 {
		Vec::new()
	} else {
		let v21_r = Vec::from_iter(vec_ext::exclusive_scan_with_total(
			i21.iter().map(|v| v.abs() as i32),
			num_vert_r as i32,
		));
		num_vert_r = *v21_r.last().unwrap() as usize;
		v21_r
	};

	let n21 = num_vert_r - n_pv - n_qv - n12; //new verts from facesP -> edgesQ

	if num_vert_r == 0 {
		return Ok(MeshBool::decimated(
			None,
			instance_rel,
			prop_stride,
			precision,
		));
	}

	let mut vert_pos = unsafe { vec_ext::uninit(num_vert_r as usize) };
	// Add vertices, duplicating for inclusion numbers not in [-1, 1].
	// Retained vertices from P and Q:
	let mut retain_p = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i03,
		vert_r: &v_p2r,
		vert_pos_a: &in_p.vert_pos,
	};
	for vert in 0..in_p.vert_pos.len() {
		retain_p.call(vert);
	}
	let mut retain_q = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i30,
		vert_r: &v_q2r,
		vert_pos_a: &in_q.vert_pos,
	};
	for vert in 0..in_q.vert_pos.len() {
		retain_q.call(vert);
	}
	// New vertices created from intersections:
	let mut new12 = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i12,
		vert_r: &v12_r,
		vert_pos_a: &xv12.v12,
	};
	for vert in 0..i12.len() {
		new12.call(vert);
	}
	drop(xv12.v12);
	let mut new21 = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i21,
		vert_r: &v21_r,
		vert_pos_a: &xv21.v12,
	};
	for vert in 0..i21.len() {
		new21.call(vert);
	}
	drop(xv21.v12);

	// Build up new polygonal faces from triangle intersections. At this point the
	// calculation switches from parallel to serial.

	// Level 3

	// This key is the forward halfedge index of P or Q. Only includes intersected
	// edges.
	let mut edges_p = FxHashMap::default(); //capacity is unpredictable
	let mut edges_q = FxHashMap::with_capacity_and_hasher(xv21.p1q2.len() / 2, FxBuildHasher);
	// This key is the face index of <P, Q>
	let mut edges_r = FxHashMap::with_capacity_and_hasher((n12 + n21) as usize, FxBuildHasher);

	construct::add_new_edge_verts(
		&mut edges_p,
		&mut edges_r,
		&xv12.p1q2,
		&i12,
		v12_r,
		&in_p.tri.halfedge,
		true,
	);
	construct::add_new_edge_verts(
		&mut edges_q,
		&mut edges_r,
		&xv21.p1q2,
		&i21,
		v21_r,
		&in_q.tri.halfedge,
		false,
	);

	// Level 4
	let sides_per_face_pq = construct::size_sides_per_face_pq(
		&in_p.tri.halfedge,
		&in_q.tri.halfedge,
		&i03,
		&i30,
		i12,
		i21,
		xv12.p1q2,
		xv21.p1q2,
	);
	let face_pq2r = Vec::from_iter(vec_ext::exclusive_scan_with_total(
		sides_per_face_pq.iter().map(|&x| if x > 0 { 1 } else { 0 }),
		0,
	));
	let num_face_r = *face_pq2r.last().unwrap() as usize;
	let face_normal = construct::size_face_normal(
		in_p.tri.normal,
		in_q.tri.normal,
		&sides_per_face_pq,
		num_face_r,
		invert_q,
	);
	let mut face_edge = Vec::with_capacity(num_face_r);
	face_edge.extend(vec_ext::exclusive_scan_with_total(
		sides_per_face_pq.into_iter().filter(|&v| v > 0),
		0,
	));

	let num_halfedge_r = *face_edge.last().unwrap() as usize;
	if num_halfedge_r == 0 {
		return Ok(MeshBool::decimated(
			None,
			instance_rel,
			prop_stride,
			precision,
		));
	}

	// This gets incremented for each halfedge that's added to a face so that the
	// next one knows where to slot in.
	let mut face_ptr_r = face_edge.clone();
	// Intersected halfedges are marked false.
	let mut whole_halfedge_p = vec![true; in_p.tri.halfedge.len()];
	let mut whole_halfedge_q = vec![true; in_q.tri.halfedge.len()];

	// The face_rel contains the data that will become triRel once the faces
	// are triangulated.
	let mut face_rel = unsafe { vec_ext::uninit(num_halfedge_r) };
	// Note that we are working with Vec<Halfedge> instead of Halfedges here,
	// since the faces can be arbitrary polygons before feeding into the
	// triangulator. prop_vert is meaningless until after create_properties
	let mut face_halfedges = unsafe { vec_ext::uninit(num_halfedge_r) };

	construct::append_partial_edges(
		&mut vert_pos,
		&mut face_halfedges,
		&mut face_ptr_r,
		&mut face_rel,
		&in_p.vert_pos,
		&in_p.tri.halfedge,
		edges_p,
		&mut whole_halfedge_p,
		&i03,
		&v_p2r,
		&face_pq2r[0..num_tri_p],
		&in_p.tri.relation,
		0,
		prop_stride > 0,
	);
	construct::append_partial_edges(
		&mut vert_pos,
		&mut face_halfedges,
		&mut face_ptr_r,
		&mut face_rel,
		&in_q.vert_pos,
		&in_q.tri.halfedge,
		edges_q,
		&mut whole_halfedge_q,
		&i30,
		&v_q2r,
		&face_pq2r[num_tri_p..],
		&in_q.tri.relation,
		instance_id_offset_q,
		prop_stride > 0,
	);

	construct::append_new_edges(
		&mut vert_pos,
		&mut face_halfedges,
		&mut face_ptr_r,
		&mut face_rel,
		edges_r,
		&face_pq2r,
		num_tri_p,
		&in_p.tri.relation,
		&in_q.tri.relation,
		instance_id_offset_q,
		prop_stride > 0,
	);

	construct::append_whole_edges(
		&mut face_ptr_r,
		&mut face_halfedges,
		&mut face_rel,
		&in_p.tri.halfedge,
		whole_halfedge_p,
		i03,
		v_p2r,
		&face_pq2r[..num_tri_p],
		&in_p.tri.relation,
		0,
		prop_stride > 0,
	);
	construct::append_whole_edges(
		&mut face_ptr_r,
		&mut face_halfedges,
		&mut face_rel,
		&in_q.tri.halfedge,
		whole_halfedge_q,
		i30,
		v_q2r,
		&face_pq2r[num_tri_p..],
		&in_q.tri.relation,
		instance_id_offset_q,
		prop_stride > 0,
	);

	drop(face_ptr_r);
	drop(face_pq2r);

	// Level 6
	let mut tri = face2tri::face2tri(
		&vert_pos,
		face_normal,
		face_edge,
		face_halfedges,
		face_rel,
		epsilon,
	);

	#[cfg(feature = "test_thoroughly")]
	debug_assert!(
		tri.halfedge.is_manifold(),
		"triangulated mesh is not manifold!"
	);

	let mut properties = Properties {
		data: construct::create_properties(
			&mut tri.halfedge,
			&mut tri.relation,
			&vert_pos,
			&instance_rel,
			instance_id_offset_q,
			in_p.vert_pos,
			in_p.tri.halfedge,
			in_p.tri.relation,
			in_p.properties,
			in_q.vert_pos,
			in_q.tri.halfedge,
			in_q.tri.relation,
			in_q.properties,
			invert_q,
			epsilon,
		),
		stride: prop_stride,
	};

	let first_new_vert = n_pv + n_qv;

	pp::split_pinched_verts(&mut tri.halfedge, &mut vert_pos);
	pp::dedupe_edges(&mut tri, &mut vert_pos);
	pp::collapse_short_edges(
		&mut tri.halfedge,
		&mut vert_pos,
		&tri.normal,
		&tri.relation,
		&instance_rel,
		prop_stride,
		epsilon,
		tolerance,
		first_new_vert,
	);
	pp::collapse_colinear_edges(
		&mut tri.halfedge,
		&mut vert_pos,
		&tri.normal,
		&tri.relation,
		&instance_rel,
		prop_stride,
		epsilon,
		first_new_vert,
	);
	pp::swap_degenerates(
		&mut tri,
		&mut vert_pos,
		&mut properties,
		&instance_rel,
		epsilon,
		tolerance,
		first_new_vert,
	);
	pp::mark_unreferenced_verts(&mut tri.halfedge, &mut vert_pos);

	#[cfg(feature = "test_thoroughly")]
	if get_intermediate_checks() {
		debug_assert!(
			tri.halfedge.is_2_manifold(),
			"simplified mesh is not 2-manifold!"
		);
	}

	let bbox = Box3D::from_cloud(&vert_pos);
	let Some(collider) =
		pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
	else {
		return Ok(MeshBool::decimated(
			None,
			instance_rel,
			prop_stride,
			precision,
		));
	};

	let out_r = MeshBool {
		original_id: None,
		precision,
		vert_pos: Rc::new(vert_pos),
		properties: Rc::new(properties),
		tri: tri.into_rc(),
		instance_relation: instance_rel,
		collider,
	};

	#[cfg(feature = "test_thoroughly")]
	if get_self_intersection_checks() {
		debug_assert!(!out_r.is_self_intersecting(), "self intersection detected");
	}

	Ok(out_r)
}
