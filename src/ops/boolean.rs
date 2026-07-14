use crate::ops::boolean::intersect::Intersections;
use crate::postprocessing as pp;
use crate::spatial::aabb::{Box3D, Overlap};
use crate::util::hash_table::DeterministicMap;
use crate::util::vec_ext;
use crate::{MeshBool, Precision, Properties};
use std::collections::BTreeMap;
use std::ptr;

#[cfg(feature = "test_thoroughly")]
use crate::test::{get_intermediate_checks, get_self_intersection_checks};

mod construct;
pub mod face2tri;
mod intersect;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BooleanError {
	ResultTooLarge,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum OpType {
	Union,
	Difference,
	Intersect,
}

impl MeshBool {
	pub fn union(&self, other: &Self) -> Result<Self, BooleanError> {
		boolean(self, OpType::Union, other)
	}

	pub fn difference(&self, other: &Self) -> Result<Self, BooleanError> {
		boolean(self, OpType::Difference, other)
	}

	pub fn intersect(&self, other: &Self) -> Result<Self, BooleanError> {
		boolean(self, OpType::Intersect, other)
	}
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
fn boolean(in_p: &MeshBool, op: OpType, in_q: &MeshBool) -> Result<MeshBool, BooleanError> {
	let prop_stride = in_p.properties.stride.max(in_q.properties.stride);
	let precision = Precision {
		epsilon: in_p.precision.epsilon.max(in_q.precision.epsilon),
		tolerance: in_p.precision.tolerance.max(in_q.precision.tolerance),
	};

	let invert_q = op == OpType::Difference;

	let instance_id_merge = in_p
		.instance_relation
		.iter()
		.map(|(&old_id, &rel)| (true, old_id, rel))
		.chain(in_q.instance_relation.iter().map(|(&old_id, rel)| {
			let mut rel = *rel;
			rel.back_side ^= invert_q;
			(false, old_id, rel)
		}))
		.enumerate()
		.map(|(new_id, (pq, old_id, rel))| (pq, old_id, new_id as u32, rel));

	let instance_rel = instance_id_merge
		.clone()
		.map(|(_, _, new_id, rel)| (new_id, rel));

	let decimated = || {
		Ok(MeshBool::decimated(
			None,
			instance_rel.clone().collect(),
			prop_stride,
			precision,
		))
	};

	let cloned = |mesh: &MeshBool| {
		Ok(MeshBool {
			instance_relation: instance_rel.clone().collect(),
			..mesh.clone()
		})
	};

	if ptr::eq(in_p, in_q) {
		if op == OpType::Difference {
			return decimated();
		}

		return cloned(in_p);
	} else if in_p.is_empty() {
		if !in_q.is_empty() && op == OpType::Union {
			return cloned(in_q);
		}

		return decimated();
	} else if in_q.is_empty() {
		if op == OpType::Intersect {
			return decimated();
		}

		return cloned(in_p);
	} else if !in_p
		.collider
		.get_bounding_box()
		.does_overlap(in_q.collider.get_bounding_box())
	{
		if op == OpType::Difference {
			return cloned(in_p);
		} else if op == OpType::Intersect {
			return decimated();
		}

		//else union can be optimized via the same technique as CsgLeafNode::Compose(),
		//copying and pasting the 2 disjoint meshes into the same buffer.
		//for now just let the full pipeline run
	}

	// Symbolic perturbation:
	// Union -> expand inP, expand inQ
	// Difference, Intersection -> contract inP, expand inQ
	// Technically Intersection should contract inQ, but doing it this way makes
	// Split faster and any suboptimal cases seem pretty rare.
	let expand_p = op == OpType::Union;
	const INT_MAX_SZ: usize = i32::MAX as usize;

	let mut w03 = vec![0; in_p.num_vert()];
	let mut w30 = vec![0; in_q.num_vert()];
	let mut xv12 = Intersections::default();
	let mut xv21 = Intersections::default();

	if in_p
		.collider
		.get_bounding_box()
		.does_overlap(in_q.collider.get_bounding_box())
	{
		let vert_normal_p = &in_p.calculate_vert_normals_internal();
		let vert_normal_q = &in_q.calculate_vert_normals_internal();

		// Level 3
		// Build up the intersection of the edges and triangles, keeping only those
		// that intersect, and record the direction the edge is passing through the
		// triangle.
		intersect::intersect12::<true>(
			&mut xv12,
			in_p,
			vert_normal_p,
			in_q,
			vert_normal_q,
			expand_p,
		);

		if xv12.x12.len() > INT_MAX_SZ {
			return Err(BooleanError::ResultTooLarge);
		}

		intersect::intersect12::<false>(
			&mut xv21,
			in_p,
			vert_normal_p,
			in_q,
			vert_normal_q,
			expand_p,
		);

		if xv21.x12.len() > INT_MAX_SZ {
			return Err(BooleanError::ResultTooLarge);
		}

		// Compute winding numbers of all vertices using flood fill
		// Vertices on the same connected component have the same winding number
		intersect::winding03::<true>(
			&mut w03,
			in_p,
			vert_normal_p,
			in_q,
			vert_normal_q,
			&xv12.p1q2,
			expand_p,
		);
		intersect::winding03::<false>(
			&mut w30,
			in_p,
			vert_normal_p,
			in_q,
			vert_normal_q,
			&xv21.p1q2,
			expand_p,
		);
	}
	//else hitting the one CsgLeafNode::Compose() case not handled by early exits

	debug_assert!(
		expand_p == (op == OpType::Union),
		"Result op type not compatible with constructor op type."
	);
	let c1 = if op == OpType::Intersect { 0 } else { 1 };
	let c2 = if op == OpType::Union { 1 } else { 0 };
	let c3 = if op == OpType::Intersect { 1 } else { -1 };

	// Convert winding numbers to inclusion values based on operation type.
	//reuses existing allocation
	let i12: Vec<_> = xv12.x12.into_iter().map(|v| c3 * v).collect();
	let i21: Vec<_> = xv21.x12.into_iter().map(|v| c3 * v).collect();
	let i03: Vec<_> = w03.into_iter().map(|v| c1 + c3 * v).collect();
	let i30: Vec<_> = w30.into_iter().map(|v| c2 + c3 * v).collect();

	let abs_sum = |a: i32, b: i32| a.abs() + b.abs();

	let v_p2r = vec_ext::exclusive_scan_transformed(&i03, 0, &abs_sum);
	let mut num_vert_r = v_p2r.last().unwrap().abs() + i03.last().unwrap().abs();
	let n_pv = num_vert_r;

	let v_q2r = vec_ext::exclusive_scan_transformed(&i30, num_vert_r, &abs_sum);
	num_vert_r = abs_sum(*v_q2r.last().unwrap(), *i30.last().unwrap());
	let n_qv = num_vert_r - n_pv;

	let v12_r = if xv12.v12.len() == 0 {
		Vec::new()
	} else {
		let v12_r = vec_ext::exclusive_scan_transformed(&i12, num_vert_r, &abs_sum);
		num_vert_r = abs_sum(*v12_r.last().unwrap(), *i12.last().unwrap());
		v12_r
	};

	//let n12 = num_vert_r - n_pv - n_qv; //new verts from edgesP -> facesQ

	let v21_r = if xv21.v12.len() == 0 {
		Vec::new()
	} else {
		let v21_r = vec_ext::exclusive_scan_transformed(&i21, num_vert_r, &abs_sum);
		num_vert_r = abs_sum(*v21_r.last().unwrap(), *i21.last().unwrap());
		v21_r
	};

	//let n21 = num_vert_r - n_pv - n_qv - n12; //new verts from facesP -> edgesQ

	if num_vert_r == 0 {
		return decimated();
	}

	let mut vert_pos = unsafe { vec_ext::uninit(num_vert_r as usize) };
	// Add vertices, duplicating for inclusion numbers not in [-1, 1].
	// Retained vertices from P and Q:
	let mut retain_p = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i03,
		vert_r: &v_p2r,
		vert_pos_p: &in_p.vert_pos,
	};
	for vert in 0..in_p.num_vert() {
		retain_p.call(vert);
	}
	let mut retain_q = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i30,
		vert_r: &v_q2r,
		vert_pos_p: &in_q.vert_pos,
	};
	for vert in 0..in_q.num_vert() {
		retain_q.call(vert);
	}
	// New vertices created from intersections:
	let mut new12 = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i12,
		vert_r: &v12_r,
		vert_pos_p: &xv12.v12,
	};
	for vert in 0..i12.len() {
		new12.call(vert);
	}
	drop(xv12.v12);
	let mut new21 = construct::DuplicateVerts {
		vert_pos_r: &mut vert_pos,
		inclusion: &i21,
		vert_r: &v21_r,
		vert_pos_p: &xv21.v12,
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
	let mut edges_p: BTreeMap<i32, Vec<construct::EdgePos>> = BTreeMap::new();
	let mut edges_q: BTreeMap<i32, Vec<construct::EdgePos>> = BTreeMap::new();
	// This key is the face index of <P, Q>
	let mut edges_r: BTreeMap<(i32, i32), Vec<construct::EdgePos>> = BTreeMap::new();

	construct::add_new_edge_verts(
		&mut edges_p,
		&mut edges_r,
		&xv12.p1q2,
		&i12,
		v12_r,
		&in_p.tri.halfedge,
		true,
		0,
	);
	construct::add_new_edge_verts(
		&mut edges_q,
		&mut edges_r,
		&xv21.p1q2,
		&i21,
		v21_r,
		&in_q.tri.halfedge,
		false,
		xv12.p1q2.len(),
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
	let mut face_pq2r = construct::size_face_pq2r(&sides_per_face_pq);
	let num_face_r = face_pq2r.pop().unwrap() as usize;
	let face_normal = construct::size_face_normal(
		&in_p.tri.normal,
		&in_q.tri.normal,
		&sides_per_face_pq,
		num_face_r,
		invert_q,
	);
	let face_edge = construct::size_face_edge(sides_per_face_pq);

	// This gets incremented for each halfedge that's added to a face so that the
	// next one knows where to slot in.
	let mut face_ptr_r = face_edge.clone();
	// Intersected halfedges are marked false.
	let mut whole_halfedge_p = vec![true; in_p.tri.halfedge.len()];
	let mut whole_halfedge_q = vec![true; in_q.tri.halfedge.len()];

	let num_halfedge_r = *face_edge.last().unwrap() as usize;
	if num_halfedge_r == 0 {
		return decimated();
	}

	// The face_rel contains the data that will become triRel once the faces
	// are triangulated.
	let mut face_rel = unsafe { vec_ext::uninit(num_halfedge_r) };
	// Note that we are working with Vec<Halfedge> instead of Halfedges here,
	// since the faces can be arbitrary polygons before feeding into the
	// triangulator. prop_vert is meaningless until after create_properties
	let mut face_halfedges = unsafe { vec_ext::uninit(num_halfedge_r) };
	let instance_id_old2new: DeterministicMap<(bool, u32), u32> = instance_id_merge
		.map(|(pq, old_id, new_id, _)| ((pq, old_id), new_id))
		.collect();

	construct::append_partial_edges(
		&mut vert_pos,
		&mut face_halfedges,
		&mut face_ptr_r,
		&mut face_rel,
		in_p,
		edges_p,
		&mut whole_halfedge_p,
		&i03,
		&v_p2r,
		&face_pq2r,
		&in_p.tri.relation,
		&instance_id_old2new,
		prop_stride > 0,
		true,
	);
	construct::append_partial_edges(
		&mut vert_pos,
		&mut face_halfedges,
		&mut face_ptr_r,
		&mut face_rel,
		in_q,
		edges_q,
		&mut whole_halfedge_q,
		&i30,
		&v_q2r,
		&face_pq2r[in_p.num_tri()..],
		&in_q.tri.relation,
		&instance_id_old2new,
		prop_stride > 0,
		false,
	);

	construct::append_new_edges(
		&mut vert_pos,
		&mut face_halfedges,
		&mut face_ptr_r,
		&mut face_rel,
		edges_r,
		&face_pq2r,
		in_p.num_tri(),
		&in_p.tri.relation,
		&in_q.tri.relation,
		&instance_id_old2new,
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
		&face_pq2r[..in_p.num_tri()],
		&in_p.tri.relation,
		&instance_id_old2new,
		prop_stride > 0,
		true,
	);
	construct::append_whole_edges(
		&mut face_ptr_r,
		&mut face_halfedges,
		&mut face_rel,
		&in_q.tri.halfedge,
		whole_halfedge_q,
		i30,
		v_q2r,
		&face_pq2r[in_p.num_tri()..],
		&in_q.tri.relation,
		&instance_id_old2new,
		prop_stride > 0,
		false,
	);

	drop(face_ptr_r);
	drop(face_pq2r);
	drop(instance_id_old2new);

	// Level 6
	let mut tri = face2tri::face2tri(
		&vert_pos,
		face_normal,
		face_edge,
		face_halfedges,
		face_rel,
		precision.epsilon,
	);

	//MANIFOLD_PAR: halfedge.reorder();
	//(aka ReorderHalfedges/reorder_halfedges, removed because single
	//threaded is already deterministic)
	//tri.halfedge.reorder();

	#[cfg(feature = "test_thoroughly")]
	if get_intermediate_checks() {
		debug_assert!(
			tri.halfedge.is_manifold(),
			"triangulated mesh is not manifold!"
		);
	}

	let mut properties = Properties {
		data: construct::create_properties(
			&mut tri.halfedge,
			&mut tri.relation,
			&vert_pos,
			in_p.instance_relation.len() as u32,
			in_p,
			in_q,
			invert_q,
			precision.epsilon,
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
		prop_stride,
		precision,
		first_new_vert,
	);
	pp::collapse_colinear_edges(
		&mut tri.halfedge,
		&mut vert_pos,
		&tri.normal,
		&tri.relation,
		prop_stride,
		precision.epsilon,
		first_new_vert,
	);
	pp::swap_degenerates(
		&mut tri,
		&mut vert_pos,
		&mut properties,
		precision,
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
		return decimated();
	};

	let out_r = MeshBool {
		original_id: None,
		precision,
		vert_pos,
		properties,
		tri,
		instance_relation: instance_rel.collect(),
		collider,
	};

	//pulled from csg_tree.cpp
	#[cfg(feature = "test_thoroughly")]
	if get_self_intersection_checks() && out_r.is_self_intersecting() {
		panic!("self intersection detected");
	}

	Ok(out_r)
}
