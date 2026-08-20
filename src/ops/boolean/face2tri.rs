use crate::TrianglesWIP;
use crate::halfedge::{Halfedge, Halfedges};
use crate::mesh_relations::TriRelation;
use crate::triangulation::{EarClipBuffers, HalfedgeTriangulation, triangulate_idx_halfedges};
use crate::util::face::{assemble_halfedges, project_polygons};
use crate::util::math::{ccw, get_axis_aligned_projection, next3_usize};
use crate::util::vec_ext;
use nalgebra::{Matrix3x2, Point3, Vector3};
use rustc_hash::FxHashMap;
use std::mem;
use std::ops::DerefMut;

///Triangulates the faces. In this case, the halfedge_ vector is not yet a set
///of triangles as required by this data structure, but is instead a set of
///general faces with the input faceEdge vector having length of the number of
///faces + 1. The values are indicies into the halfedge_ vector for the first
///edge of each face, with the final value being the length of the halfedge_
///vector itself. Upon return, halfedge_ has been lengthened and properly
///represents the mesh as a set of triangles as usual. In this process the
///faceNormal_ values are retained, repeated as necessary.
pub fn face2tri(
	vert_pos: &[Point3<f64>],
	face_normal: Vec<Vector3<f64>>,
	face_edge: Vec<i32>,
	face_halfedge: Vec<Halfedge>,
	halfedge_rel: Vec<TriRelation>,
	epsilon: f64,
) -> TrianglesWIP {
	let mut scratch_buffer = EarClipBuffers::default();
	let mut general_triangulation = |face| {
		let normal = face_normal[face];
		let projection = get_axis_aligned_projection(normal);
		let polys = project_polygons(
			&assemble_halfedges(
				&face_halfedge[(face_edge[face] as usize)..(face_edge[face + 1] as usize)],
				face_edge[face],
			),
			&face_halfedge,
			vert_pos,
			projection,
		);

		triangulate_idx_halfedges(&polys, epsilon, false, Some(&mut scratch_buffer))
	};

	let mut tri_offset = unsafe { vec_ext::uninit(face_edge.len()) };
	*tri_offset.last_mut().unwrap() = 0;

	let mut results: FxHashMap<i32, HalfedgeTriangulation> = FxHashMap::default();
	for face in 0..face_edge.len() - 1 {
		let num_edge = face_edge[face + 1] - face_edge[face];
		if num_edge == 0 {
			tri_offset[face] = 0;
			continue;
		}
		debug_assert!(num_edge >= 3, "face has less than three edges.");
		tri_offset[face] = (num_edge - 2) as usize;
		if num_edge > 4 {
			let triangulation = general_triangulation(face);
			tri_offset[face] = triangulation.num_tri();
			results.entry(face as i32).or_insert(triangulation);
		}
	}

	drop(scratch_buffer);

	vec_ext::exclusive_scan_in_place(&mut tri_offset, 0);
	let tri_offset_back = *tri_offset.last().unwrap();
	let mut tris = TrianglesWIP {
		halfedge: unsafe { Halfedges::uninit(3 * tri_offset_back) },
		normal: unsafe { vec_ext::uninit(tri_offset_back) },
		relation: unsafe { vec_ext::uninit(tri_offset_back) },
	};
	let mut contour2tri = vec![-1; face_halfedge.len()];

	for face in 0..(face_edge.len() - 1) {
		output_face(
			&mut tris,
			&mut contour2tri,
			vert_pos,
			&face_normal,
			&face_edge,
			&face_halfedge,
			&halfedge_rel,
			face,
			tri_offset[face],
			results.get(&(face as i32)),
			epsilon,
		);
	}
	drop(face_normal);
	drop(face_edge);
	drop(halfedge_rel);

	for edge in 0..face_halfedge.len() {
		let tri_edge = contour2tri[edge];
		if tri_edge < 0 {
			continue;
		}
		let pair = face_halfedge[edge].paired_halfedge;
		if pair < 0 {
			continue;
		}
		let pair_tri = contour2tri[pair as usize];
		debug_assert!(
			pair_tri >= 0,
			"boundary edge did not triangulate with its pair"
		);
		tris.halfedge.pair[tri_edge as usize] = pair_tri;
	}

	tris
}

fn output_face(
	tris: &mut TrianglesWIP,
	contour2tri: &mut [i32],
	vert_pos: &[Point3<f64>],
	face_normal: &[Vector3<f64>],
	face_edge: &[i32],
	face_halfedge: &[Halfedge],
	halfedge_rel: &[TriRelation],
	face: usize,
	first_tri: usize,
	general: Option<&HalfedgeTriangulation>,
	epsilon: f64,
) {
	let first_edge_i32 = face_edge[face];
	let first_edge = first_edge_i32 as usize;
	let last_edge = face_edge[face + 1];
	let num_edge = (last_edge - first_edge_i32) as usize;
	if num_edge == 0 {
		return;
	}
	debug_assert!(num_edge >= 3, "face has less than three edges.");
	let normal = face_normal[face];
	let mut num_tri = num_edge - 2;

	if num_edge == 3
	//single triangle
	{
		let mut tri_edge = Vector3::new(first_edge_i32, first_edge_i32 + 1, first_edge_i32 + 2);
		let mut tri = Vector3::new(
			face_halfedge[first_edge].start_vert,
			face_halfedge[first_edge + 1].start_vert,
			face_halfedge[first_edge + 2].start_vert,
		);
		let mut ends = Vector3::new(
			face_halfedge[first_edge].end_vert,
			face_halfedge[first_edge + 1].end_vert,
			face_halfedge[first_edge + 2].end_vert,
		);

		if ends[0] == tri[2] {
			let switcheroo = tri_edge.deref_mut();
			mem::swap(&mut switcheroo.y, &mut switcheroo.z);
			let switcheroo = tri.deref_mut();
			mem::swap(&mut switcheroo.y, &mut switcheroo.z);
			let switcheroo = ends.deref_mut();
			mem::swap(&mut switcheroo.y, &mut switcheroo.z);
		}

		debug_assert!(
			ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
			"These 3 edges do not form a triangle!"
		);

		write_local_triangles(
			&mut tris.halfedge,
			contour2tri,
			face_halfedge,
			first_tri,
			tri_edge.as_slice(),
		);
	} else if num_edge == 4
	//pair of triangles
	{
		let projection = get_axis_aligned_projection(normal);
		let tri_ccw = |tri: Vector3<i32>| {
			ccw(
				projection * vert_pos[face_halfedge[tri[0] as usize].start_vert as usize],
				projection * vert_pos[face_halfedge[tri[1] as usize].start_vert as usize],
				projection * vert_pos[face_halfedge[tri[2] as usize].start_vert as usize],
				epsilon,
			) >= 0
		};

		let quad = &assemble_halfedges(
			&face_halfedge[face_edge[face] as usize..face_edge[face + 1] as usize],
			face_edge[face],
		)[0];

		let quads = [
			Matrix3x2::<i32>::new(quad[0], quad[0], quad[1], quad[2], quad[2], quad[3]),
			Matrix3x2::<i32>::new(quad[1], quad[0], quad[2], quad[1], quad[3], quad[3]),
		];

		let mut choice = 0;
		if !(tri_ccw(quads[0].column(0).into()) && tri_ccw(quads[0].column(1).into())) {
			choice = 1;
		} else if tri_ccw(quads[1].column(0).into()) && tri_ccw(quads[1].column(1).into()) {
			let diag0 = vert_pos[face_halfedge[quad[0] as usize].start_vert as usize]
				- vert_pos[face_halfedge[quad[2] as usize].start_vert as usize];
			let diag1 = vert_pos[face_halfedge[quad[1] as usize].start_vert as usize]
				- vert_pos[face_halfedge[quad[3] as usize].start_vert as usize];

			if diag0.magnitude_squared() > diag1.magnitude_squared() {
				choice = 1;
			}
		}

		write_local_triangles(
			&mut tris.halfedge,
			contour2tri,
			face_halfedge,
			first_tri,
			quads[choice].as_slice(),
		);
	} else {
		// General triangulation
		let general = general.expect("general face missing triangulation result");
		num_tri = general.num_tri();
		write_general_triangulation(
			&mut tris.halfedge,
			contour2tri,
			face_halfedge,
			first_tri,
			general,
		);
	}

	write_tri_rels(
		&mut tris.normal,
		&mut tris.relation,
		first_tri,
		num_tri,
		normal,
		halfedge_rel[first_edge],
	);
}

fn write_local_triangles(
	output: &mut Halfedges,
	contour2tri: &mut [i32],
	face_halfedge: &[Halfedge],
	first_tri: usize,
	triangles: &[i32],
) {
	debug_assert!(
		triangles.len() == 3 || triangles.len() == 6,
		"local face path only handles tris/quads"
	);
	let num_tri = triangles.len() / 3;
	let mut local_edges = [Vector3::default(); 6];
	let first_out = 3 * first_tri;
	let mut num_edge = 0;
	for tri in 0..num_tri {
		for i in 0..3 {
			let out = first_out + num_edge;
			let start = triangles[tri * 3 + i];
			let end = triangles[tri * 3 + next3_usize(i)];
			local_edges[num_edge] = Vector3::new(start, end, out as i32);
			output.start[out] = face_halfedge[start as usize].start_vert;
			output.pair[out] = -1;
			num_edge += 1;
		}
	}

	for i in 0..num_edge {
		let edge = local_edges[i];
		let mut pair = -1;
		for j in 0..num_edge {
			if local_edges[j][0] == edge[1] && local_edges[j][1] == edge[0] {
				pair = local_edges[j][2];
				break;
			}
		}
		if pair >= 0 {
			output.pair[edge[2] as usize] = pair;
		} else {
			contour2tri[edge[0] as usize] = edge[2];
		}
	}
}

fn write_general_triangulation(
	output: &mut Halfedges,
	contour2tri: &mut [i32],
	face_halfedge: &[Halfedge],
	first_tri: usize,
	triangulation: &HalfedgeTriangulation,
) {
	let first_out = 3 * first_tri;
	let num_tri_halfedge = 3 * triangulation.num_tri();
	for local in 0..num_tri_halfedge {
		let out = first_out + local;
		let edge = &triangulation.halfedges[triangulation.contour_end + local];
		output.start[out] = face_halfedge[edge.start_vert as usize].start_vert;
		if edge.paired_halfedge >= triangulation.contour_end as i32 {
			output.pair[out] =
				(first_out + (edge.paired_halfedge as usize) - triangulation.contour_end) as i32;
		} else {
			output.pair[out] = -1;
		}
	}

	for contour in 0..triangulation.contour_end {
		let edge = &triangulation.halfedges[contour];
		if edge.paired_halfedge < 0 {
			continue;
		}
		debug_assert!(
			edge.paired_halfedge as usize >= triangulation.contour_end,
			"contour paired to another contour"
		);
		let boundary = edge.end_vert;
		debug_assert!(
			boundary >= 0 && (boundary as usize) < contour2tri.len(),
			"contour edge index out of bounds"
		);
		contour2tri[boundary as usize] =
			(first_out + (edge.paired_halfedge as usize) - triangulation.contour_end) as i32;
	}
}

fn write_tri_rels(
	tri_normal: &mut [Vector3<f64>],
	tri_rels: &mut [TriRelation],
	first_tri: usize,
	num_tri: usize,
	normal: Vector3<f64>,
	tri_rel: TriRelation,
) {
	for tri in 0..num_tri {
		tri_normal[first_tri + tri] = normal;
		tri_rels[first_tri + tri] = tri_rel;
	}
}
