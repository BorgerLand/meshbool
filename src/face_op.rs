use crate::AABB;
use crate::collider::SimpleRecorder;
use crate::common::{DeterministicMap, DeterministicSet, Polygons, SimplePolygon};
use crate::meshboolimpl::MeshBoolImpl;
use crate::parallel::exclusive_scan_in_place;
use crate::polygon::{PolyVert, PolygonsIdx, SimplePolygonIdx, triangulate_idx_halfedges};
use crate::polygon_internal::HalfedgeTriangulation;
use crate::shared::{Halfedge, Halfedges, TriRef, get_axis_aligned_projection};
use crate::utils::{ccw, next3_i32, next3_usize};
use crate::vec::{vec_resize, vec_uninit};
use nalgebra::{Matrix2x3, Matrix3x2, Point3, Vector3};
use std::collections::VecDeque;
use std::collections::hash_map::Entry;
use std::mem;
use std::ops::DerefMut;

///Returns an assembled set of vertex index loops of the input list of
///Halfedges, where each vert must be referenced the same number of times as a
///startVert and endVert. If startHalfedgeIdx is given, instead of putting
///vertex indices into the returned polygons structure, it will use the halfedge
///indices instead.
fn assemble_halfedges(edges: &[Halfedge], start_halfedge_idx: i32) -> Vec<Vec<i32>> {
	let mut vert_edge: DeterministicMap<i32, VecDeque<i32>> = DeterministicMap::new(); //originally a c++ multimap
	for (i, edge) in edges.iter().enumerate() {
		vert_edge
			.entry(edge.start_vert)
			.or_default()
			.push_back(i as i32);
	}

	let mut polys = Vec::new();
	let mut start_edge = 0;
	let mut this_edge = start_edge;
	loop {
		if this_edge == start_edge {
			if vert_edge.is_empty() {
				break;
			}
			start_edge = vert_edge.values().next().unwrap()[0];
			this_edge = start_edge;
			polys.push(Vec::new());
		}

		polys
			.last_mut()
			.unwrap()
			.push(start_halfedge_idx + this_edge);
		let Entry::Occupied(mut result) = vert_edge.entry(edges[this_edge as usize].end_vert)
		else {
			panic!("non-manifold edge");
		};
		this_edge = result.get_mut().pop_front().expect("non-manifold edge");
		if result.get().is_empty() {
			result.remove();
		}
	}

	polys
}

///Add the vertex position projection to the indexed polygons.
fn project_polygons(
	polys: &[Vec<i32>],
	halfedge: &[Halfedge],
	vert_pos: &[Point3<f64>],
	projection: Matrix2x3<f64>,
) -> PolygonsIdx {
	let mut polygons = PolygonsIdx::new();
	for poly in polys {
		let mut polygon = SimplePolygonIdx::new();
		for &edge in poly {
			polygon.push(PolyVert {
				pos: (projection * vert_pos[halfedge[edge as usize].start_vert as usize]),
				idx: edge,
			});
		} //for vert

		polygons.push(polygon);
	} //for poly

	polygons
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
			let out = (first_out + num_edge) as i32;
			let start = triangles[tri * 3 + i];
			let end = triangles[tri * 3 + next3_usize(i)];
			local_edges[num_edge] = Vector3::new(start, end, out);
			output.set_start(out, face_halfedge[start as usize].start_vert);
			output.set_prop(out, face_halfedge[start as usize].prop_vert);
			output.set_pair(out, -1);
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
			output.set_pair(edge[2], pair);
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
		let out = (first_out + local) as i32;
		let edge = &triangulation.halfedges[triangulation.contour_end + local];
		output.set_start(out, face_halfedge[edge.start_vert as usize].start_vert);
		output.set_prop(out, face_halfedge[edge.start_vert as usize].prop_vert);
		if edge.paired_halfedge >= triangulation.contour_end as i32 {
			output.set_pair(
				out,
				(first_out + (edge.paired_halfedge as usize) - triangulation.contour_end) as i32,
			);
		} else {
			output.set_pair(out, -1);
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

fn write_tri_refs(
	tri_normal: &mut [Vector3<f64>],
	tri_refs: &mut [TriRef],
	first_tri: usize,
	num_tri: usize,
	normal: Vector3<f64>,
	tri_ref: TriRef,
) {
	for tri in 0..num_tri {
		tri_normal[first_tri + tri] = normal;
		tri_refs[first_tri + tri] = tri_ref;
	}
}

impl MeshBoolImpl {
	///Triangulates the faces. In this case, the halfedge_ vector is not yet a set
	///of triangles as required by this data structure, but is instead a set of
	///general faces with the input faceEdge vector having length of the number of
	///faces + 1. The values are indicies into the halfedge_ vector for the first
	///edge of each face, with the final value being the length of the halfedge_
	///vector itself. Upon return, halfedge_ has been lengthened and properly
	///represents the mesh as a set of triangles as usual. In this process the
	///faceNormal_ values are retained, repeated as necessary.
	pub fn face2tri(
		&mut self,
		face_edge: &[i32],
		face_halfedge: &[Halfedge],
		halfedge_ref: &[TriRef],
		allow_convex: bool,
	) {
		let general_triangulation = Some(|face| {
			let normal = self.face_normal[face];
			let projection = get_axis_aligned_projection(normal);
			let polys = project_polygons(
				&assemble_halfedges(
					&face_halfedge[(face_edge[face] as usize)..(face_edge[face + 1] as usize)],
					face_edge[face],
				),
				face_halfedge,
				&self.vert_pos,
				projection,
			);

			triangulate_idx_halfedges(&polys, self.epsilon, allow_convex)
		});

		let mut tri_offset: Vec<usize> = unsafe { vec_uninit(face_edge.len()) };
		*tri_offset.last_mut().unwrap() = 0;

		let mut results: DeterministicMap<i32, HalfedgeTriangulation> = DeterministicMap::new();
		for face in 0..face_edge.len() - 1 {
			let num_edge = face_edge[face + 1] - face_edge[face];
			if num_edge == 0 {
				tri_offset[face] = 0;
				continue;
			}
			debug_assert!(num_edge >= 3, "face has less than three edges.");
			tri_offset[face] = (num_edge - 2) as usize;
			if num_edge > 4 {
				let triangulation = (general_triangulation.unwrap())(face);
				tri_offset[face] = triangulation.num_tri();
				results.entry(face as i32).or_insert(triangulation);
			}
		}

		exclusive_scan_in_place(&mut tri_offset, 0);
		let tri_offset_back = *tri_offset.last().unwrap();
		unsafe {
			self.halfedge.resize_nofill(3 * tri_offset_back);
		}
		let mut tri_normal = vec![Vector3::default(); tri_offset_back];
		let tri_ref = &mut self.mesh_relation.tri_ref;
		*tri_ref = vec![TriRef::default(); tri_offset_back];
		let mut contour2tri = vec![-1; face_halfedge.len()];

		for face in 0..(face_edge.len() - 1) {
			let result_ptr = results.get(&(face as i32));
			self.output_face(
				&mut contour2tri,
				&mut tri_normal,
				face_edge,
				face_halfedge,
				halfedge_ref,
				face,
				tri_offset[face],
				result_ptr,
			);
		}

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
			self.halfedge.set_pair(tri_edge, pair_tri);
		}

		self.face_normal = tri_normal;
	}

	fn output_face(
		&mut self,
		contour2tri: &mut [i32],
		tri_normal: &mut [Vector3<f64>],
		face_edge: &[i32],
		face_halfedge: &[Halfedge],
		halfedge_ref: &[TriRef],
		face: usize,
		first_tri: usize,
		general: Option<&HalfedgeTriangulation>,
	) {
		let first_edge_i32 = face_edge[face];
		let first_edge = first_edge_i32 as usize;
		let last_edge = face_edge[face + 1];
		let num_edge = (last_edge - first_edge_i32) as usize;
		if num_edge == 0 {
			return;
		}
		debug_assert!(num_edge >= 3, "face has less than three edges.");
		let normal = self.face_normal[face];
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
				&mut self.halfedge,
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
					projection * self.vert_pos[face_halfedge[tri[0] as usize].start_vert as usize],
					projection * self.vert_pos[face_halfedge[tri[1] as usize].start_vert as usize],
					projection * self.vert_pos[face_halfedge[tri[2] as usize].start_vert as usize],
					self.epsilon,
				) >= 0
			};

			let quad = &assemble_halfedges(
				&face_halfedge[face_edge[face] as usize..face_edge[face + 1] as usize],
				face_edge[face],
			)[0];

			let tris = [
				Matrix3x2::<i32>::new(quad[0], quad[0], quad[1], quad[2], quad[2], quad[3]),
				Matrix3x2::<i32>::new(quad[1], quad[0], quad[2], quad[1], quad[3], quad[3]),
			];

			let mut choice = 0;
			if !(tri_ccw(tris[0].column(0).into()) && tri_ccw(tris[0].column(1).into())) {
				choice = 1;
			} else if tri_ccw(tris[1].column(0).into()) && tri_ccw(tris[1].column(1).into()) {
				let diag0 = self.vert_pos[face_halfedge[quad[0] as usize].start_vert as usize]
					- self.vert_pos[face_halfedge[quad[2] as usize].start_vert as usize];
				let diag1 = self.vert_pos[face_halfedge[quad[1] as usize].start_vert as usize]
					- self.vert_pos[face_halfedge[quad[3] as usize].start_vert as usize];

				if diag0.magnitude_squared() > diag1.magnitude_squared() {
					choice = 1;
				}
			}

			write_local_triangles(
				&mut self.halfedge,
				contour2tri,
				face_halfedge,
				first_tri,
				tris[choice].as_slice(),
			);
		} else {
			// General triangulation
			let general = general.expect("general face missing triangulation result");
			num_tri = general.num_tri();
			write_general_triangulation(
				&mut self.halfedge,
				contour2tri,
				face_halfedge,
				first_tri,
				general,
			);
		}

		write_tri_refs(
			tri_normal,
			&mut self.mesh_relation.tri_ref,
			first_tri,
			num_tri,
			normal,
			halfedge_ref[first_edge],
		);
	}

	pub fn slice(&self, height: f64) -> Polygons {
		let mut plane: AABB = self.bbox;
		plane.min.z = height;
		plane.max.z = height;
		let mut query: Vec<AABB> = vec![];
		query.push(plane);

		let mut tris = DeterministicSet::<i32>::new();
		let mut record_collision = |_, tri: i32| {
			let mut min: f64 = core::f64::INFINITY;
			let mut max: f64 = core::f64::NEG_INFINITY;
			for j in [0, 1, 2] {
				let z: f64 = self.vert_pos[self.halfedge.start(3 * tri + j) as usize].z;
				min = min.min(z);
				max = max.max(z);
			}

			if min <= height && max > height {
				tris.insert(tri);
			}
		};

		let mut recorder = SimpleRecorder::new(&mut record_collision);
		self.collider
			.collisions_from_slice::<false, _>(&mut recorder, &query, false);

		let mut polys = Polygons::default();
		while !tris.is_empty() {
			let start_tri = *tris.iter().next().unwrap();
			let mut poly = SimplePolygon::default();

			let mut k = 0;
			for j in 0..3 {
				if self.vert_pos[self.halfedge.start(3 * start_tri + j) as usize].z > height
					&& self.vert_pos[self.halfedge.start(3 * start_tri + next3_i32(j)) as usize].z
						<= height
				{
					k = next3_i32(j);
					break;
				}
			}

			let mut tri = start_tri;
			loop {
				tris.take(&tri).unwrap();
				let edge = 3 * tri + k;
				if self.vert_pos[self.halfedge.end(edge) as usize].z <= height {
					k = next3_i32(k);
				}

				let up = 3 * tri + k;
				let below = self.vert_pos[self.halfedge.start(up) as usize];
				let above = self.vert_pos[self.halfedge.end(up) as usize];
				let a = (height - below.z) / (above.z - below.z);
				poly.push(below.lerp(&above, a).xy().into());

				let pair = self.halfedge.pair(up);
				tri = pair / 3;
				k = next3_i32(pair % 3);

				if tri == start_tri {
					break;
				}
			}

			polys.push(poly);
		}

		polys
	}

	pub fn project(&self) -> Polygons {
		let projection = get_axis_aligned_projection(Vector3::new(0.0, 0.0, 1.0));
		let mut cusps: Vec<Halfedge> = unsafe { vec_uninit(self.num_edge()) };
		let mut num_cusps = 0;
		for i in 0..self.halfedge.len() {
			let pair = self.halfedge.pair(i as i32);
			if self.face_normal[(self.halfedge.pair(pair) / 3) as usize].z >= 0.0
				&& self.face_normal[(pair / 3) as usize].z < 0.0
			{
				cusps[num_cusps] = self.halfedge.get(i as i32);
				num_cusps += 1;
			}
		}
		vec_resize(&mut cusps, num_cusps, Halfedge::default());

		let polys_indexed = project_polygons(
			&assemble_halfedges(&cusps, 0),
			&cusps,
			&self.vert_pos,
			projection,
		);

		let mut polys: Polygons = vec![];
		for poly in polys_indexed.iter() {
			let mut simple: SimplePolygon = vec![];
			for poly_vert in poly.iter() {
				simple.push(poly_vert.pos);
			}
			polys.push(simple);
		}

		polys
	}
}
