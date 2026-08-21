use crate::halfedge::Halfedges;
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::vec_ext;
use crate::{Properties, Triangles, TrianglesPartial, TrianglesWIP};
use nalgebra::Point3;
use std::f64;
use std::mem::{self};
use std::rc::Rc;

#[cfg(feature = "test_thoroughly")]
use {crate::halfedge::Halfedge, crate::test::get_intermediate_checks};

const K_NO_CODE: u32 = 0xFFFFFFFF;

///Once halfedge_ has been filled in, this function can be called to create the
///rest of the internal data structures. This function also removes the verts
///and halfedges flagged for removal (NaN verts and -1 halfedges). Returns None
///if prior preprocessing stages compacted the mesh out of existence. If no edge
///postprocessing was done, it's safe to unwrap.
pub fn sort_and_compact_geometry(
	vert_pos: &mut Vec<Point3<f64>>,
	properties: &mut Properties,
	mut tri: TrianglesPartial,
	bbox: Box3D,
) -> Option<Rc<BVHCollider>> {
	sort_verts(vert_pos, &mut tri.halfedge, bbox, properties.stride > 0);

	if vert_pos.len() == 0 {
		//decimated
		return None;
	}

	let (mut tri_box, tri_morton) = get_tri_box_morton(&tri.halfedge, vert_pos, Some(bbox));
	let mut tri_morton = tri_morton.unwrap();
	sort_tris(tri.reborrow(), &mut tri_box, &mut tri_morton);

	if tri.halfedge.len() == 0 {
		//decimated
		return None;
	}

	compact_props(properties, &mut tri.halfedge);

	//presumably this happens here because sort+compact is usually
	//the final postprocessing stage, so the mesh is done now
	debug_assert!(
		tri.halfedge.len() % 6 == 0,
		"Not an even number of faces after sorting faces!"
	);

	#[cfg(feature = "test_thoroughly")]
	{
		if get_intermediate_checks() {
			let max_or_minus = |a: i32, b: i32| {
				if a.min(b) < 0 { -1 } else { a.max(b) }
			};
			let mut tri_idx = 0;
			let mut extrema = Halfedge::default();
			for i in 0..tri.halfedge.len() {
				let start = if tri.halfedge.is_forward(i) {
					tri.halfedge.start[i]
				} else {
					tri.halfedge.end(i)
				};
				let end = if tri.halfedge.is_forward(i) {
					tri.halfedge.end(i)
				} else {
					tri.halfedge.start[i]
				};
				extrema.start_vert = extrema.start_vert.min(start);
				extrema.end_vert = extrema.end_vert.min(end);
				extrema.paired_halfedge =
					max_or_minus(extrema.paired_halfedge, tri.halfedge.pair[i]);
				tri_idx = max_or_minus(tri_idx, (i / 3) as i32);
			}
			debug_assert!(extrema.start_vert >= 0, "Vertex index is negative!");
			debug_assert!(
				extrema.end_vert < vert_pos.len() as i32,
				"Vertex index exceeds number of verts!"
			);
			debug_assert!(extrema.paired_halfedge >= 0, "Halfedge index is negative!");
			debug_assert!(
				extrema.paired_halfedge < 2 * tri.halfedge.num_edge() as i32,
				"Halfedge index exceeds number self halfedges!"
			);
			debug_assert!(tri_idx >= 0, "Face index is negative!");
			debug_assert!(
				tri_idx < tri.halfedge.num_tri() as i32,
				"Face index exceeds number of faces!"
			);
		}

		if let Some(tri_rel) = tri.relation.as_deref() {
			debug_assert!(
				tri_rel.len() == tri.halfedge.num_tri(),
				"Mesh Relation doesn't fit!"
			);
		}

		debug_assert!(tri.halfedge.is_2_manifold(), "mesh is not 2-manifold!");
	}

	Some(Rc::new(BVHCollider::new(&tri_box, &tri_morton)))
}

///Sorts the vertices according to their Morton code.
fn sort_verts(
	vert_pos: &mut Vec<Point3<f64>>,
	halfedge: &mut Halfedges,
	bbox: Box3D,
	has_prop: bool,
) {
	let vert_morton = Vec::from_iter(vert_pos.iter().map(|&vert| morton_code(vert, bbox)));

	let num_vert = vert_pos.len();
	let mut vert_new2old = Vec::from_iter(0..num_vert as i32);
	vert_new2old.sort_unstable_by_key(|&i| vert_morton[i as usize]);

	reindex_verts(halfedge, &vert_new2old, num_vert, has_prop);

	// Verts were flagged for removal with NaNs and assigned kNoCode to sort
	// them to the end, which allows them to be removed.
	let new_num_vert = vert_new2old.partition_point(|&vert| vert_morton[vert as usize] < K_NO_CODE);

	vert_new2old.truncate(new_num_vert);
	*vert_pos = vec_ext::gather(vert_pos, vert_new2old.iter());
}

///Updates the halfedges to point to new vert indices based on a mapping,
///vertNew2Old. This may be a subset, so the total number of original verts is
///also given.
pub fn reindex_verts(
	halfedge: &mut Halfedges,
	vert_new2old: &[i32],
	old_num_vert: usize,
	has_prop: bool,
) {
	let vert_old2new = unsafe { vec_ext::scatter(vert_new2old.iter(), old_num_vert) };
	for idx in 0..halfedge.len() {
		let start_vert = halfedge.start[idx];
		if start_vert < 0 {
			continue;
		}
		let new_start = vert_old2new[start_vert as usize];
		halfedge.start[idx] = new_start;
		if !has_prop {
			halfedge.prop[idx] = new_start;
		}
	}
}

///Fills the faceBox and optionally faceMorton input with the bounding boxes and
///Morton codes of the faces, respectively. The Morton code is based on the center
///of the bounding box.
pub fn get_tri_box_morton(
	halfedge: &Halfedges,
	vert_pos: &[Point3<f64>],
	bbox: Option<Box3D>,
) -> (Vec<Box3D>, Option<Vec<u32>>) {
	let mut tri_morton = bbox.map(|_| Vec::with_capacity(halfedge.num_tri()));
	let tri_box = (0..halfedge.num_tri())
		.map(|tri| {
			let mut cur_box = Box3D::empty();

			// Removed tris are marked by all halfedges having pairedHalfedge
			// = -1, and this will sort them to the end (the Morton code only
			// uses the first 30 of 32 bits).
			if halfedge.pair[3 * tri] < 0 {
				if let Some(tri_morton) = &mut tri_morton {
					tri_morton.push(K_NO_CODE);
				}
				return cur_box;
			}

			let mut center = Point3::new(0.0, 0.0, 0.0);

			for i in 0..3 {
				let pos = vert_pos[halfedge.start[3 * tri + i] as usize];
				center += pos.coords;
				cur_box.union_point_mut(pos);
			}

			if let Some(tri_morton) = &mut tri_morton {
				tri_morton.push(morton_code(center / 3.0, bbox.unwrap()));
			}

			cur_box
		})
		.collect();

	(tri_box, tri_morton)
}

///Sorts the faces of this manifold according to their input Morton code. The
///bounding box and Morton code arrays are also sorted accordingly.
fn sort_tris(tri: TrianglesPartial, tri_box: &mut Vec<Box3D>, tri_morton: &mut Vec<u32>) {
	let mut tri_new2old = Vec::from_iter(0..tri.halfedge.num_tri() as i32);
	tri_new2old.sort_unstable_by_key(|&i| tri_morton[i as usize]);

	// Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
	// to sort them to the end, which allows them to be removed.
	let new_num_tri = tri_new2old.partition_point(|&face| tri_morton[face as usize] < K_NO_CODE);

	tri_new2old.truncate(new_num_tri);

	*tri_box = vec_ext::gather(tri_box, tri_new2old.iter());
	*tri_morton = vec_ext::gather(tri_morton, tri_new2old.iter());
	gather_tris_in_place(tri, &tri_new2old);
}

//couldn't think of a clean way of merging the duplicate logic without
//a large increase in ram. in place version drops old tris immediately
//after sorting while other version must keep old tris alive
fn gather_tris_in_place(mut tri: TrianglesPartial, tri_new2old: &[i32]) {
	let num_tri = tri_new2old.len();
	if let Some(tri_rel) = tri.relation.as_deref_mut() {
		*tri_rel = vec_ext::gather(tri_rel, tri_new2old.iter());
	}
	if let Some(tri_normal) = tri.normal.as_deref_mut() {
		*tri_normal = vec_ext::gather(tri_normal, tri_new2old.iter());
	}

	let old_halfedge = mem::replace(tri.halfedge, unsafe { Halfedges::uninit(3 * num_tri) });
	let tri_old2new = unsafe { vec_ext::scatter(tri_new2old.iter(), old_halfedge.num_tri()) };

	let mut reindex_face = ReindexTri {
		halfedge: &mut tri.halfedge,
		old_halfedge: &old_halfedge,
		tri_new2old,
		tri_old2new: &tri_old2new,
	};
	for new_face in 0..num_tri {
		reindex_face.call(new_face);
	}
}

///Creates the halfedge_ vector for this manifold by copying a set of faces from
///another manifold, given by oldHalfedge. Input faceNew2Old defines the old
///faces to gather into this.
pub fn gather_tris(old: &Triangles, tri_new2old: &[i32]) -> TrianglesWIP {
	let num_tri = tri_new2old.len();
	let new_tri_rel = vec_ext::gather(&old.relation, tri_new2old.iter());
	let new_tri_normal = vec_ext::gather(&old.normal, tri_new2old.iter());

	let mut new_halfedge = unsafe { Halfedges::uninit(3 * num_tri) };
	let tri_old2new = unsafe { vec_ext::scatter(tri_new2old.iter(), old.halfedge.num_tri()) };

	let mut reindex_face = ReindexTri {
		halfedge: &mut new_halfedge,
		old_halfedge: &old.halfedge,
		tri_new2old,
		tri_old2new: &tri_old2new,
	};
	for new_face in 0..num_tri {
		reindex_face.call(new_face);
	}

	TrianglesWIP {
		halfedge: new_halfedge,
		normal: new_tri_normal,
		relation: new_tri_rel,
	}
}

struct ReindexTri<'a> {
	halfedge: &'a mut Halfedges,
	old_halfedge: &'a Halfedges,
	tri_new2old: &'a [i32],
	tri_old2new: &'a [i32],
}

impl ReindexTri<'_> {
	//permute halfedge same as the other tri soa's and simultaneously update paired
	fn call(&mut self, new_face: usize) {
		let old_face = self.tri_new2old[new_face] as usize;
		for i in 0..3 {
			let old_edge = 3 * old_face + i;
			let mut edge = self.old_halfedge.get(old_edge);
			let paired_face = edge.paired_halfedge / 3;
			let offset = edge.paired_halfedge - 3 * paired_face;
			edge.paired_halfedge = 3 * self.tri_old2new[paired_face as usize] + offset;
			let new_edge = 3 * new_face + i;
			self.halfedge.set(
				new_edge,
				edge.start_vert,
				edge.paired_halfedge,
				edge.prop_vert,
			);
		}
	}
}

///Removes unreferenced property verts and reindexes propVerts.
fn compact_props(properties: &mut Properties, halfedge: &mut Halfedges) {
	if properties.stride == 0 {
		return;
	}
	let num_prop_verts = properties.data.len() / properties.stride;
	let mut keep = vec![0; num_prop_verts];

	for idx in 0..halfedge.len() {
		keep[halfedge.prop[idx] as usize] = 1;
	}

	let prop_old2new = vec_ext::exclusive_scan_with_total(keep.iter().cloned(), 0);

	let num_verts_new = prop_old2new[num_prop_verts] as usize;
	let old_prop = mem::replace(&mut properties.data, unsafe {
		vec_ext::uninit(properties.stride * num_verts_new)
	});
	for old_idx in 0..num_prop_verts {
		if keep[old_idx] == 0 {
			continue;
		}
		for p in 0..properties.stride {
			properties.data[prop_old2new[old_idx] as usize * properties.stride + p] =
				old_prop[old_idx * properties.stride + p];
		}
	}

	for idx in 0..halfedge.len() {
		halfedge.prop[idx] = prop_old2new[halfedge.prop[idx] as usize];
	}
}

pub fn morton_code(position: Point3<f64>, bbox: Box3D) -> u32 {
	// Unreferenced vertices are marked NaN, and this will sort them to the end
	// (the Morton code only uses the first 30 of 32 bits).
	if position.x.is_nan() {
		K_NO_CODE
	} else {
		BVHCollider::morton_code(position, bbox)
	}
}
