use crate::MeshBool;
use crate::ops::boolean::BooleanError;
use crate::ops::boolean::face2tri::{assemble_halfedges, project_polygons};
use crate::postprocessing as pp;
use crate::postprocessing::sort::{gather_tris, reindex_verts};
use crate::spatial::aabb::Box3D;
use crate::spatial::bvh_collider::SimpleRecorder;
use crate::triangulation::{Polygons, SimplePolygon};
use crate::util::disjoint_sets::DisjointSets;
use crate::util::hash_table::DeterministicSet;
use crate::util::math::{get_axis_aligned_projection, next3_i32};
use crate::util::vec_ext;
use nalgebra::{Point3, Vector3};

impl MeshBool {
	// This operation returns a vector of Manifolds that are topologically
	// disconnected. If everything is connected, the vector is length one,
	// containing a copy of the original. It is the inverse operation of Compose().
	pub fn decompose(&self) -> Vec<Self> {
		let uf = DisjointSets::new(self.num_vert());
		for edge in 0..self.tri.halfedge.len() as i32 {
			if self.tri.halfedge.is_forward(edge) {
				uf.unite(
					self.tri.halfedge.start(edge) as usize,
					self.tri.halfedge.end(edge) as usize,
				);
			}
		}

		let (vert_label, num_components) = uf.connected_components();

		if num_components == 1 {
			return vec![self.clone()];
		}

		let num_vert = self.num_vert();
		let mut meshes: Vec<Self> = Vec::with_capacity(num_components);
		for i in 0..num_components as i32 {
			let mut vert_new2old: Vec<i32> = unsafe { vec_ext::uninit(num_vert) };
			let n_vert = vec_ext::copy_if(0..num_vert as i32, &mut vert_new2old, |v| {
				vert_label[v as usize] == i
			});
			let mut vert_pos = vec![Point3::default(); n_vert];
			vert_new2old.resize(n_vert, Default::default());
			vec_ext::gather(&vert_new2old, &self.vert_pos, &mut vert_pos);

			let mut face_new2old: Vec<i32> = Vec::with_capacity(self.num_tri());
			let halfedge = &self.tri.halfedge;
			for face in 0..self.num_tri() as i32 {
				if vert_label[halfedge.start(3 * face) as usize] == i {
					face_new2old.push(face);
				}
			}

			if face_new2old.is_empty() {
				continue;
			}

			let mut tri = gather_tris(&self.tri, &face_new2old);
			reindex_verts(&mut tri.halfedge, &vert_new2old, self.num_vert());
			let mut properties = self.properties.clone();
			let bbox = Box3D::from_cloud(&vert_pos);
			let collider =
				pp::sort_and_compact_geometry(&mut vert_pos, &mut properties, tri.partial(), bbox)
					.unwrap();

			meshes.push(Self {
				original_id: None,
				precision: self.precision, // inherit original object's precision
				vert_pos,
				properties,
				tri,
				instance_relation: self.instance_relation.clone(),
				collider,
			});
		}
		meshes
	}

	///Returns polygons representing the projected outline of this object
	///onto the X-Y plane. These polygons will often self-intersect, so it is
	///recommended to run them through the positive fill rule of CrossSection to get
	///a sensible result before using them.
	pub fn project(&self) -> Polygons {
		let projection = get_axis_aligned_projection(Vector3::new(0.0, 0.0, 1.0));
		let mut cusps = Vec::with_capacity(self.num_edge());
		for i in 0..self.tri.halfedge.len() {
			let pair = self.tri.halfedge.pair(i as i32);
			if self.tri.normal[(self.tri.halfedge.pair(pair) / 3) as usize].z >= 0.0
				&& self.tri.normal[(pair / 3) as usize].z < 0.0
			{
				cusps.push(self.tri.halfedge.get(i as i32));
			}
		}

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

	///Split cuts this manifold in two using the cutter manifold. The first result
	///is the intersection, second is the difference. This is more efficient than
	///doing them separately.
	///
	///@param cutter
	pub fn split(&self, cutter: &Self) -> Result<(Self, Self), BooleanError> {
		//this could be optimized like c++, which runs the intersections
		//half of the boolean pipeline once and reuses the results
		let result1 = self.intersect(cutter)?;
		let result2 = self.difference(cutter)?;
		Ok((result1, result2))
	}

	///Convenient version of Split() for a half-space.
	///
	///@param normal This vector is normal to the cutting plane and its length does
	///not matter. The first result is in the direction of this vector, the second
	///result is on the opposite side.
	///@param originOffset The distance of the plane from the origin in the
	///direction of the normal vector.
	pub fn split_by_plane(
		&self,
		normal: Vector3<f64>,
		origin_offset: f64,
	) -> Result<(Self, Self), BooleanError> {
		if self.is_empty() {
			let decimated1 = Self::decimated(
				None,
				self.instance_relation.clone(),
				self.properties.stride,
				self.precision,
			);
			let decimated2 = decimated1.clone();
			return Ok((decimated1, decimated2));
		}

		self.split(&halfspace(self.bounding_box(), normal, origin_offset))
	}

	///Identical to SplitByPlane(), but calculating and returning only the first
	///result.
	///
	///@param normal This vector is normal to the cutting plane and its length does
	///not matter. The result is in the direction of this vector from the plane.
	///@param originOffset The distance of the plane from the origin in the
	///direction of the normal vector.
	pub fn trim_by_plane(
		&self,
		normal: Vector3<f64>,
		origin_offset: f64,
	) -> Result<Self, BooleanError> {
		self.intersect(&halfspace(self.bounding_box(), normal, origin_offset))
	}

	///Returns the cross section of this object parallel to the X-Y plane at the
	///specified Z height, defaulting to zero. Using a height equal to the bottom of
	///the bounding box will return the bottom faces, while using a height equal to
	///the top of the bounding box will return empty.
	pub fn slice(&self, height: f64) -> Polygons {
		let mut plane = self.bounding_box();
		plane.min.z = height;
		plane.max.z = height;
		let mut query = vec![];
		query.push(plane);

		let mut tris = DeterministicSet::new();
		let mut record_collision = |_, tri: i32| {
			let mut min = f64::INFINITY;
			let mut max = f64::NEG_INFINITY;
			for j in 0..3 {
				let z: f64 = self.vert_pos[self.tri.halfedge.start(3 * tri + j) as usize].z;
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
				if self.vert_pos[self.tri.halfedge.start(3 * start_tri + j) as usize].z > height
					&& self.vert_pos[self.tri.halfedge.start(3 * start_tri + next3_i32(j)) as usize]
						.z <= height
				{
					k = next3_i32(j);
					break;
				}
			}

			let mut tri = start_tri;
			loop {
				tris.take(&tri).unwrap();
				let edge = 3 * tri + k;
				if self.vert_pos[self.tri.halfedge.end(edge) as usize].z <= height {
					k = next3_i32(k);
				}

				let up = 3 * tri + k;
				let below = self.vert_pos[self.tri.halfedge.start(up) as usize];
				let above = self.vert_pos[self.tri.halfedge.end(up) as usize];
				let a = (height - below.z) / (above.z - below.z);
				poly.push(below.lerp(&above, a).xy().into());

				let pair = self.tri.halfedge.pair(up);
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
}

fn halfspace(b_box: Box3D, mut normal: Vector3<f64>, origin_offset: f64) -> MeshBool {
	normal.normalize_mut();
	let mut cutter = MeshBool::cube(Vector3::repeat(2.0), true)
		.unwrap()
		.translate(Vector3::new(1.0, 0.0, 0.0));
	let size: f64 = (b_box.center() - normal * origin_offset).norm() + 0.5 * b_box.size().norm();
	cutter = cutter
		.scale(Vector3::repeat(size))
		.translate(Vector3::new(origin_offset, 0.0, 0.0));
	let y_deg: f64 = (-libm::asin(normal.z)).to_degrees();
	let z_deg: f64 = libm::atan2(normal.y, normal.x).to_degrees();
	return cutter.rotate(0.0, y_deg, z_deg);
}
