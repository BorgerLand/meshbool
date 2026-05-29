use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{TriRef, next_halfedge, safe_normalize};
use nalgebra::{Vector3, Vector4};
use std::collections::BTreeMap;

// Minimum sharp angle in degrees, below which edges are considered coplanar.
// Floating point noise in the dihedral angle computation can reach ~1e-6
// degrees for nearly-parallel face normals; this threshold must exceed that.
const K_MIN_SHARP_ANGLE: f64 = 1e-4;

///Get the angle between two unit-vectors.
fn angle_between(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
	let dot: f64 = a.dot(&b);
	if dot >= 1.0 {
		0.0
	} else {
		if dot <= -1.0 {
			core::f64::consts::PI
		} else {
			libm::acos(dot)
		}
	}
}

///Calculate a tangent vector in the form of a weighted cubic Bezier taking as
///input the desired tangent direction (length doesn't matter) and the edge
///vector to the neighboring vertex. In a symmetric situation where the tangents
///at each end are mirror images of each other, this will result in a circular
///arc.
fn circular_tangent(tangent: Vector3<f64>, edge_vec: Vector3<f64>) -> Vector4<f64> {
	let dir: Vector3<f64> = safe_normalize(tangent);

	let weight: f64 = 0.5_f64.max(dir.dot(&safe_normalize(edge_vec)));
	// Quadratic weighted bezier for circular interpolation
	let bz2: Vector4<f64> = (dir * 0.5 * edge_vec.norm()).push(weight);
	// Equivalent cubic weighted bezier
	let bz3: Vector4<f64> = Vector4::new(0.0, 0.0, 0.0, 1.0).lerp(&bz2, 2.0 / 3.0);
	// Convert from homogeneous form to geometric form
	return (bz3.xyz() / bz3.w).push(bz3.w);
}

impl MeshBoolImpl {
	///Returns a circular tangent for the requested halfedge, orthogonal to the
	///given normal vector, and avoiding folding when the tangent needs to be more
	///than 90 degrees from the edge vector.
	pub fn tangent_from_normal(&self, normal: &Vector3<f64>, halfedge: i32) -> Vector4<f64> {
		let edge_vec: Vector3<f64> = self.vert_pos[self.halfedge.end(halfedge) as usize]
			- self.vert_pos[self.halfedge.start(halfedge) as usize];
		let edge_normal: Vector3<f64> = self.face_normal[halfedge as usize / 3]
			+ self.face_normal[(self.halfedge.pair(halfedge) / 3) as usize];
		let bi_tangent = if normal.dot(&edge_normal) < 0.0 {
			edge_normal.cross(&edge_vec)
		} else {
			normal.cross(&edge_vec)
		};
		return circular_tangent(bi_tangent.cross(&normal), edge_vec);
	}

	///Returns true if this halfedge should be marked as the interior of a quad, as
	///defined by its two triangles referring to the same face, and those triangles
	///having no further face neighbors beyond.
	pub fn is_inside_quad(&self, halfedge: i32) -> bool {
		// if self.halfedge_tangent.len() > 0 {
		//   return self.halfedge_tangent[halfedge as usize].w < 0;
		// }
		let tri: i32 = halfedge / 3;
		let tref: TriRef = self.mesh_relation.tri_ref[tri as usize];
		let pair: i32 = self.halfedge.pair(halfedge);
		let pair_tri: i32 = pair / 3;
		let pair_ref: TriRef = self.mesh_relation.tri_ref[pair_tri as usize];
		if !tref.same_face(&pair_ref) {
			return false;
		}

		let same_face = |halfedge: i32, tref: &TriRef| {
			tref.same_face(&self.mesh_relation.tri_ref[(self.halfedge.pair(halfedge) / 3) as usize])
		};

		let mut neighbor: i32 = next_halfedge(halfedge);
		if same_face(neighbor, &tref) {
			return false;
		}
		neighbor = next_halfedge(neighbor);
		if same_face(neighbor, &tref) {
			return false;
		}
		neighbor = next_halfedge(pair);
		if same_face(neighbor, &pair_ref) {
			return false;
		}
		neighbor = next_halfedge(neighbor);
		if same_face(neighbor, &pair_ref) {
			return false;
		}
		return true;
	}

	///Returns true if this halfedge is an interior of a quad, as defined by its
	///halfedge tangent having negative weight.
	pub fn is_marked_inside_quad(&self, _halfedge: i32) -> bool {
		// if !self.halfedge_tangent.is_empty() {
		// 	return self.halfedge_tangent[halfedge as usize].w < 0;
		// }
		return false;
	}

	///Find faces containing at least 3 triangles - these will not have
	///interpolated normals - all their vert normals must match their face normal.
	pub fn flat_faces(&self) -> Vec<bool> {
		let num_tri: i32 = self.num_tri() as i32;
		let mut tri_is_flat_face: Vec<bool> = vec![false; num_tri as usize];
		(0..num_tri).into_iter().for_each(|tri| {
			let tref = &self.mesh_relation.tri_ref[tri as usize];
			let mut face_neighbors: i32 = 0;
			let mut face_tris: Vector3<i32> = Vector3::new(-1, -1, -1);
			for j in 0..3 {
				let neighbor_tri = self.halfedge.pair(3 * tri + j) / 3;
				let j_ref = &self.mesh_relation.tri_ref[neighbor_tri as usize];
				if j_ref.same_face(tref) {
					face_neighbors += 1;
					face_tris[j as usize] = neighbor_tri;
				}
			}
			if face_neighbors > 1 {
				tri_is_flat_face[tri as usize] = true;
				for j in [0, 1, 2] {
					if face_tris[j] >= 0 {
						tri_is_flat_face[face_tris[j as usize] as usize] = true;
					}
				}
			}
		});
		return tri_is_flat_face;
	}

	///Returns a vector of length numVert that has a tri that is part of a
	///neighboring flat face if there is only one flat face. If there are none it
	///gets -1, and if there are more than one it gets -2.
	pub fn vert_flat_face(&self, flat_faces: &[bool]) -> Vec<i32> {
		let mut vert_flat_face: Vec<i32> = vec![-1; self.num_vert()];
		let mut vert_ref: Vec<TriRef> = vec![
			TriRef {
				mesh_id: -1,
				original_id: -1,
				face_id: -1,
				coplanar_id: -1,
			};
			self.num_vert()
		];
		for tri in 0..self.num_tri() {
			if flat_faces[tri] {
				for j in 0..3 {
					let vert = self.halfedge.start((3 * tri + j) as i32);
					if vert_ref[vert as usize].same_face(&self.mesh_relation.tri_ref[tri as usize])
					{
						continue;
					}
					vert_ref[vert as usize] = self.mesh_relation.tri_ref[tri as usize];
					vert_flat_face[vert as usize] = if vert_flat_face[vert as usize] == -1 {
						tri as i32
					} else {
						-2
					};
				}
			}
		}
		return vert_flat_face;
	}

	///Instead of calculating the internal shared normals like CalculateNormals
	///does, this method fills in vertex properties, unshared across edges that
	///are bent more than minSharpAngle.
	pub fn set_normals(&mut self, normal_idx: i32, mut min_sharp_angle: f64) {
		if self.is_empty() {
			return;
		}
		if normal_idx < 0 {
			return;
		}
		// Ensure minSharpAngle is large enough to avoid treating nearly-coplanar
		// faces as sharp due to floating point noise in the dihedral computation.
		min_sharp_angle = min_sharp_angle.max(K_MIN_SHARP_ANGLE);

		let old_num_prop = self.num_prop() as i32;

		let mut vert_num_sharp = vec![0; self.num_vert()];
		for e in 0..self.halfedge.len() as i32 {
			if !self.halfedge.is_forward(e) {
				continue;
			}
			let pair = self.halfedge.pair(e);
			let tri1 = e / 3;
			let tri2 = pair / 3;
			let dihedral = angle_between(
				self.face_normal[tri1 as usize],
				self.face_normal[tri2 as usize],
			)
			.to_degrees();
			if dihedral > min_sharp_angle {
				vert_num_sharp[self.halfedge.start(e) as usize] += 1;
				vert_num_sharp[self.halfedge.end(e) as usize] += 1;
			}
		}

		let num_prop: i32 = old_num_prop.max(normal_idx + 3);
		let mut old_properties: Vec<f64> = vec![0.0; num_prop as usize * self.num_prop_vert()];
		core::mem::swap(&mut self.properties, &mut old_properties);
		self.num_prop = num_prop;

		let mut old_halfedge_prop: Vec<i32> = vec![0; self.halfedge.len()];
		(0..self.halfedge.len()).for_each(|i| {
			old_halfedge_prop[i] = self.halfedge.prop(i as i32);
			self.halfedge.set_prop(i as i32, -1);
		});

		// Cached per-meshID inverse-normal-transform for the legacy non-zero
		// normalIdx path. Lazily populated on first lookup; reused across all
		// verts in the loop below.
		// TODO: drop this and its only caller below when the non-zero normalIdx
		// parameter on CalculateNormals is removed.
		let mut mesh_id_to_normal_transform = BTreeMap::new();
		let mut get_transform = |myself: &mut Self, mesh_id: i32| {
			mesh_id_to_normal_transform
				.entry(mesh_id)
				.or_insert_with(|| {
					myself
						.mesh_relation
						.mesh_id_transform
						.get(&mesh_id)
						.unwrap()
						.get_inverse_normal_transform()
				})
				.clone()
		};

		let num_edge = self.halfedge.len() as i32;
		for start_edge in 0..num_edge {
			if self.halfedge.prop(start_edge) >= 0 {
				continue;
			}
			let vert = self.halfedge.start(start_edge) as usize;

			if vert_num_sharp[vert] < 2 {
				// vertex has single normal
				let world_normal = self.vert_normal[vert];
				// Non-zero normalIdx is the legacy deferred-transform path: store in
				// per-mesh frame so GetMeshGL's runTransform application on export
				// recovers world frame even after later transforms. Standard slot 0
				// uses the eager-transform contract: store world-frame directly.
				// Caveat: for legacy idx!=0, if a single propVert is shared between
				// triangles of different meshIDs, we pick startEdge's meshID for the
				// per-mesh-frame mapping. Other meshIDs reading the same propVert
				// through a different runTransform on export will get a wrong
				// rotation. Same shape as master; out of scope here.
				let normal = if normal_idx == 0 {
					world_normal
				} else {
					get_transform(
						self,
						self.mesh_relation.tri_ref[(start_edge / 3) as usize].mesh_id,
					) * world_normal
				};
				let mut last_prop: i32 = -1;
				self.for_vert_mut(start_edge, |self_mut, current| {
					let prop: i32 = old_halfedge_prop[current as usize];
					self_mut.halfedge.set_prop(current, prop);
					if prop == last_prop {
						return;
					}
					last_prop = prop;
					// update property vertex
					let start = &old_properties[(prop * old_num_prop) as usize..];
					self_mut.properties
						[(prop * num_prop) as usize..(prop * num_prop + old_num_prop) as usize]
						.copy_from_slice(&start[..old_num_prop as usize]);
					for i in [0, 1, 2] {
						self_mut.properties[(prop * num_prop + normal_idx + i) as usize] =
							normal[i as usize];
					}
				});
			}

			// vertex has multiple normals
			let center_pos: Vector3<f64> = self.vert_pos[vert as usize].coords;
			// Length degree
			let mut groups: Vec<i32> = vec![];
			// Length number of normals
			let mut normals: Vec<Vector3<f64>> = vec![];
			let mut mesh_ids = vec![];
			let mut current = start_edge;
			let mut prev_face = current / 3;

			loop {
				// find a sharp edge to start on
				let next = next_halfedge(self.halfedge.pair(current));
				let face = next / 3;

				let dihedral = angle_between(
					self.face_normal[face as usize],
					self.face_normal[prev_face as usize],
				)
				.to_degrees();
				if dihedral > min_sharp_angle {
					break;
				}
				current = next;
				prev_face = face;
				if current == start_edge {
					break;
				}
			}

			let end_edge = current;

			struct FaceEdge {
				face: i32,
				normalized_edge: Vector3<f64>,
			}

			// calculate pseudo-normals between each sharp edge
			self.for_vert_fn(
				end_edge,
				|current| {
					let vert = self.halfedge.end(current);
					FaceEdge {
						face: current / 3,
						normalized_edge: safe_normalize(
							(self.vert_pos[vert as usize] - center_pos).coords,
						),
					}
				},
				|_, here: &FaceEdge, next: &mut FaceEdge| {
					let dihedral = angle_between(
						self.face_normal[here.face as usize],
						self.face_normal[next.face as usize],
					)
					.to_degrees();
					if dihedral > min_sharp_angle {
						normals.push(Vector3::default());
						mesh_ids.push(self.mesh_relation.tri_ref[next.face as usize].mesh_id);
					}
					groups.push((normals.len() - 1) as i32);
					if next.normalized_edge.x.is_finite() {
						let dir = safe_normalize(next.normalized_edge.cross(&here.normalized_edge));
						*normals.last_mut().unwrap() +=
							dir * angle_between(here.normalized_edge, next.normalized_edge);
					} else {
						next.normalized_edge = here.normalized_edge;
					}
				},
			);

			for normal in normals.iter_mut() {
				*normal = safe_normalize(normal.clone());
			}
			for i in 0..normals.len() {
				let mut n = normals[i];
				// Same frame-storage rule as the single-normal path above.
				if normal_idx != 0 {
					n = get_transform(self, mesh_ids[i]) * n;
				}
				normals[i] = safe_normalize(n);
			}

			let mut last_group: i32 = 0;
			let mut last_prop: i32 = -1;
			let mut new_prop: i32 = -1;
			let mut idx: i32 = 0;
			self.for_vert_mut(end_edge, |self_mut, current1| {
				let prop: i32 = old_halfedge_prop[current1 as usize];
				let start = &mut old_properties[(prop * old_num_prop) as usize..];

				if groups[idx as usize] != last_group
					&& groups[idx as usize] != 0
					&& prop == last_prop
				{
					// split property vertex, duplicating but with an updated normal
					last_group = groups[idx as usize];
					new_prop = self_mut.num_prop_vert() as i32;
					self_mut
						.properties
						.resize(self_mut.properties.len() + num_prop as usize, 0.0);
					self_mut.properties[(new_prop * num_prop) as usize
						..(new_prop * num_prop + old_num_prop) as usize]
						.copy_from_slice(&start[..old_num_prop as usize]);
					for i in [0, 1, 2] {
						self_mut.properties[(new_prop * num_prop + normal_idx + i) as usize] =
							normals[groups[idx as usize] as usize][i as usize];
					}
				} else if prop != last_prop {
					// update property vertex
					last_prop = prop;
					new_prop = prop;
					self_mut.properties
						[(prop * num_prop) as usize..(prop * num_prop + old_num_prop) as usize]
						.copy_from_slice(&start[..old_num_prop as usize]);
					for i in [0, 1, 2] {
						self_mut.properties[(prop * num_prop + normal_idx + i) as usize] =
							normals[groups[idx as usize] as usize][i as usize];
					}
				}

				// point to updated property vertex
				self_mut.halfedge.set_prop(current1, new_prop);
				idx += 1;
			});
		}
	}
}
