use crate::halfedge::{Halfedges, next_halfedge};
use crate::util::hash_table::DeterministicMap;
use crate::util::math::{atomic_add, safe_normalize3};
use crate::util::vec_ext;
use crate::{MeshBool, Properties, Triangles};
use nalgebra::{Point3, Vector3};
use std::{f64, mem};

// Minimum sharp angle in degrees, below which edges are considered coplanar.
// Floating point noise in the dihedral angle computation can reach ~1e-6
// degrees for nearly-parallel face normals; this threshold must exceed that.
const K_MIN_SHARP_ANGLE: f64 = 1e-4;

impl MeshBool {
	///Create a new copy of this manifold with updated vertex properties by
	///supplying a function that takes the existing position and properties as
	///input. You may specify any number of output properties, allowing creation and
	///removal of channels. Note: undefined behavior will result if you read past
	///the number of input properties or write past the number of output properties.
	///
	///If prop_func is a None, this function will just set the channel to zeroes.
	///
	///Any normals recording set by `CalculateNormals()` is preserved. If the new
	///properties overwrite slot 0..2 with non-normal data, the recording becomes
	///stale; re-call `CalculateNormals()` (or use a numProp < 3 call followed by
	///CalculateNormals) to reset.
	///
	///@param prop_stride The new number of properties per vertex.
	///@param prop_func A function that modifies the properties of a given vertex.
	pub fn set_properties(
		&self,
		prop_stride: usize,
		prop_func: Option<impl FnMut(&mut [f64], Point3<f64>, &[f64])>,
	) -> Self {
		let old_prop_stride = self.prop_stride();

		let mut halfedge = self.tri.halfedge.clone();
		if old_prop_stride == 0 && prop_stride > 0 {
			//workaround for removal of logic here:
			//https://github.com/elalish/manifold/blob/51f178f012a2951734bbe4583b384066300e317f/src/sort.cpp#L354-L356
			halfedge.init_prop_from_start();
		}

		let properties = if prop_stride == 0 {
			Vec::new()
		} else {
			let mut properties = vec![0.0; prop_stride as usize * self.num_prop_vert()];

			if let Some(mut prop_func) = prop_func {
				for tri in 0..self.num_tri() {
					for i in 0..3 {
						let edge = (3 * tri + i) as i32;
						let vert = halfedge.start(edge) as usize;
						let prop_vert = halfedge.prop(edge) as usize;
						prop_func(
							&mut properties
								[(prop_stride * prop_vert)..(prop_stride * (prop_vert + 1))],
							self.vert_pos[vert],
							&self.properties.data[(old_prop_stride * prop_vert)
								..(old_prop_stride * (prop_vert + 1))],
						);
					}
				}
			}

			properties
		};

		return Self {
			original_id: None,
			precision: self.precision.clone(),
			vert_pos: self.vert_pos.clone(),
			properties: Properties {
				data: properties,
				stride: prop_stride,
			},
			tri: Triangles {
				halfedge,
				normal: self.tri.normal.clone(),
				relation: self.tri.relation.clone(),
			},
			instance_relation: self.instance_relation.clone(),
			collider: self.collider.clone(),
		};
	}

	///Fills in vertex properties for normal vectors, calculated from the mesh
	///geometry.
	///
	///@param normalIdx The property channel in which to store the X values of the
	///normals. The X, Y, and Z channels will be sequential. The property set will
	///be automatically expanded such that NumProp will be at least normalIdx + 3.
	///Default is 0, the standard slot (MeshGL channels 3, 4, 5); the Manifold
	///records the recording per-meshID in that case so subsequent `GetMeshGL()` /
	///`GetMeshGL64()` without an explicit normalIdx will return world-frame
	///normals and mark each output run via runFlags bit 1. Non-zero values are
	///retained for compatibility and will not be supported in a future release.
	///
	///@param minSharpAngle Any edges with angles greater than this value will
	///remain sharp, getting different normal vector properties on each side of the
	///edge. By default, no edges are sharp and all normals are shared. With a value
	///of zero, the model is faceted and all normals match their triangle normals,
	///but in this case it would be better not to calculate normals at all.
	pub fn calculate_normals(&self, normal_idx: usize, mut min_sharp_angle: f64) -> Self {
		// Mark per-meshID hasNormals so GetMeshGL(-1) can auto-substitute slot 0 on
		// export. Restricted to the standard slot since a non-zero slot would be
		// ambiguous when round-tripping through MeshGL.
		let instance_relation = self
			.instance_relation
			.iter()
			.map(|&rel| {
				let mut rel = rel;
				if normal_idx == 0 {
					rel.has_normals = true;
				}

				rel
			})
			.collect();

		let old_prop_stride = self.prop_stride();
		let prop_stride = old_prop_stride.max(normal_idx + 3);

		if self.is_empty() {
			return Self {
				original_id: None,
				precision: self.precision,
				vert_pos: self.vert_pos.clone(),
				properties: self.properties.clone(),
				tri: self.tri.clone(),
				instance_relation,
				collider: self.collider.clone(),
			};
		}

		// Ensure minSharpAngle is large enough to avoid treating nearly-coplanar
		// faces as sharp due to floating point noise in the dihedral computation.
		min_sharp_angle = min_sharp_angle.max(K_MIN_SHARP_ANGLE);

		let mut vert_num_sharp = vec![0; self.num_vert()];
		for e in 0..self.tri.halfedge.len() as i32 {
			if !self.tri.halfedge.is_forward(e) {
				continue;
			}
			let pair = self.tri.halfedge.pair(e);
			let tri1 = e / 3;
			let tri2 = pair / 3;
			let dihedral = self.tri.normal[tri1 as usize]
				.angle(&self.tri.normal[tri2 as usize])
				.to_degrees();
			if dihedral > min_sharp_angle {
				vert_num_sharp[self.tri.halfedge.start(e) as usize] += 1;
				vert_num_sharp[self.tri.halfedge.end(e) as usize] += 1;
			}
		}

		let mut halfedge = self.tri.halfedge.clone();
		let old_halfedge_prop = Vec::from_iter((0..self.tri.halfedge.len() as i32).map(|i| {
			halfedge.set_prop(i, -1);
			if old_prop_stride > 0 {
				self.tri.halfedge.prop(i)
			} else {
				//workaround for removal of logic here:
				//https://github.com/elalish/manifold/blob/51f178f012a2951734bbe4583b384066300e317f/src/sort.cpp#L354-L356
				self.tri.halfedge.start(i)
			}
		}));

		// Cached per-meshID inverse-normal-transform for the legacy non-zero
		// normalIdx path. Lazily populated on first lookup; reused across all
		// verts in the loop below.
		// TODO: drop this and its only caller below when the non-zero normalIdx
		// parameter on CalculateNormals is removed.
		let mut instance_id_to_normal_transform = DeterministicMap::new();
		let mut get_transform = |instance_id: u32| {
			instance_id_to_normal_transform
				.entry(instance_id)
				.or_insert_with(|| {
					instance_relation[instance_id as usize].get_inverse_normal_transform()
				})
				.clone()
		};

		let num_edge = self.tri.halfedge.len() as i32;
		let vert_normal = self.calculate_vert_normals_internal();
		let mut properties =
			unsafe { vec_ext::uninit(prop_stride as usize * self.num_prop_vert()) };
		for start_edge in 0..num_edge {
			if halfedge.prop(start_edge) >= 0 {
				continue;
			}
			let vert = self.tri.halfedge.start(start_edge) as usize;

			if vert_num_sharp[vert] < 2 {
				// vertex has single normal
				let world_normal = vert_normal[vert];
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
					get_transform(self.tri.relation[(start_edge / 3) as usize].instance_id)
						* world_normal
				};
				let mut last_prop = None;
				halfedge.for_vert_mut(start_edge, |halfedge, current| {
					let prop = old_halfedge_prop[current as usize];
					halfedge.set_prop(current, prop);
					let prop = prop as usize;
					if Some(prop) == last_prop {
						return;
					}
					last_prop = Some(prop);
					// update property vertex
					let start = &self.properties.data[(prop * old_prop_stride)..];
					properties[(prop * prop_stride)..(prop * prop_stride + old_prop_stride)]
						.copy_from_slice(&start[..old_prop_stride]);
					for i in 0..3 {
						properties[prop * prop_stride + normal_idx + i] = normal[i];
					}
				});
				continue;
			}

			// vertex has multiple normals
			let center_pos: Vector3<f64> = self.vert_pos[vert].coords;
			// Length degree
			let mut groups: Vec<i32> = vec![];
			// Length number of normals
			let mut normals: Vec<Vector3<f64>> = vec![];
			let mut instance_ids = vec![];
			let mut current = start_edge;
			let mut prev_face = current / 3;

			loop {
				// find a sharp edge to start on
				let next = next_halfedge(self.tri.halfedge.pair(current));
				let face = next / 3;

				let dihedral = self.tri.normal[face as usize]
					.angle(&self.tri.normal[prev_face as usize])
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
			self.tri.halfedge.for_vert_fn(
				end_edge,
				|current| {
					let vert = self.tri.halfedge.end(current);
					FaceEdge {
						face: current / 3,
						normalized_edge: safe_normalize3(
							(self.vert_pos[vert as usize] - center_pos).coords,
						),
					}
				},
				|_, here: &FaceEdge, next: &mut FaceEdge| {
					let dihedral = self.tri.normal[here.face as usize]
						.angle(&self.tri.normal[next.face as usize])
						.to_degrees();
					if dihedral > min_sharp_angle {
						normals.push(Vector3::default());
						instance_ids.push(self.tri.relation[next.face as usize].instance_id);
					}
					groups.push((normals.len() - 1) as i32);
					if next.normalized_edge.x.is_finite() {
						let dir =
							safe_normalize3(next.normalized_edge.cross(&here.normalized_edge));
						*normals.last_mut().unwrap() +=
							dir * here.normalized_edge.angle(&next.normalized_edge);
					} else {
						next.normalized_edge = here.normalized_edge;
					}
				},
			);

			for i in 0..normals.len() {
				let mut n = normals[i];
				// Same frame-storage rule as the single-normal path above.
				if normal_idx != 0 {
					n = get_transform(instance_ids[i]) * n;
				}
				normals[i] = safe_normalize3(n);
			}

			let mut last_group = 0;
			let mut last_prop = None;
			let mut new_prop_vert = 0;
			let mut idx = 0;
			halfedge.for_vert_mut(end_edge, |halfedge, current1| {
				let prop = old_halfedge_prop[current1 as usize] as usize;
				let start = &self.properties.data[(prop * old_prop_stride)..];

				if groups[idx] != last_group && groups[idx] != 0 && Some(prop) == last_prop {
					// split property vertex, duplicating but with an updated normal
					last_group = groups[idx];
					new_prop_vert = properties.len() / prop_stride;
					properties.resize(properties.len() + prop_stride, 0.0);
					properties[(new_prop_vert * prop_stride)
						..(new_prop_vert * prop_stride + old_prop_stride)]
						.copy_from_slice(&start[..old_prop_stride]);
					for i in 0..3 {
						properties[new_prop_vert * prop_stride + normal_idx + i] =
							normals[groups[idx] as usize][i];
					}
				} else if Some(prop) != last_prop {
					// update property vertex
					last_prop = Some(prop);
					new_prop_vert = prop;
					properties[(prop * prop_stride)..(prop * prop_stride + old_prop_stride)]
						.copy_from_slice(&start[..old_prop_stride]);
					for i in 0..3 {
						properties[prop * prop_stride + normal_idx + i] =
							normals[groups[idx] as usize][i];
					}
				}

				// point to updated property vertex
				halfedge.set_prop(current1, new_prop_vert as i32);
				idx += 1;
			});
		}

		Self {
			original_id: None,
			precision: self.precision,
			vert_pos: self.vert_pos.clone(),
			properties: Properties {
				data: properties,
				stride: prop_stride,
			},
			tri: Triangles {
				halfedge,
				normal: self.tri.normal.clone(),
				relation: self.tri.relation.clone(),
			},
			instance_relation,
			collider: self.collider.clone(),
		}
	}

	///This function uses the face normals to compute
	///vertex normals (angle-weighted pseudo-normals). Face normals should only be
	///calculated when needed because nearly degenerate faces will accrue rounding
	///error, while the Boolean can retain their original normal, which is more
	///accurate and can help with merging coplanar faces.
	pub(crate) fn calculate_vert_normals_internal(&self) -> Vec<Vector3<f64>> {
		let num_vert = self.vert_pos.len();
		let mut vert_normal = vec![Vector3::default(); num_vert];

		let mut vert_halfedge_map = vec![i32::MAX; self.num_vert()];

		let mut atomic_min = |value, vert: i32| {
			if vert < 0 {
				return;
			}
			let current = &mut vert_halfedge_map[vert as usize];
			if value < *current {
				*current = value;
			}
		};

		for i in 0..self.tri.halfedge.len() as i32 {
			atomic_min(i, self.tri.halfedge.start(i));
		}

		for vert in 0..num_vert {
			let first_edge = vert_halfedge_map[vert];
			// not referenced
			if first_edge == i32::MAX {
				vert_normal[vert] = Vector3::from_element(0.0);
				continue;
			}

			let mut normal = Vector3::from_element(0.0);
			self.tri.halfedge.for_vert(first_edge, |edge| {
				let tri_verts = Vector3::new(
					self.tri.halfedge.start(edge),
					self.tri.halfedge.end(edge),
					self.tri.halfedge.end(next_halfedge(edge)),
				);
				let curr_edge = (self.vert_pos[tri_verts[1] as usize]
					- self.vert_pos[tri_verts[0] as usize])
					.normalize();
				let prev_edge = (self.vert_pos[tri_verts[0] as usize]
					- self.vert_pos[tri_verts[2] as usize])
					.normalize();

				// if it is not finite, this means that the triangle is degenerate, and we
				// should just exclude it from the normal calculation...
				if !curr_edge[0].is_finite() || !prev_edge[0].is_finite() {
					return;
				}
				let dot = -prev_edge.dot(&curr_edge);
				let phi = if dot >= 1.0 {
					0.0
				} else if dot <= -1.0 {
					f64::consts::PI
				} else {
					libm::acos(dot)
				};
				normal += phi * self.tri.normal[(edge / 3) as usize];
			});

			vert_normal[vert] = safe_normalize3(normal);
		}

		vert_normal
	}

	///Curvature is the inverse of the radius of curvature, and signed such that
	///positive is convex and negative is concave. There are two orthogonal
	///principal curvatures at any point on a manifold, with one maximum and the
	///other minimum. Gaussian curvature is their product, while mean
	///curvature is their sum. This approximates them for every vertex and assigns
	///them as vertex properties on the given channels.
	///
	///@param gaussian_idx The property channel index in which to store the Gaussian
	///curvature. An index < 0 will be ignored (stores nothing). The property set
	///will be automatically expanded to include the channel index specified.
	///
	///@param mean_idx The property channel index in which to store the mean
	///curvature. An index < 0 will be ignored (stores nothing). The property set
	///will be automatically expanded to include the channel index specified.
	pub fn calculate_curvature(
		&self,
		gaussian_idx: Option<usize>,
		mean_idx: Option<usize>,
	) -> Self {
		if self.is_empty() || (gaussian_idx.is_none() && mean_idx.is_none()) {
			return self.clone();
		}
		let mut vert_mean_curvature = vec![0.0; self.num_vert()];
		let mut vert_gaussian_curvature = vec![f64::consts::TAU; self.num_vert()];
		let mut vert_area = vec![0.0; self.num_vert()];
		let mut degree = vec![0.0; self.num_vert()];
		{
			let mut ca = CurvatureAngles {
				mean_curvature: &mut vert_mean_curvature,
				gaussian_curvature: &mut vert_gaussian_curvature,
				area: &mut vert_area,
				degree: &mut degree,
				halfedge: &self.tri.halfedge,
				vert_pos: &self.vert_pos,
				tri_normal: &self.tri.normal,
			};
			(0..self.num_tri()).for_each(|i| ca.call(i));
		}
		(0..self.num_vert()).for_each(|vert| {
			let factor: f64 = degree[vert] / (6.0 * vert_area[vert]);
			vert_mean_curvature[vert] *= factor;
			vert_gaussian_curvature[vert] *= factor;
		});

		let old_prop_stride = self.prop_stride();
		let prop_stride = old_prop_stride.max(gaussian_idx.max(mean_idx).unwrap_or(0) + 1);
		let mut properties = vec![0.0; prop_stride as usize * self.num_prop_vert()];

		let mut halfedge = self.tri.halfedge.clone();
		if old_prop_stride == 0 {
			//workaround for removal of logic here:
			//https://github.com/elalish/manifold/blob/51f178f012a2951734bbe4583b384066300e317f/src/sort.cpp#L354-L356
			halfedge.init_prop_from_start();
		}

		let mut counters: Vec<bool> = vec![false; self.num_prop_vert()];
		for tri in 0..self.num_tri() {
			for i in 0..3 {
				let edge = (3 * tri + i) as i32;
				let vert = halfedge.start(edge) as usize;
				let prop_vert = halfedge.prop(edge) as usize;

				let old = mem::replace(&mut counters[prop_vert], true);
				if old {
					continue;
				}

				for p in 0..old_prop_stride {
					properties[prop_stride * prop_vert + p] =
						self.properties.data[old_prop_stride * prop_vert + p];
				}

				if let Some(gaussian_idx) = gaussian_idx {
					properties[prop_stride * prop_vert + gaussian_idx] =
						vert_gaussian_curvature[vert];
				}
				if let Some(mean_idx) = mean_idx {
					properties[prop_stride * prop_vert + mean_idx] = vert_mean_curvature[vert];
				}
			}
		}

		Self {
			original_id: None,
			precision: self.precision.clone(),
			vert_pos: self.vert_pos.clone(),
			properties: Properties {
				data: properties,
				stride: prop_stride,
			},
			tri: Triangles {
				halfedge,
				normal: self.tri.normal.clone(),
				relation: self.tri.relation.clone(),
			},
			instance_relation: self.instance_relation.clone(),
			collider: self.collider.clone(),
		}
	}
}

struct CurvatureAngles<'a> {
	mean_curvature: &'a mut [f64],
	gaussian_curvature: &'a mut [f64],
	area: &'a mut [f64],
	degree: &'a mut [f64],
	halfedge: &'a Halfedges,
	vert_pos: &'a [Point3<f64>],
	tri_normal: &'a [Vector3<f64>],
}

impl<'a> CurvatureAngles<'a> {
	pub fn call(&mut self, tri: usize) {
		let mut edge: [Vector3<f64>; 3] = Default::default();
		let mut edge_length = Vector3::repeat(0.0_f64);
		for i in 0..3 {
			let edge_idx = (3 * tri + i) as i32;
			let start_vert = self.halfedge.start(edge_idx);
			let end_vert = self.halfedge.end(edge_idx);
			edge[i] = self.vert_pos[end_vert as usize] - self.vert_pos[start_vert as usize];
			edge_length[i] = edge[i].norm();
			edge[i] /= edge_length[i];
			let neighbor_tri = self.halfedge.pair(edge_idx) / 3;
			let dihedral = 0.25
				* edge_length[i]
				* libm::asin(
					self.tri_normal[tri]
						.cross(&self.tri_normal[neighbor_tri as usize])
						.dot(&edge[i]),
				);
			atomic_add(&mut self.mean_curvature[start_vert as usize], dihedral);
			atomic_add(&mut self.mean_curvature[end_vert as usize], dihedral);
			atomic_add(&mut self.degree[start_vert as usize], 1.0);
		}

		let mut phi = Vector3::<f64>::default();
		phi[0] = libm::acos(-edge[2].dot(&edge[0]));
		phi[1] = libm::acos(-edge[0].dot(&edge[1]));
		phi[2] = core::f64::consts::PI - phi[0] - phi[1];
		let area3: f64 = edge_length[0] * edge_length[1] * edge[0].cross(&edge[1]).norm() / 6.0;

		for i in 0..3 {
			let vert: i32 = self.halfedge.start((3 * tri + i) as i32);
			atomic_add(&mut self.gaussian_curvature[vert as usize], -phi[i]);
			atomic_add(&mut self.area[vert as usize], area3);
		}
	}
}
