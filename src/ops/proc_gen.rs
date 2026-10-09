use crate::Triangles;
use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation, reserve_original_id};
use crate::postprocessing as pp;
use crate::triangulation::{
	PolyVert, Polygons, PolygonsIdx, SimplePolygon, SimplePolygonIdx, triangulate, triangulate_idx,
};
use crate::util::math::{cosd, sind};
use crate::util::segment_resolution::SegmentResolution;
use crate::{Box3D, MeshBool, Precision, Properties, TrianglesPartial};
use nalgebra::{Matrix2, Matrix3x4, Point2, Point3, Vector2, Vector3};
use std::f64::consts::FRAC_PI_2;
use std::rc::Rc;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ConstructorError {
	InvalidConstruction,
}

impl MeshBool {
	///Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
	///and the rest at similarly symmetric points.
	pub fn tetrahedron() -> Self {
		let vert_pos = vec![
			Point3::new(-1.0, -1.0, 1.0),
			Point3::new(-1.0, 1.0, -1.0),
			Point3::new(1.0, -1.0, -1.0),
			Point3::new(1.0, 1.0, 1.0),
		];

		let tri_verts = vec![
			Vector3::new(2, 0, 1),
			Vector3::new(0, 3, 1),
			Vector3::new(2, 3, 0),
			Vector3::new(3, 2, 1),
		];

		MeshBool::from_tri_mesh(vert_pos, tri_verts, None)
	}

	///Constructs a unit cube (edge lengths all one), by default in the first
	///octant, touching the origin. If any dimensions in size are negative, or if
	///all are zero, an empty Manifold will be returned.
	///
	///@param size The X, Y, and Z dimensions of the box.
	///@param center Set to true to shift the center to the origin.
	pub fn cube(size: Vector3<f64>, center: bool) -> Result<Self, ConstructorError> {
		if size.x < 0.0 || size.y < 0.0 || size.z < 0.0 || size.magnitude_squared() == 0.0 {
			return Err(ConstructorError::InvalidConstruction);
		}

		let vert_pos = vec![
			Point3::new(0.0, 0.0, 0.0),
			Point3::new(0.0, 0.0, 1.0),
			Point3::new(0.0, 1.0, 0.0),
			Point3::new(0.0, 1.0, 1.0),
			Point3::new(1.0, 0.0, 0.0),
			Point3::new(1.0, 0.0, 1.0),
			Point3::new(1.0, 1.0, 0.0),
			Point3::new(1.0, 1.0, 1.0),
		];

		let tri_verts = vec![
			Vector3::new(1, 0, 4),
			Vector3::new(2, 4, 0),
			Vector3::new(1, 3, 0),
			Vector3::new(3, 1, 5),
			Vector3::new(3, 2, 0),
			Vector3::new(3, 7, 2),
			Vector3::new(5, 4, 6),
			Vector3::new(5, 1, 4),
			Vector3::new(6, 4, 2),
			Vector3::new(7, 6, 2),
			Vector3::new(7, 3, 5),
			Vector3::new(7, 5, 6),
		];

		let m = Matrix3x4::from_columns(&[
			Vector3::new(size.x, 0.0, 0.0),
			Vector3::new(0.0, size.y, 0.0),
			Vector3::new(0.0, 0.0, size.z),
			if center {
				-size / 2.0
			} else {
				Vector3::zeros()
			},
		]);

		Ok(MeshBool::from_tri_mesh(vert_pos, tri_verts, Some(m)))
	}

	///A convenience constructor for the common case of extruding a circle. Can also
	///form cones if both radii are specified.
	///
	///@param height Z-extent
	///@param radiusLow Radius of bottom circle. Must be positive.
	///@param radiusHigh Radius of top circle. Can equal zero. Default is equal to
	///radiusLow.
	///@param circularSegments How many line segments to use around the circle.
	///Default is calculated by the static Defaults.
	///@param center Set to true to shift the center to the origin. Default is
	///origin at the bottom.
	pub fn cylinder(
		height: f64,
		radius_low: f64,
		radius_high: f64,
		quality: SegmentResolution,
		center: bool,
	) -> Result<Self, ConstructorError> {
		if height <= 0.0 || radius_low < 0.0 {
			return Err(ConstructorError::InvalidConstruction);
		}
		if radius_low == 0.0 {
			if radius_high <= 0.0 {
				return Err(ConstructorError::InvalidConstruction);
			}
			// Cone with apex at bottom: create the centered apex-at-top version and
			// mirror it
			let mut cone = MeshBool::cylinder(height, radius_high, 0.0, quality, true)?;
			cone = cone.mirror(Vector3::new(0.0, 0.0, 1.0)).eval().unwrap();
			if !center {
				cone = cone
					.translate(Vector3::new(0.0, 0.0, height / 2.0))
					.eval()
					.unwrap();
			}
			return Ok(cone.as_original());
		}
		let scale = if radius_high >= 0.0 {
			radius_high / radius_low
		} else {
			1.0
		};
		let radius = radius_low.max(radius_high);
		let n = quality.get_circular_segments(radius).max(3);

		let mut circle: SimplePolygon = vec![Point2::default(); n as usize];
		let d_phi = 360.0 / (n as f64);
		for i in 0..n {
			circle[i as usize] = Point2::new(
				radius_low * cosd(d_phi * i as f64),
				radius_low * sind(d_phi * i as f64),
			);
		}

		let cylinder = Self::extrude(&vec![circle], height, 0, 0.0, Vector2::new(scale, scale))?;

		Ok(if center {
			cylinder
				.translate(Vector3::new(0.0, 0.0, -height / 2.0))
				.eval()
				.unwrap()
				.as_original()
		} else {
			cylinder
		})
	}

	///Constructs a geodesic sphere of a given radius.
	///
	///@param radius Radius of the sphere. Must be positive.
	///@param circularSegments Number of segments along its
	///diameter. This number will always be rounded up to the nearest factor of
	///four, as this sphere is constructed by refining an octahedron. This means
	///there are a circle of vertices on all three of the axis planes. Default is
	///calculated by the static Defaults.
	pub fn sphere(radius: f64, quality: SegmentResolution) -> Result<Self, ConstructorError> {
		if radius <= 0.0 {
			return Err(ConstructorError::InvalidConstruction);
		}

		let vert_pos = vec![
			Point3::new(1.0, 0.0, 0.0),
			Point3::new(-1.0, 0.0, 0.0),
			Point3::new(0.0, 1.0, 0.0),
			Point3::new(0.0, -1.0, 0.0),
			Point3::new(0.0, 0.0, 1.0),
			Point3::new(0.0, 0.0, -1.0),
		];

		let tri_verts = vec![
			Vector3::new(0, 2, 4),
			Vector3::new(1, 5, 3),
			Vector3::new(2, 1, 4),
			Vector3::new(3, 5, 0),
			Vector3::new(1, 3, 4),
			Vector3::new(0, 5, 2),
			Vector3::new(3, 0, 4),
			Vector3::new(2, 5, 1),
		];

		let octahedron = Self::from_tri_mesh(vert_pos, tri_verts, None);

		let n = if quality.circular_segments > 0 {
			quality.circular_segments + 3
		} else {
			quality.get_circular_segments(radius)
		} / 4;

		let (sphere, _) = octahedron.subdivide(|_, _, _| (n - 1) as i32, false);
		let mut vert_pos = Rc::try_unwrap(sphere.vert_pos).unwrap();
		for v in vert_pos.iter_mut() {
			let v_vec = Vector3::new(
				libm::cos(FRAC_PI_2 * (1.0 - v.x)),
				libm::cos(FRAC_PI_2 * (1.0 - v.y)),
				libm::cos(FRAC_PI_2 * (1.0 - v.z)),
			);

			*v = (radius * v_vec.normalize()).into();
			if v.x.is_nan() {
				*v = Point3::default();
			}
		}

		Ok(Self::from_halfedges(
			vert_pos,
			Rc::try_unwrap(sphere.tri.halfedge).unwrap(),
		))
	}

	///Constructs a manifold from a set of polygons by extruding them along the
	///Z-axis.
	///Note that high twistDegrees with small nDivisions may cause
	///self-intersection. This is not checked here and it is up to the user to
	///choose the correct parameters.
	///
	///@param crossSection A set of non-overlapping polygons to extrude.
	///@param height Z-extent of extrusion.
	///@param nDivisions Number of extra copies of the crossSection to insert into
	///the shape vertically; especially useful in combination with twistDegrees to
	///avoid interpolation artifacts. Default is none.
	///@param twistDegrees Amount to twist the top crossSection relative to the
	///bottom, interpolated linearly for the divisions in between.
	///@param scaleTop Amount to scale the top (independently in X and Y). If the
	///scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
	///Note that scale is applied after twist.
	///Default {1, 1}.
	pub fn extrude(
		cross_section: &Polygons,
		height: f64,
		mut n_divisions: u32,
		twist_degrees: f64,
		mut scale_top: Vector2<f64>,
	) -> Result<Self, ConstructorError> {
		if cross_section.len() == 0 || height <= 0.0 {
			return Err(ConstructorError::InvalidConstruction);
		}

		scale_top = scale_top.sup(&Vector2::new(0.0, 0.0));

		let mut vert_pos = Vec::new();
		n_divisions += 1;
		let mut tri_verts: Vec<Vector3<i32>> = Vec::new();
		let mut n_cross_section = 0;
		let is_cone = scale_top.x == 0.0 && scale_top.y == 0.0;
		let mut idx = 0;
		let mut polygons_indexed: PolygonsIdx = Vec::new();
		for poly in cross_section {
			n_cross_section += poly.len();
			let mut simple_indexed: SimplePolygonIdx = Vec::new();
			for poly_vert in poly {
				vert_pos.push(Point3::new(poly_vert.x, poly_vert.y, 0.0));
				simple_indexed.push(PolyVert {
					pos: *poly_vert,
					idx,
				});
				idx += 1;
			}

			polygons_indexed.push(simple_indexed);
		}

		if n_cross_section == 0 {
			return Err(ConstructorError::InvalidConstruction);
		}

		for i in 1..(n_divisions + 1) {
			let alpha = (i as f64) / (n_divisions as f64);
			let phi = alpha * twist_degrees;
			let scale = Vector2::new(1.0, 1.0).lerp(&scale_top, alpha);
			let rotation = Matrix2::new(cosd(phi), -sind(phi), sind(phi), cosd(phi));
			let transform = Matrix2::new(scale.x, 0.0, 0.0, scale.y) * rotation;
			let mut j = 0;
			let mut idx = 0;
			for poly in cross_section {
				for vert in 0..poly.len() {
					let offset = idx + n_cross_section * i as usize;
					let this_vert = vert + offset;
					let last_vert = (if vert == 0 { poly.len() } else { vert }) - 1 + offset;
					if i == n_divisions && is_cone {
						tri_verts.push(Vector3::new(
							(n_cross_section * (i as usize) + j) as i32,
							(last_vert - n_cross_section) as i32,
							(this_vert - n_cross_section) as i32,
						));
					} else {
						let pos = transform * poly[vert];
						vert_pos.push(Point3::new(pos.x, pos.y, height * alpha));
						tri_verts.push(Vector3::new(
							this_vert as i32,
							last_vert as i32,
							(this_vert - n_cross_section) as i32,
						));
						tri_verts.push(Vector3::new(
							last_vert as i32,
							(last_vert - n_cross_section) as i32,
							(this_vert - n_cross_section) as i32,
						));
					}
				}

				j += 1;
				idx += poly.len();
			}
		}

		if is_cone {
			for _ in 0..cross_section.len()
			// Duplicate vertex for Genus
			{
				vert_pos.push(Point3::new(0.0, 0.0, height));
			}
		}

		let top = triangulate_idx(&polygons_indexed, -1.0, true);
		for tri in &top {
			tri_verts.push(Vector3::new(tri[0], tri[2], tri[1]));
			if !is_cone {
				tri_verts.push(tri.add_scalar((n_cross_section as i32) * (n_divisions as i32)));
			}
		}

		Ok(MeshBool::from_tri_mesh(vert_pos, tri_verts, None))
	}

	///Constructs a manifold from a set of polygons by revolving this cross-section
	///around its Y-axis and then setting this as the Z-axis of the resulting
	///manifold. If the polygons cross the Y-axis, only the part on the positive X
	///side is used. Geometrically valid input will result in geometrically valid
	///output.
	///
	///@param crossSection A set of non-overlapping polygons to revolve.
	///@param circularSegments Number of segments along its diameter. Default is
	///calculated by the static Defaults.
	///@param revolveDegrees Number of degrees to revolve. Default is 360 degrees.
	pub fn revolve(
		cross_section: &Polygons,
		quality: SegmentResolution,
		mut revolve_degrees: f64,
	) -> Result<Self, ConstructorError> {
		let mut polygons: Polygons = vec![];
		let mut radius: f64 = 0.0;
		for poly in cross_section.iter() {
			let mut i: usize = 0;
			while i < poly.len() && poly[i].x < 0.0 {
				i += 1;
			}
			if i == poly.len() {
				continue;
			}
			polygons.push(Vec::default());
			let start: usize = i;
			loop {
				if poly[i].x >= 0.0 {
					polygons.last_mut().unwrap().push(poly[i]);
					radius = radius.max(poly[i].x);
				}
				let next: usize = if i + 1 == poly.len() { 0 } else { i + 1 };
				if (poly[next].x < 0.0) != (poly[i].x < 0.0) {
					let y: f64 = poly[next].y
						- poly[next].x * (poly[i].y - poly[next].y) / (poly[i].x - poly[next].x);
					polygons.last_mut().unwrap().push(Point2::new(0.0, y));
				}
				i = next;
				if i == start {
					break;
				}
			}
		}

		if polygons.is_empty() {
			return Err(ConstructorError::InvalidConstruction);
		}

		if revolve_degrees > 360.0 {
			revolve_degrees = 360.0;
		}
		let is_full_revolution = revolve_degrees == 360.0;

		let n_divisions = ((quality.get_circular_segments(radius) as f64 * revolve_degrees / 360.0)
			as u32)
			.max(3);

		let mut vert_pos = Vec::new();
		let mut tri_verts: Vec<Vector3<i32>> = vec![];

		let mut start_poses: Vec<i32> = vec![];
		let mut end_poses: Vec<i32> = vec![];

		let d_phi: f64 = revolve_degrees / n_divisions as f64;
		// first and last slice are distinguished if not a full revolution.
		let n_slices = if is_full_revolution {
			n_divisions
		} else {
			n_divisions + 1
		};

		for poly in polygons.iter() {
			let mut n_pos_verts = 0;
			let mut n_revolve_axis_verts = 0;
			for pt in poly.iter() {
				if pt.x > 0.0 {
					n_pos_verts += 1;
				} else {
					n_revolve_axis_verts += 1;
				}
			}

			for poly_vert in 0..poly.len() {
				let start_pos_index = vert_pos.len() as u32;

				if !is_full_revolution {
					start_poses.push(start_pos_index as i32);
				}

				let curr_poly_vertex: Vector2<f64> = poly[poly_vert].coords;
				let prev_poly_vertex: Vector2<f64> = poly[if poly_vert == 0 {
					poly.len() - 1
				} else {
					poly_vert - 1
				}]
				.coords;

				let prev_start_pos_index = start_pos_index
					+ (if poly_vert == 0 {
						n_revolve_axis_verts + (n_slices * n_pos_verts)
					} else {
						0
					}) - (if prev_poly_vertex.x == 0.0 {
					1
				} else {
					n_slices
				});

				for slice in 0..n_slices {
					let phi: f64 = slice as f64 * d_phi;
					if slice == 0 || curr_poly_vertex.x > 0.0 {
						vert_pos.push(Point3::new(
							curr_poly_vertex.x * cosd(phi),
							curr_poly_vertex.x * sind(phi),
							curr_poly_vertex.y,
						));
					}

					if is_full_revolution || slice > 0 {
						let last_slice = (if slice == 0 { n_divisions } else { slice }) - 1;
						if curr_poly_vertex.x > 0.0 {
							tri_verts.push(
								Vector3::new(
									start_pos_index + slice,
									start_pos_index + last_slice,
									// "Reuse" vertex of first slice if it lies on the revolve axis
									if prev_poly_vertex.x == 0.0 {
										prev_start_pos_index
									} else {
										prev_start_pos_index + last_slice
									},
								)
								.map(|x| x as i32),
							);
						}

						if prev_poly_vertex.x > 0.0 {
							tri_verts.push(
								Vector3::new(
									prev_start_pos_index + last_slice,
									prev_start_pos_index + slice,
									if curr_poly_vertex.x == 0.0 {
										start_pos_index
									} else {
										start_pos_index + slice
									},
								)
								.map(|x| x as i32),
							);
						}
					}
				}
				if !is_full_revolution {
					end_poses.push(vert_pos.len() as i32 - 1);
				}
			}
		}

		// Add front and back triangles if not a full revolution.
		if !is_full_revolution {
			let front_triangles: Vec<Vector3<i32>> = triangulate(&polygons, -1.0, true);
			for t in front_triangles.iter() {
				tri_verts.push(Vector3::new(
					start_poses[t.x as usize],
					start_poses[t.y as usize],
					start_poses[t.z as usize],
				));
			}

			for t in front_triangles.iter() {
				tri_verts.push(Vector3::new(
					end_poses[t.z as usize],
					end_poses[t.y as usize],
					end_poses[t.x as usize],
				));
			}
		}

		Ok(Self::from_tri_mesh(vert_pos, tri_verts, None))
	}

	fn from_tri_mesh(
		mut vert_pos: Vec<Point3<f64>>,
		tri_vert: Vec<Vector3<i32>>,
		m: Option<Matrix3x4<f64>>,
	) -> Self {
		if let Some(m) = m {
			for v in vert_pos.iter_mut() {
				v.coords = m * v.coords.push(1.0);
			}
		}

		let vert_count = vert_pos.len();
		Self::from_halfedges(
			vert_pos,
			Halfedges::from_tri_indices(vert_count, tri_vert, None),
		)
	}

	fn from_halfedges(mut vert_pos: Vec<Point3<f64>>, mut halfedge: Halfedges) -> Self {
		let original_id = reserve_original_id();
		let bbox = Box3D::from_cloud(&vert_pos);
		let precision = Precision::from_box(bbox);
		let mut properties = Properties::default();
		let collider = pp::sort_and_compact_geometry(
			&mut vert_pos,
			&mut properties,
			TrianglesPartial {
				halfedge: &mut halfedge,
				normal: None,
				relation: None,
			},
			bbox,
		)
		.unwrap();
		let mut tri_rel = vec![TriRelation::default(); halfedge.num_tri()];
		let instance_rel = vec![InstanceRelation::new(original_id)];
		let tri_normal = pp::set_normals_and_coplanar(
			&mut tri_rel,
			&instance_rel,
			&halfedge,
			&vert_pos,
			precision.tolerance,
		);

		MeshBool {
			original_id: Some(original_id),
			precision,
			vert_pos: Rc::new(vert_pos),
			properties: Rc::new(properties),
			tri: Triangles {
				halfedge: Rc::new(halfedge),
				normal: Rc::new(tri_normal),
				relation: Rc::new(tri_rel),
			},
			instance_relation: Rc::new(instance_rel),
			collider,
		}
	}
}
