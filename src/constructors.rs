use crate::common::{Polygons, Quality, SimplePolygon, cosd, sind};
use crate::meshboolimpl::{MeshBoolImpl, Shape};
use crate::polygon::{PolyVert, PolygonsIdx, SimplePolygonIdx, triangulate_idx};
use crate::{MeshBool, triangulate};
use nalgebra::{Matrix2, Matrix3x4, Point2, Point3, Vector2, Vector3};

impl MeshBool {
	///Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
	///and the rest at similarly symmetric points.
	pub fn tetrahedron() -> Self {
		return Self::from(MeshBoolImpl::from_shape(
			Shape::Tetrahedron,
			Matrix3x4::identity(),
		));
	}

	///Constructs a unit cube (edge lengths all one), by default in the first
	///octant, touching the origin. If any dimensions in size are negative, or if
	///all are zero, an empty Manifold will be returned.
	///
	///@param size The X, Y, and Z dimensions of the box.
	///@param center Set to true to shift the center to the origin.
	pub fn cube(size: Vector3<f64>, center: bool) -> Self {
		if size.x < 0.0 || size.y < 0.0 || size.z < 0.0 || size.magnitude_squared() == 0.0 {
			return Self::invalid();
		}

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

		Self::from(MeshBoolImpl::from_shape(Shape::Cube, m))
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
		circular_segments: u32,
		center: bool,
	) -> Self {
		if height <= 0.0 || radius_low <= 0.0 {
			return Self::invalid();
		}

		let scale = if radius_high >= 0.0 {
			radius_high / radius_low
		} else {
			1.0
		};
		let radius = radius_low.max(radius_high);
		let n = if circular_segments > 2 {
			circular_segments
		} else {
			Quality::get_circular_segments(radius)
		};

		let mut circle: SimplePolygon = vec![Point2::default(); n as usize];
		let d_phi = 360.0 / (n as f64);
		for i in 0..n {
			circle[i as usize] = Point2::<f64>::new(
				radius_low * cosd(d_phi * i as f64),
				radius_low * sind(d_phi * i as f64),
			);
		}

		let cylinder = Self::extrude(&vec![circle], height, 0, 0.0, Vector2::new(scale, scale));

		if center {
			cylinder
				.translate(Vector3::new(0.0, 0.0, -height / 2.0))
				.as_original()
		} else {
			cylinder
		}
	}

	///Constructs a geodesic sphere of a given radius.
	///
	///@param radius Radius of the sphere. Must be positive.
	///@param circularSegments Number of segments along its
	///diameter. This number will always be rounded up to the nearest factor of
	///four, as this sphere is constructed by refining an octahedron. This means
	///there are a circle of vertices on all three of the axis planes. Default is
	///calculated by the static Defaults.
	pub fn sphere(radius: f64, circular_segments: i32) -> Self {
		if radius <= 0.0 {
			return Self::invalid();
		}
		let n: i32 = if circular_segments > 0 {
			(circular_segments + 3) / 4
		} else {
			(Quality::get_circular_segments(radius) / 4) as i32
		};
		let mut meshbool_impl = MeshBoolImpl::from_shape(Shape::Octahedron, Matrix3x4::identity());
		meshbool_impl.subdivide(|_, _, _| n - 1, false);
		meshbool_impl.vert_pos.iter_mut().for_each(|v| {
			*v = Vector3::from(core::f64::consts::FRAC_PI_2 * (Vector3::repeat(1.0) - v.coords))
				.into();
			v.iter_mut().for_each(|i| *i = i.cos());
			*v = (radius * Vector3::from(v.coords.normalize())).into();
			if v.x.is_nan() {
				*v = Vector3::repeat(0.0).into();
			}
		});
		meshbool_impl.finish();
		// Ignore preceding octahedron.
		meshbool_impl.initialize_original(false);
		return Self::from(meshbool_impl);
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
	) -> Self {
		if cross_section.len() == 0 || height <= 0.0 {
			return Self::invalid();
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

		let mut meshbool_impl = MeshBoolImpl {
			vert_pos,
			..Default::default()
		};

		meshbool_impl.create_halfedges(tri_verts, Vec::new());
		meshbool_impl.finish();
		meshbool_impl.initialize_original(false);
		meshbool_impl.mark_coplanar();
		Self::from(meshbool_impl)
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
		circular_segments: i32,
		mut revolve_degrees: f64,
	) -> Self {
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
			return Self::invalid();
		}

		if revolve_degrees > 360.0 {
			revolve_degrees = 360.0;
		}
		let is_full_revolution = revolve_degrees == 360.0;

		let n_divisions: i32 = if circular_segments > 2 {
			circular_segments
		} else {
			(Quality::get_circular_segments(radius) as f64 * revolve_degrees / 360.0) as i32
		};

		let mut meshbool_impl = MeshBoolImpl::default();
		let vert_pos = &mut meshbool_impl.vert_pos;
		let mut tri_verts_dh: Vec<Vector3<i32>> = vec![];
		let tri_verts = &mut tri_verts_dh;

		let mut start_poses: Vec<i32> = vec![];
		let mut end_poses: Vec<i32> = vec![];

		let d_phi: f64 = revolve_degrees / n_divisions as f64;
		// first and last slice are distinguished if not a full revolution.
		let n_slices: i32 = if is_full_revolution {
			n_divisions
		} else {
			n_divisions + 1
		};

		for poly in polygons.iter() {
			let mut n_pos_verts: usize = 0;
			let mut n_revolve_axis_verts: usize = 0;
			for pt in poly.iter() {
				if pt.x > 0.0 {
					n_pos_verts += 1;
				} else {
					n_revolve_axis_verts += 1;
				}
			}

			for poly_vert in 0..poly.len() {
				let start_pos_index: usize = vert_pos.len();

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

				let prev_start_pos_index: i32 = start_pos_index as i32
					+ (if poly_vert == 0 {
						n_revolve_axis_verts as i32 + (n_slices * n_pos_verts as i32)
					} else {
						0
					}) + (if prev_poly_vertex.x == 0.0 {
					-1
				} else {
					-n_slices
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
						let last_slice: i32 = (if slice == 0 { n_divisions } else { slice }) - 1;
						if curr_poly_vertex.x > 0.0 {
							tri_verts.push(Vector3::new(
								start_pos_index as i32 + slice,
								start_pos_index as i32 + last_slice,
								// "Reuse" vertex of first slice if it lies on the revolve axis
								if prev_poly_vertex.x == 0.0 {
									prev_start_pos_index
								} else {
									prev_start_pos_index + last_slice
								},
							));
						}

						if prev_poly_vertex.x > 0.0 {
							tri_verts.push(Vector3::new(
								prev_start_pos_index + last_slice,
								prev_start_pos_index + slice,
								if curr_poly_vertex.x == 0.0 {
									start_pos_index as i32
								} else {
									start_pos_index as i32 + slice
								},
							));
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
			let front_triangles: Vec<Vector3<i32>> =
				triangulate(&polygons, meshbool_impl.epsilon, true);
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

		meshbool_impl.create_halfedges(tri_verts_dh, vec![]);
		meshbool_impl.finish();
		meshbool_impl.initialize_original(false);
		meshbool_impl.mark_coplanar();
		return Self::from(meshbool_impl);
	}
}
