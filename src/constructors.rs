use crate::MeshBool;
use crate::common::{Polygons, Quality, SimplePolygon, cosd, sind};
use crate::disjoint_sets::DisjointSets;
use crate::meshboolimpl::{MeshBoolImpl, Shape};
use crate::parallel::{copy_if, gather};
use crate::polygon::{PolyVert, PolygonsIdx, SimplePolygonIdx, triangulate_idx};
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

	//
	// This operation returns a vector of Manifolds that are topologically
	// disconnected. If everything is connected, the vector is length one,
	// containing a copy of the original. It is the inverse operation of Compose().
	//
	pub fn decompose(&self) -> Vec<Self> {
		let uf = DisjointSets::new(self.num_vert() as u32);
		// Graph graph;
		let p_impl = &self.meshbool_impl;
		for halfedge in p_impl.halfedge.iter() {
			if halfedge.is_forward() {
				uf.unite(halfedge.start_vert as u32, halfedge.end_vert as u32);
			}
		}
		let mut component_indices: Vec<i32> = vec![];
		let num_components = uf.connected_components(&mut component_indices);

		if num_components == 1 {
			return vec![self.clone()];
		}
		let vert_label: Vec<i32> = component_indices;

		let num_vert = self.num_vert();
		let mut meshes: Vec<Self> = vec![];
		for i in 0..num_components {
			let mut meshbool_impl = MeshBoolImpl::default();
			// inherit original object's precision
			meshbool_impl.epsilon = p_impl.epsilon;
			meshbool_impl.tolerance = p_impl.tolerance;

			let mut vert_new2old: Vec<i32> = vec![0; num_vert];
			let n_vert = copy_if(0..num_vert as i32, &mut vert_new2old, |v| {
				vert_label[v as usize] == i as i32
			});
			meshbool_impl.vert_pos.resize(n_vert, Default::default());
			vert_new2old.resize(n_vert, Default::default());
			gather(&vert_new2old, &p_impl.vert_pos, &mut meshbool_impl.vert_pos);

			let mut face_new2old: Vec<i32> = vec![0; self.num_tri()];
			let halfedge = &p_impl.halfedge;
			let n_face = copy_if(0..self.num_tri() as i32, &mut face_new2old, |face| {
				vert_label[halfedge[3 * face as usize].start_vert as usize] == i as i32
			});

			if n_face == 0 {
				continue;
			}
			face_new2old.resize(n_face, Default::default());

			meshbool_impl.gather_faces_with_old(p_impl, &face_new2old);
			meshbool_impl.reindex_verts(&vert_new2old, p_impl.num_vert());
			meshbool_impl.finish();

			meshes.push(Self::from(meshbool_impl));
		}
		return meshes;
	}
}
