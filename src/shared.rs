use crate::common::AABB;
use crate::utils::{K_PRECISION, mat3, next3_usize};
use crate::vec::{vec_resize, vec_resize_nofill, vec_uninit};
use core::f64;
use nalgebra::{Matrix2x3, Matrix3, Matrix3x4, Point3, Vector2, Vector3, Vector4};
use std::ops::MulAssign;

#[inline]
pub fn safe_normalize(mut v: Vector3<f64>) -> Vector3<f64> {
	v = v.normalize();
	if v.x.is_finite() {
		v
	} else {
		Vector3::repeat(0.0)
	}
}

#[inline]
pub fn max_epsilon(min_epsilon: f64, bbox: &AABB) -> f64 {
	let epsilon = min_epsilon.max(K_PRECISION * bbox.scale());
	if epsilon.is_finite() { epsilon } else { -1.0 }
}

#[inline]
pub fn next_halfedge(current: i32) -> i32 {
	current + (if current % 3 == 2 { -2 } else { 1 })
}

pub fn normal_transform(transform: &Matrix3x4<f64>) -> Matrix3<f64> {
	mat3(transform)
		.transpose()
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
}

pub fn inverse_normal_transform(transform: &Matrix3x4<f64>) -> Matrix3<f64> {
	mat3(transform)
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
		.transpose()
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
}

///Symbolic perturbation primitives shared by Boolean3 and Boolean2.
///Carefully designed to minimize FP rounding error and eliminate it at edge
///cases.
#[inline]
pub fn with_sign(pos: bool, v: f64) -> f64 {
	if pos { v } else { -v }
}

///Interpolate the (y, z) of segment aL-aR at the given x. The choice of
///(x - aL) vs (x - aR) is the smaller in magnitude, which keeps FP error
///low near either endpoint. Domain check via DEBUG_ASSERT.
#[inline]
pub fn interpolate(a_l: Point3<f64>, a_r: Point3<f64>, x: f64) -> Vector2<f64> {
	let dx_l = x - a_l.x;
	let dx_r = x - a_r.x;
	debug_assert!(dx_l * dx_r <= 0.0, "Boolean manifold error: not in domain");

	let use_l = dx_l.abs() < dx_r.abs();
	let d_lr = a_r - a_l;
	let lambda = (if use_l { dx_l } else { dx_r }) / d_lr.x;
	if !lambda.is_finite() || !d_lr.y.is_finite() || !d_lr.z.is_finite() {
		return Vector2::new(a_l.y, a_l.z);
	}

	let mut yz = Vector2::default();
	yz[0] = lambda * d_lr.y + (if use_l { a_l.y } else { a_r.y });
	yz[1] = lambda * d_lr.z + (if use_l { a_l.z } else { a_r.z });
	return yz;
}

///`p < q` with symbolic perturbation: when `p == q` exactly, `dir < 0`
///acts as the tiebreaker. Used to give consistent strict-ordering answers
///regardless of which side of an FP equality we land on.
#[inline]
pub fn shadows(p: f64, q: f64, dir: f64) -> bool {
	if p == q { dir < 0.0 } else { p < q }
}

///By using the closest axis-aligned projection to the normal instead of a
///projection along the normal, we avoid introducing any rounding error.
#[inline]
pub fn get_axis_aligned_projection(normal: Vector3<f64>) -> Matrix2x3<f64> {
	let abs_normal = normal.abs();
	let (xyz_max, mut projection) = if abs_normal.z > abs_normal.x && abs_normal.z > abs_normal.y {
		(normal.z, Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
	} else if abs_normal.y > abs_normal.x {
		(normal.y, Matrix2x3::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
	} else {
		(normal.x, Matrix2x3::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
	};

	if xyz_max < 0.0 {
		projection.row_mut(0).mul_assign(-1.0);
	}
	projection
}

#[inline]
pub fn get_barycentric(v: &Point3<f64>, tri_pos: &Matrix3<f64>, tolerance: f64) -> Vector3<f64> {
	let edges = Matrix3::from_columns(&[
		tri_pos.column(2) - tri_pos.column(1),
		tri_pos.column(0) - tri_pos.column(2),
		tri_pos.column(1) - tri_pos.column(0),
	]);

	let d2 = Vector3::new(
		edges.column(0).magnitude_squared(),
		edges.column(1).magnitude_squared(),
		edges.column(2).magnitude_squared(),
	);

	let long_side = if d2[0] > d2[1] && d2[0] > d2[2] {
		0
	} else if d2[1] > d2[2] {
		1
	} else {
		2
	};

	let cross_p = edges.column(0).cross(&edges.column(1));
	let area2 = cross_p.magnitude_squared();
	let tol2 = tolerance.powi(2);

	let mut uvw = Vector3::default();
	for i in 0..3 {
		let dv = v - tri_pos.column(i);
		if dv.coords.magnitude_squared() < tol2 {
			// Return exactly equal if within tolerance of vert.
			uvw[i] = 1.0;
			return uvw;
		}
	}

	if d2[long_side] < tol2
	//point
	{
		return Vector3::new(1.0, 0.0, 0.0);
	} else if area2 > d2[long_side] * tol2
	//triangle
	{
		for i in 0..3 {
			let j = next3_usize(i);
			let cross_pv = edges.column(i).cross(&(v.coords - tri_pos.column(j)));
			let area_2v = cross_pv.magnitude_squared();
			// Return exactly equal if within tolerance of edge.
			uvw[i] = if area_2v < d2[i] * tol2 {
				0.0
			} else {
				cross_pv.dot(&cross_p)
			};
		}

		uvw /= uvw[0] + uvw[1] + uvw[2];
		return uvw;
	} else
	//line
	{
		let next_v = next3_usize(long_side);
		let alpha = (v - tri_pos.column(next_v))
			.coords
			.dot(&edges.column(long_side))
			/ d2[long_side];
		uvw[long_side] = 0.0;
		uvw[next_v] = 1.0 - alpha;
		let last_v = next3_usize(next_v);
		uvw[last_v] = alpha;
		return uvw;
	}
}

///Temporary or value-style halfedge record. Persistent Manifold storage uses
///Halfedges below, which derives endVert from the next halfedge in each face.
#[derive(Default, Clone, Copy, Debug)]
pub struct Halfedge {
	pub start_vert: i32,
	pub end_vert: i32,
	pub paired_halfedge: i32,
	pub prop_vert: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Halfedges {
	start: Vec<i32>,
	paired: Vec<i32>,
	prop_vert: Vec<i32>,
}

impl Halfedges {
	pub fn len(&self) -> usize {
		self.start.len()
	}

	pub fn is_empty(&self) -> bool {
		self.start.is_empty()
	}

	pub fn start(&self, idx: i32) -> i32 {
		self.start[idx as usize]
	}

	pub fn end(&self, idx: i32) -> i32 {
		self.start[next_halfedge(idx) as usize]
	}

	pub fn pair(&self, idx: i32) -> i32 {
		self.paired[idx as usize]
	}

	pub fn prop(&self, idx: i32) -> i32 {
		self.prop_vert[idx as usize]
	}

	pub fn set_start(&mut self, idx: i32, vert: i32) {
		self.start[idx as usize] = vert;
	}

	pub fn set_end(&mut self, idx: i32, vert: i32) {
		self.start[next_halfedge(idx) as usize] = vert;
	}

	pub fn set_pair(&mut self, idx: i32, pair: i32) {
		self.paired[idx as usize] = pair;
	}

	pub fn set_prop(&mut self, idx: i32, prop: i32) {
		self.prop_vert[idx as usize] = prop;
	}

	pub fn is_forward(&self, idx: i32) -> bool {
		self.start(idx) < self.end(idx)
	}

	pub fn get(&self, idx: i32) -> Halfedge {
		Halfedge {
			start_vert: self.start(idx),
			end_vert: self.end(idx),
			paired_halfedge: self.pair(idx),
			prop_vert: self.prop(idx),
		}
	}

	pub fn set(&mut self, idx: i32, start_vert: i32, paired_halfedge: i32, prop_vert: i32) {
		self.set_start(idx, start_vert);
		self.set_pair(idx, paired_halfedge);
		self.set_prop(idx, prop_vert);
	}

	pub fn push(&mut self, start_vert: i32, paired_halfedge: i32, prop_vert: i32) {
		self.start.push(start_vert);
		self.paired.push(paired_halfedge);
		self.prop_vert.push(prop_vert);
	}

	pub fn resize(&mut self, new_size: usize) {
		vec_resize(&mut self.start, new_size, -1);
		vec_resize(&mut self.paired, new_size, -1);
		vec_resize(&mut self.prop_vert, new_size, -1);
	}

	pub unsafe fn resize_nofill(&mut self, new_size: usize) {
		unsafe {
			vec_resize_nofill(&mut self.start, new_size);
			vec_resize_nofill(&mut self.paired, new_size);
			vec_resize_nofill(&mut self.prop_vert, new_size);
		}
	}

	pub fn to_data(&self) -> Vec<Halfedge> {
		let mut data = unsafe { vec_uninit(self.len()) };
		for idx in 0..self.len() {
			data[idx] = self.get(idx as i32);
		}
		data
	}
}

impl Halfedge {
	pub fn is_forward(&self) -> bool {
		self.start_vert < self.end_vert
	}
}

#[derive(Default, Clone)]
pub struct Barycentric {
	pub tri: i32,
	pub uvw: Vector4<f64>,
}

#[derive(Default, Copy, Clone, Debug)]
pub struct TriRef {
	/// The unique ID of the mesh instance of this triangle. If .meshID and .tri
	/// match for two triangles, then they are coplanar and came from the same
	/// face.
	pub mesh_id: i32,
	/// The OriginalID of the mesh this triangle came from. This ID is ideal for
	/// reapplying properties like UV coordinates to the output mesh.
	pub original_id: i32,
	/// If set as an input of MeshGL, it is passed along unchanged. This is how
	/// the user can tell us not to collapse certain edges: those that divide
	/// difference faceIDs. If not set, this is always -1.
	pub face_id: i32,
	/// Triangles with the same coplanar ID are coplanar. Starts as a canonical
	/// triangle index, but after boolean operations it may refer to a triangle
	/// that is no longer present in this mesh.
	pub coplanar_id: i32,
}

impl TriRef {
	pub fn same_face(&self, other: &TriRef) -> bool {
		self.mesh_id == other.mesh_id
			&& self.coplanar_id == other.coplanar_id
			&& self.face_id == other.face_id
	}
}

///This is a temporary edge structure which only stores edges forward and
///references the halfedge it was created from.
#[derive(Default, Clone)]
pub struct TmpEdge {
	pub first: i32,
	pub second: i32,
	pub halfedge_idx: i32,
}

impl TmpEdge {
	fn new(start: i32, end: i32, idx: i32) -> Self {
		Self {
			first: start.min(end),
			second: start.max(end),
			halfedge_idx: idx,
		}
	}
}

#[inline]
pub fn create_tmp_edges(halfedge: &Halfedges) -> Vec<TmpEdge> {
	let edges: Vec<TmpEdge>;
	edges = (0..halfedge.len())
		.into_iter()
		.map(|idx| {
			let idx = idx as i32;
			TmpEdge::new(
				halfedge.start(idx),
				halfedge.end(idx),
				if halfedge.is_forward(idx) { idx } else { -1 },
			)
		})
		.collect();

	let edges: Vec<TmpEdge> = edges
		.into_iter()
		.filter(|edge| !(edge.halfedge_idx < 0))
		.collect();
	debug_assert_eq!(edges.len(), halfedge.len() / 2, "Not oriented!");
	return edges;
}
