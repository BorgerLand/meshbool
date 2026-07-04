use nalgebra::{Matrix2x3, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector2, Vector3};
use std::f64;
use std::ops::{AddAssign, MulAssign};

pub const K_PRECISION: f64 = 1e-12;

macro_rules! next3 {
	($i:expr) => {
		match $i {
			0 => 1,
			1 => 2,
			2 => 0,
			_ => panic!("Invalid triangle index"),
		}
	};
}

macro_rules! prev3 {
	($i:expr) => {
		match $i {
			0 => 2,
			1 => 0,
			2 => 1,
			_ => panic!("Invalid triangle index"),
		}
	};
}

#[inline(always)]
pub const fn next3_i32(i: i32) -> i32 {
	next3!(i)
}

#[inline(always)]
pub const fn next3_usize(i: usize) -> usize {
	next3!(i)
}

#[inline(always)]
pub const fn prev3_i32(i: i32) -> i32 {
	prev3!(i)
}

#[inline(always)]
pub fn safe_normalize2(mut v: Vector2<f64>) -> Vector2<f64> {
	v = v.normalize();
	if v.x.is_finite() {
		v
	} else {
		Vector2::repeat(0.0)
	}
}

#[inline(always)]
pub fn safe_normalize3(mut v: Vector3<f64>) -> Vector3<f64> {
	v = v.normalize();
	if v.x.is_finite() {
		v
	} else {
		Vector3::repeat(0.0)
	}
}

pub fn normal_transform(transform: Matrix3x4<f64>) -> Matrix3<f64> {
	mat3(transform)
		.transpose()
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
}

pub fn inverse_normal_transform(transform: Matrix3x4<f64>) -> Matrix3<f64> {
	mat3(transform)
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
		.transpose()
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
}

pub fn transform_normal(transform: Matrix3<f64>, mut normal: Vector3<f64>) -> Vector3<f64> {
	normal = (transform * normal).normalize();
	if normal.x.is_nan() {
		return Vector3::zeros();
	}

	normal
}

#[inline(always)]
pub fn mat4(a: Matrix3x4<f64>) -> Matrix4<f64> {
	let mut result = Matrix4::identity();
	result.fixed_view_mut::<3, 4>(0, 0).copy_from(&a);
	result
}

#[inline(always)]
pub fn mat3(a: Matrix3x4<f64>) -> Matrix3<f64> {
	a.fixed_columns::<3>(0).into_owned()
}

///By using the closest axis-aligned projection to the normal instead of a
///projection along the normal, we avoid introducing any rounding error.
#[inline(always)]
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

#[inline(always)]
pub fn get_barycentric(v: Point3<f64>, tri_pos: Matrix3<f64>, tolerance: f64) -> Vector3<f64> {
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
	let tol2 = tolerance * tolerance;

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

///Determines if the three points are wound counter-clockwise, clockwise, or
///colinear within the specified tolerance.
///
///@param p0 First point
///@param p1 Second point
///@param p2 Third point
///@param tol Tolerance value for colinearity
///@return int, like Signum, this returns 1 for CCW, -1 for CW, and 0 if within
///tol of colinear.
#[inline(always)]
pub fn ccw(p0: Point2<f64>, p1: Point2<f64>, p2: Point2<f64>, tol: f64) -> i32 {
	let v1 = p1 - p0;
	let v2 = p2 - p0;
	let area = v1.x * v2.y - v1.y * v2.x;
	let base2 = v1.magnitude_squared().max(v2.magnitude_squared());
	if area * area * 4.0 <= base2 * tol * tol {
		0
	} else if area > 0.0 {
		1
	} else {
		-1
	}
}

///Sine function where multiples of 90 degrees come out exact.
///
///@param x Angle in degrees.
#[inline(always)]
pub fn sind(mut x: f64) -> f64 {
	if !x.is_finite() {
		return f64::NAN;
	}
	if x < 0.0 {
		return -sind(-x);
	}
	let quo: i32;
	(x, quo) = libm::remquo(x.abs(), 90.0);
	let xr = x.to_radians();
	match quo % 4 {
		0 => libm::sin(xr),
		1 => libm::cos(xr),
		2 => -libm::sin(xr),
		3 => -libm::cos(xr),
		_ => 0.0,
	}
}

///Cosine function where multiples of 90 degrees come out exact.
///
///@param x Angle in degrees.
#[inline(always)]
pub fn cosd(x: f64) -> f64 {
	sind(x + 90.0)
}

//yeah it's not atomic sue me
pub fn atomic_add<T>(target: &mut T, add: T) -> T
where
	T: Copy + AddAssign,
{
	let old = *target;
	*target += add;
	old
}

pub fn is_axis_aligned(transform: Matrix3x4<f64>) -> bool {
	for row in 0..3 {
		let mut count = 0;
		for col in 0..3 {
			if transform[(row, col)] == 0.0 {
				count += 1;
			}
		}

		if count != 2 {
			return false;
		}
	}

	true
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
	a * (1.0 - t) + b * t
}
