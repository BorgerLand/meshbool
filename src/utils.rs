use crate::common::LossyInto;
use crate::parallel::gather;
use crate::vec::vec_uninit;
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Point2};
use std::mem;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

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

#[inline]
pub const fn next3_i32(i: i32) -> i32 {
	next3!(i)
}

#[inline]
pub const fn next3_usize(i: usize) -> usize {
	next3!(i)
}

#[inline]
pub const fn prev3_i32(i: i32) -> i32 {
	prev3!(i)
}

pub fn permute<IO, Map>(in_out: &mut Vec<IO>, new2old: &[Map])
where
	IO: Copy,
	Map: Copy + LossyInto<usize>,
{
	let mut tmp = unsafe { vec_uninit(new2old.len()) };
	mem::swap(&mut tmp, in_out);
	gather(new2old, &tmp, in_out);
}

pub unsafe fn atomic_add_i32(target: &mut i32, add: i32) -> i32 {
	let atomic_ref: &AtomicI32 = unsafe { std::mem::transmute(target) };
	atomic_ref.fetch_add(add, Ordering::SeqCst)
}

pub unsafe fn atomic_add_f64(target: &mut f64, add: f64) -> f64 {
	let atomic_ref: &AtomicU64 = unsafe { std::mem::transmute(target) };
	atomic_ref
		.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |old| {
			let old = f64::from_bits(old);
			Some((old + add).to_bits())
		})
		.map(|v| f64::from_bits(v))
		.unwrap()
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
#[inline]
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

#[inline]
pub fn mat4(a: &Matrix3x4<f64>) -> Matrix4<f64> {
	let mut result = Matrix4::identity();
	result.fixed_view_mut::<3, 4>(0, 0).copy_from(&a);
	result
}

#[inline]
pub fn mat3(a: &Matrix3x4<f64>) -> Matrix3<f64> {
	a.fixed_columns::<3>(0).into_owned()
}
