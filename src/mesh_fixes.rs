use crate::shared::Halfedges;
use nalgebra::{Matrix3, Vector3};
use std::mem;

#[inline]
fn flip_halfedge(halfedge: i32) -> i32 {
	let tri = halfedge / 3;
	let vert = 2 - (halfedge - 3 * tri);
	3 * tri + vert
}

pub fn transform_normal(transform: Matrix3<f64>, mut normal: Vector3<f64>) -> Vector3<f64> {
	normal = (transform * normal).normalize();
	if normal.x.is_nan() {
		return Vector3::zeros();
	}

	normal
}

pub struct FlipTris<'a> {
	pub halfedge: &'a mut Halfedges,
}

impl<'a> FlipTris<'a> {
	pub fn call(&mut self, tri: i32) {
		let mut face = [
			self.halfedge.get(3 * tri + 2),
			self.halfedge.get(3 * tri + 1),
			self.halfedge.get(3 * tri),
		];
		for i in 0..3 {
			mem::swap(&mut face[i].start_vert, &mut face[i].end_vert);
			face[i].paired_halfedge = flip_halfedge(face[i].paired_halfedge);
		}
		for i in 0..3 {
			self.halfedge
				.set_start(3 * tri + i, face[i as usize].start_vert);
			self.halfedge
				.set_pair(3 * tri + i, face[i as usize].paired_halfedge);
			self.halfedge
				.set_prop(3 * tri + i, face[i as usize].prop_vert);
		}
	}
}
