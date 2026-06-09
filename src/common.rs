use crate::mesh::MeshGLP;
use core::f64;
use nalgebra::{Matrix3x4, Point2, Point3, Vector2, Vector3};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::ops::{Deref, DerefMut, Div};

#[cfg(feature = "test")]
use std::sync::atomic::AtomicBool;

pub type MeshGL32 = MeshGLP<f32, u32>;
pub type MeshGL64 = MeshGLP<f64, u64>;

pub type SimplePolygon = Vec<Point2<f64>>;
pub type Polygons = Vec<SimplePolygon>;

//struct was originally named "Box", causing name conflict with the built in rust type
///Axis-aligned 3D box, primarily for bounding.
#[derive(Clone, Copy, Debug)]
pub struct AABB {
	pub min: Point3<f64>,
	pub max: Point3<f64>,
}

impl Default for AABB {
	///Default constructor is an infinite box that contains all space.
	fn default() -> Self {
		Self {
			min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
			max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
		}
	}
}

impl AABB {
	///Creates a box that contains the two given points.
	pub fn new(p1: Point3<f64>, p2: Point3<f64>) -> Self {
		Self {
			min: p1.inf(&p2),
			max: p1.sup(&p2),
		}
	}

	///Returns the dimensions of the Box.
	pub fn size(&self) -> Vector3<f64> {
		self.max - self.min
	}

	///Returns the center point of the Box.
	pub fn center(&self) -> Vector3<f64> {
		0.5 * (self.max.coords + self.min.coords)
	}

	///Returns the absolute-largest coordinate value of any contained
	///point.
	pub fn scale(&self) -> f64 {
		self.min.coords.abs().sup(&self.max.coords.abs()).max()
	}

	///Expand this box to include the given point.
	pub fn union_point(&mut self, p: Point3<f64>) {
		self.min = self.min.inf(&p);
		self.max = self.max.sup(&p);
	}

	///Expand this box to include the given box.
	pub fn union_aabb(&self, other: &Self) -> Self {
		Self {
			min: self.min.inf(&other.min),
			max: self.max.sup(&other.max),
		}
	}

	///Transform the given box by the given axis-aligned affine transform.
	///
	///Ensure the transform passed in is axis-aligned (rotations are all
	///multiples of 90 degrees), or else the resulting bounding box will no longer
	///bound properly.
	pub fn transform(&self, transform: Matrix3x4<f64>) -> Self {
		let mut out = Self::default();
		let min_t = Point3::from(transform * self.min.coords.push(1.0));
		let max_t = Point3::from(transform * self.max.coords.push(1.0));
		out.min = min_t.inf(&max_t);
		out.max = min_t.sup(&max_t);
		out
	}

	///Does this box have finite bounds?
	pub fn is_finite(&self) -> bool {
		self.min.iter().all(|x| x.is_finite()) && self.max.iter().all(|x| x.is_finite())
	}
}

pub trait AABBOverlap<T> {
	fn does_overlap(&self, other: &T) -> bool;
}

impl AABBOverlap<AABB> for AABB {
	///Does this box overlap the one given (including equality)?
	fn does_overlap(&self, other: &AABB) -> bool {
		self.min.x <= other.max.x
			&& self.min.y <= other.max.y
			&& self.min.z <= other.max.z
			&& self.max.x >= other.min.x
			&& self.max.y >= other.min.y
			&& self.max.z >= other.min.z
	}
}

impl AABBOverlap<Point3<f64>> for AABB {
	///Does the given point project within the XY extent of this box
	///(including equality)?
	fn does_overlap(&self, p: &Point3<f64>) -> bool {
		// projected in z
		p.x <= self.max.x && p.x >= self.min.x && p.y <= self.max.y && p.y >= self.min.y
	}
}

#[derive(Clone, Copy, Debug)]
pub struct Rect {
	pub min: Point2<f64>,
	pub max: Point2<f64>,
}

impl Default for Rect {
	///Default constructor is an infinite box that contains all space.
	fn default() -> Self {
		Self {
			min: Point2::new(f64::INFINITY, f64::INFINITY),
			max: Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
		}
	}
}

impl Rect {
	pub fn new(a: Point2<f64>, b: Point2<f64>) -> Rect {
		Rect {
			min: a.inf(&b),
			max: a.sup(&b),
		}
	}

	///Return the dimensions of the rectangle.
	pub fn size(&self) -> Vector2<f64> {
		self.max - self.min
	}

	///Returns the absolute-largest coordinate value of any contained
	///point.
	pub fn scale(&self) -> f64 {
		self.min.coords.abs().sup(&self.max.coords.abs()).max()
	}

	///Does this rectangle contain (includes on border) the given point?
	pub fn contains(&self, p: &Point2<f64>) -> bool {
		p.x >= self.min.x && p.y >= self.min.y && p.x <= self.max.x && p.y <= self.max.y
	}

	///Does this rectangle overlap the one given (including equality)?
	pub fn does_overlap(&self, rect: &Rect) -> bool {
		self.min.x <= rect.max.x
			&& self.min.y <= rect.max.y
			&& self.max.x >= rect.min.x
			&& self.max.y >= rect.min.y
	}

	///Expand this rectangle (in place) to include the given point.
	pub fn union(&mut self, p: Point2<f64>) {
		self.min = self.min.inf(&p);
		self.max = self.max.sup(&p);
	}
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OpType {
	Add,
	Subtract,
	Intersect,
}

pub struct Quality;
impl Quality {
	const DEFAULT_SEGMENTS: u32 = 0;
	const DEFAULT_ANGLE: f64 = 10.0;
	const DEFAULT_LENGTH: f64 = 1.0;

	const CIRCULAR_SEGMENTS: u32 = Self::DEFAULT_SEGMENTS;
	const CIRCULAR_ANGLE: f64 = Self::DEFAULT_ANGLE;
	const CIRCULAR_EDGE_LENGTH: f64 = Self::DEFAULT_LENGTH;

	///Determine the result of the SetMinCircularAngle(),
	///SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
	///
	///@param radius For a given radius of circle, determine how many default
	///segments there will be.
	pub fn get_circular_segments(radius: f64) -> u32 {
		if Self::CIRCULAR_SEGMENTS > 0 {
			return Self::CIRCULAR_SEGMENTS;
		}
		let n_seg_a = (360.0 / Self::CIRCULAR_ANGLE) as u32;
		let n_seg_l = (2.0 * radius * f64::consts::PI / Self::CIRCULAR_EDGE_LENGTH) as u32;
		let mut n_seg = n_seg_a.min(n_seg_l) + 3;
		n_seg -= n_seg % 4;
		n_seg.max(4)
	}
}

//disgusting cursed reimplementation of c++ implicit number type coercion
pub trait LossyFrom<T: Copy>: Div<Output = Self> + Sized {
	fn lossy_from(other: T) -> Self;
}

//impl lossyfrom instead!
pub trait LossyInto<T: Copy> {
	fn lossy_into(self) -> T;
}

impl<T, U> LossyInto<U> for T
where
	T: Copy,
	U: Copy + LossyFrom<T>,
{
	fn lossy_into(self) -> U {
		U::lossy_from(self)
	}
}

//lossy_from!([from, from, from], to)
macro_rules! lossy_from {
	([ $( $f:ty ),* ], $t:ty) => {
		$(
			impl LossyFrom<$f> for $t {
				fn lossy_from(other: $f) -> Self {
					other as Self
				}
			}
		)*
	};
}

lossy_from!([i32, u32, u64, usize], usize);
lossy_from!([usize], u64);
lossy_from!([usize], u32);
lossy_from!([f64], f64);
lossy_from!([f64], f32);

pub struct OrderedF64(pub f64);

impl Ord for OrderedF64 {
	fn cmp(&self, other: &Self) -> Ordering {
		self.0.total_cmp(&other.0)
	}
}

impl PartialOrd for OrderedF64 {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl Eq for OrderedF64 {}

impl PartialEq for OrderedF64 {
	fn eq(&self, other: &Self) -> bool {
		self.0.total_cmp(&other.0) == Ordering::Equal
	}
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
	a * (1.0 - t) + b * t
}

///Sine function where multiples of 90 degrees come out exact.
///
///@param x Angle in degrees.
#[inline]
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
#[inline]
pub fn cosd(x: f64) -> f64 {
	sind(x + 90.0)
}

/// Perform extra sanity checks and assertions on the intermediate data
/// structures.
#[cfg(feature = "test")]
pub static INTERMEDIATE_CHECKS: AtomicBool = AtomicBool::new(false);
/// Perform 3D mesh self-intersection test on intermediate boolean results to
/// test for ϵ-validity. For debug purposes only.
#[cfg(feature = "test")]
pub static SELF_INTERSECTION_CHECKS: AtomicBool = AtomicBool::new(false);
/// If processOverlaps is false, a geometric check will be performed to assert
/// all triangles are CCW.
#[cfg(feature = "test")]
pub static PROCESS_OVERLAPS: AtomicBool = AtomicBool::new(true);

//wrappers around hashmap/hashset to remove the nondeterministic hasher
#[derive(Clone)]
pub struct DeterministicMap<K, V>(HashMap<K, V, BuildHasherDefault<DefaultHasher>>);

impl<K, V> Deref for DeterministicMap<K, V> {
	type Target = HashMap<K, V, BuildHasherDefault<DefaultHasher>>;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<K, V> DerefMut for DeterministicMap<K, V> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

impl<K, V> Default for DeterministicMap<K, V> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K, V> DeterministicMap<K, V> {
	pub fn new() -> Self {
		Self(HashMap::with_hasher(BuildHasherDefault::default()))
	}

	pub fn with_capacity(capacity: usize) -> Self {
		Self(HashMap::with_capacity_and_hasher(
			capacity,
			BuildHasherDefault::default(),
		))
	}
}

#[derive(Clone)]
pub struct DeterministicSet<K>(HashSet<K, BuildHasherDefault<DefaultHasher>>);

impl<K> Deref for DeterministicSet<K> {
	type Target = HashSet<K, BuildHasherDefault<DefaultHasher>>;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<K> DerefMut for DeterministicSet<K> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

impl<K> Default for DeterministicSet<K> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K> DeterministicSet<K> {
	pub fn new() -> Self {
		Self(HashSet::with_hasher(BuildHasherDefault::default()))
	}

	pub fn into_iter(self) -> impl Iterator<Item = K> {
		self.0.into_iter()
	}
}
