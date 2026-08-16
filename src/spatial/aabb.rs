use nalgebra::{Matrix3x4, Point2, Point3, Vector2, Vector3};

///Axis-aligned 3D box, primarily for bounding.
#[derive(Clone, Copy, Debug)]
pub struct Box3D {
	pub min: Point3<f64>,
	pub max: Point3<f64>,
}

impl Box3D {
	///Creates a box that contains the two given points.
	pub fn new(p1: Point3<f64>, p2: Point3<f64>) -> Self {
		Self {
			min: p1.inf(&p2),
			max: p1.sup(&p2),
		}
	}

	pub fn empty() -> Self {
		Self {
			min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
			max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
		}
	}

	pub fn infinite() -> Self {
		Self {
			min: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
			max: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
		}
	}

	pub fn from_cloud(p: &[Point3<f64>]) -> Self {
		Self {
			min: p.iter().fold(
				Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
				|a, &b| {
					if a.x.is_nan() {
						return b;
					}
					if b.x.is_nan() {
						return a;
					}
					a.inf(&b)
				},
			),

			max: p.iter().fold(
				Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
				|a, &b| {
					if a.x.is_nan() {
						return b;
					}
					if b.x.is_nan() {
						return a;
					}
					a.sup(&b)
				},
			),
		}
	}

	///Returns the dimensions of the Box.
	pub fn size(self) -> Vector3<f64> {
		self.max - self.min
	}

	///Returns the center point of the Box.
	pub fn center(self) -> Point3<f64> {
		(0.5 * (self.max.coords + self.min.coords)).into()
	}

	///Returns the absolute-largest coordinate value of any contained
	///point.
	pub fn scale(self) -> f64 {
		self.min.coords.abs().sup(&self.max.coords.abs()).max()
	}

	///Expand this box to include the given point.
	pub fn union_point_mut(&mut self, p: Point3<f64>) {
		self.min = self.min.inf(&p);
		self.max = self.max.sup(&p);
	}

	///Expand this box to include the given box.
	pub fn union_box3(self, other: Self) -> Self {
		Self {
			min: self.min.inf(&other.min),
			max: self.max.sup(&other.max),
		}
	}

	pub fn intersection_box3(self, other: Self) -> Self {
		Self {
			min: self.min.sup(&other.min),
			max: self.max.inf(&other.max),
		}
	}

	///Transform the given box by the given axis-aligned affine transform.
	///
	///Ensure the transform passed in is axis-aligned (rotations are all
	///multiples of 90 degrees), or else the resulting bounding box will no longer
	///bound properly.
	pub fn transform_axis_aligned(self, transform: Matrix3x4<f64>) -> Self {
		let min_t = Point3::from(transform * self.min.coords.push(1.0));
		let max_t = Point3::from(transform * self.max.coords.push(1.0));
		Self {
			min: min_t.inf(&max_t),
			max: min_t.sup(&max_t),
		}
	}

	///Transform the given box by the given affine transform using Arvo's method.
	///
	///https://dl.acm.org/doi/10.5555/90767.90922
	pub fn transform(self, transform: Matrix3x4<f64>) -> Self {
		let translate = transform.column(3).into_owned().into();
		let mut out = Self {
			min: translate,
			max: translate,
		};
		for j in 0..3 {
			let col = transform.column(j);
			let (a, b) = (col * self.min[j], col * self.max[j]);
			out.min += a.inf(&b);
			out.max += a.sup(&b);
		}

		out
	}

	///Does this box have finite bounds?
	pub fn is_finite(self) -> bool {
		self.min.iter().all(|v| v.is_finite()) && self.max.iter().all(|v| v.is_finite())
	}

	pub fn is_empty(self) -> bool {
		self.min.x >= self.max.x || self.min.y >= self.max.y || self.min.z >= self.max.z
	}
}

pub trait Overlap<T: Copy = Self> {
	fn overlaps(self, other: T) -> bool;
}

impl Overlap for Box3D {
	///Does this box overlap the one given (including equality)?
	fn overlaps(self, other: Box3D) -> bool {
		self.min.x <= other.max.x
			&& self.min.y <= other.max.y
			&& self.min.z <= other.max.z
			&& self.max.x >= other.min.x
			&& self.max.y >= other.min.y
			&& self.max.z >= other.min.z
	}
}

impl Overlap<Point3<f64>> for Box3D {
	///Does the given point project within the XY extent of this box
	///(including equality)?
	fn overlaps(self, p: Point3<f64>) -> bool {
		// projected in z
		p.x <= self.max.x && p.x >= self.min.x && p.y <= self.max.y && p.y >= self.min.y
	}
}

#[derive(Clone, Copy, Debug)]
pub struct Box2D {
	pub min: Point2<f64>,
	pub max: Point2<f64>,
}

impl Box2D {
	pub fn new(a: Point2<f64>, b: Point2<f64>) -> Box2D {
		Box2D {
			min: a.inf(&b),
			max: a.sup(&b),
		}
	}

	pub fn empty() -> Self {
		Self {
			min: Point2::new(f64::INFINITY, f64::INFINITY),
			max: Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
		}
	}

	///Return the dimensions of the rectangle.
	pub fn size(self) -> Vector2<f64> {
		self.max - self.min
	}

	///Returns the absolute-largest coordinate value of any contained
	///point.
	pub fn scale(self) -> f64 {
		self.min.coords.abs().sup(&self.max.coords.abs()).max()
	}

	///Does this rectangle contain (includes on border) the given point?
	pub fn contains(self, p: &Point2<f64>) -> bool {
		p.x >= self.min.x && p.y >= self.min.y && p.x <= self.max.x && p.y <= self.max.y
	}

	///Expand this rectangle (in place) to include the given point.
	pub fn union_point_mut(&mut self, p: Point2<f64>) {
		self.min = self.min.inf(&p);
		self.max = self.max.sup(&p);
	}
}

impl Overlap for Box2D {
	///Does this rectangle overlap the one given (including equality)?
	fn overlaps(self, other: Self) -> bool {
		self.min.x <= other.max.x
			&& self.min.y <= other.max.y
			&& self.max.x >= other.min.x
			&& self.max.y >= other.min.y
	}
}
