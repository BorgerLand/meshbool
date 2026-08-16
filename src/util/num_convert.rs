use std::cmp::Ordering;
use std::ops::Div;

//disgusting cursed trait-based reimplementation of c++ implicit number type coercion
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

			impl LossyFrom<&$f> for $t {
				fn lossy_from(other: &$f) -> Self {
					*other as Self
				}
			}
		)*
	};
}

lossy_from!([i32, u32, u64, usize], usize);
lossy_from!([u32, usize], u64);
lossy_from!([i32, u32, u64, usize], u32);
lossy_from!([f64], f32);
lossy_from!([u32, u64, usize], i32);
lossy_from!([i32, u64], u64);

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
