use crate::util::num_convert::{LossyFrom, LossyInto};
use crate::util::vec_ext;
use std::iter;
use std::ops::AddAssign;

//keywords to search the c++ codebase if ever porting parallelization (ignore case):
//manifold_par
//tbb
//atomic
//transform(

///safety: all elements are uninitialized
pub unsafe fn uninit<T>(size: usize) -> Vec<T> {
	let mut vec = Vec::with_capacity(size);
	unsafe {
		vec.set_len(size);
	}
	vec
}

//it is more beneficial to collect() this for parallelization,
//but single threaded can sometimes avoid that allocation
pub fn exclusive_scan_with_total<IO>(
	input: impl IntoIterator<Item = IO>,
	init: IO,
) -> impl Iterator<Item = IO>
where
	IO: Copy + AddAssign + Default,
{
	let mut acc = IO::default(); //boldly assuming this always returns 0
	iter::once(init).chain(input.into_iter()).map(move |input| {
		acc += input;
		acc
	})
}

///Compute the inclusive prefix sum for the range `[first, last)` using the
///binary operator `f`, with initial value `init` and
///identity element `identity`, and store the result in the range
///starting from `d_first`.
///
///This is different from `exclusive_scan` in the sequential algorithm by
///requiring an identity element. This is needed so that each block can be
///scanned in parallel and combined later.
///
///The input range `[first, last)` and
///the output range `[d_first, d_first + last - first)`
///must be equal or non-overlapping.
pub fn exclusive_scan_in_place<IO>(io: &mut [IO], init: IO)
where
	IO: Copy + AddAssign,
{
	let mut acc = init;
	for i in 0..io.len() {
		let old_val = io[i];
		io[i] = acc;
		acc += old_val;
	}
}

///`scatter` copies elements from a source range into an output array according
///to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
///is assigned to `outputFirst[mapFirst[i - first]]`.  If the same index appears
///more than once in the range `[mapFirst, mapFirst + (last - first))`, the
///result is undefined.
///
///The map range, input range and the output range must not overlap.
pub unsafe fn scatter<IO, Map>(
	map_new2old: impl IntoIterator<Item = Map>,
	out_len: usize,
) -> Vec<IO>
where
	IO: Copy + LossyFrom<usize>,
	Map: Copy + LossyInto<usize>,
{
	let mut output = unsafe { vec_ext::uninit(out_len) };
	for (i, mapped) in map_new2old.into_iter().enumerate() {
		output[mapped.lossy_into()] = i.lossy_into();
	}

	output
}

///`gather` copies elements from a source array into a destination range
///according to a map. For each input iterator `i`
///in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
///is assigned to `outputFirst[i - map_first]`.
///
///The map range, input range and the output range must not overlap.
pub fn gather<IO, Map>(input: &[IO], map_new2old: impl ExactSizeIterator<Item = Map>) -> Vec<IO>
where
	IO: Copy,
	Map: Copy + LossyInto<usize>,
{
	map_new2old
		.map(|mapped| input[mapped.lossy_into()])
		.collect()
}

//c++ std::partition
pub fn unstable_partition<T, F>(slice: &mut [T], mut predicate: F) -> usize
where
	F: FnMut(&T) -> bool,
{
	let mut left = 0;
	let mut right = slice.len();

	while left < right {
		if predicate(&slice[left]) {
			left += 1;
		} else {
			right -= 1;
			slice.swap(left, right);
		}
	}

	left
}
