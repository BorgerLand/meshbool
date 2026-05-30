use crate::{common::LossyInto, vec::vec_uninit};
use std::ops::{Add, AddAssign};

///Compute the inclusive prefix sum for the range `[first, last)`
///using the summation operator, and store the result in the range
///starting from `d_first`.
///
///The input range `[first, last)` and
///the output range `[d_first, d_first + last - first)`
///must be equal or non-overlapping.
pub fn inclusive_scan<IO>(mut input: impl Iterator<Item = IO>, output: &mut [IO])
where
	IO: Default + Copy + Add<Output = IO>,
{
	if output.len() == 0 {
		return;
	}

	output[0] = input.next().unwrap();
	for i in 1..output.len() {
		output[i] = input.next().unwrap() + output[i - 1];
	}
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
pub fn exclusive_scan_transformed<IO, F>(input: &[IO], init: IO, mut transform: F) -> Vec<IO>
where
	IO: Copy,
	F: FnMut(IO, IO) -> IO,
{
	let mut output = unsafe { vec_uninit(input.len()) };
	if input.len() == 0 {
		return output;
	}

	output[0] = init;
	for i in 1..input.len() {
		output[i] = transform(output[i - 1], input[i - 1]);
	}

	output
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

pub fn exclusive_scan_iter<IO>(input: impl Iterator<Item = IO>, output: &mut [IO], init: IO)
where
	IO: Copy + AddAssign,
{
	let mut acc = init;
	for (idx, i) in input.enumerate() {
		output[idx] = acc;
		acc += i;
	}
}

///Copy values in the input range `[first, last)` to the output range
///starting from `d_first` that satisfies the predicate `pred`,
///i.e. `pred(x) == true`, and returns `d_first + n` where `n` is the number of
///times the predicate is evaluated to true.
///
///This function is stable, meaning that the relative order of elements in the
///output range remains unchanged.
///
///The input range `[first, last)` and
///the output range `[d_first, d_first + last - first)`
///must not overlap.
pub fn copy_if<IO, F>(input: impl Iterator<Item = IO>, output: &mut [IO], mut pred: F) -> usize
where
	IO: Copy,
	F: FnMut(IO) -> bool,
{
	let mut i = 0;
	for input in input {
		if pred(input) {
			output[i] = input;
			i += 1;
		}
	}

	i
}

///`scatter` copies elements from a source range into an output array according
///to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
///is assigned to `outputFirst[mapFirst[i - first]]`.  If the same index appears
///more than once in the range `[mapFirst, mapFirst + (last - first))`, the
///result is undefined.
///
///The map range, input range and the output range must not overlap.
pub fn scatter<IO, Map>(input: impl Iterator<Item = IO>, map: &[Map], output: &mut [IO])
where
	IO: Copy,
	Map: Copy + LossyInto<usize>,
{
	for (i, input) in input.enumerate() {
		output[map[i].lossy_into()] = input;
	}
}

///`gather` copies elements from a source array into a destination range
///according to a map. For each input iterator `i`
///in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
///is assigned to `outputFirst[i - map_first]`.
///
///The map range, input range and the output range must not overlap.
pub fn gather<IO, Map>(map: &[Map], input: &[IO], output: &mut [IO])
where
	IO: Copy,
	Map: Copy + LossyInto<usize>,
{
	for i in 0..map.len() {
		output[i] = input[map[i].lossy_into()];
	}
}

///`gather` copies elements from a source array into a destination range
///according to a map. For each input iterator `i`
///in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
///is assigned to `outputFirst[i - map_first]`.
///
///The map range, input range and the output range must not overlap.
pub fn gather_transformed<IO, Map, F>(
	map: &[Map],
	input: &[IO],
	output: &mut [IO],
	mut transform: F,
) where
	IO: Copy,
	Map: Copy + LossyInto<usize>,
	F: FnMut(IO) -> IO,
{
	for i in 0..map.len() {
		output[i] = transform(input[map[i].lossy_into()]);
	}
}
