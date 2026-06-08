///calls vec.resize() and potentially vec.shrink_to_fit()
pub fn vec_resize<T>(vec: &mut Vec<T>, new_size: usize, val: T)
where
	T: Clone,
{
	let shrink = vec.len() > 2 * new_size && vec.len() > 16;
	vec.resize(new_size, val.clone());
	if shrink {
		vec.shrink_to_fit();
	}
}

///safety: any new elements added to the vec are uninitialized
pub unsafe fn vec_resize_nofill<T>(vec: &mut Vec<T>, new_size: usize) {
	//no-op
	if new_size == vec.len() {
		return;
	}

	//shrink
	if new_size < vec.len() {
		let shrink = vec.len() > 2 * new_size && vec.len() > 16;
		vec.truncate(new_size);
		if shrink {
			vec.shrink_to_fit();
		}

		return;
	}

	//grow
	vec.reserve(new_size - vec.len());
	unsafe {
		vec.set_len(new_size);
	}
}

///safety: all elements are uninitialized
pub unsafe fn vec_uninit<T>(size: usize) -> Vec<T> {
	let mut vec = Vec::with_capacity(size);
	unsafe {
		vec.set_len(size);
	}
	vec
}

//c++ std::partition
pub fn partition<T, F>(slice: &mut [T], mut predicate: F) -> usize
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
