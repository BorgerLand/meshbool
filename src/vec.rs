use std::collections::VecDeque;

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

//this is used to recreate c++ multiset/multimap in a probably
//very poorly optimized way. logarithmic lookup to find insertion
//point, linear insertion time moving larger elements up one slot.
//if new_item is equal to at least 1 existing_item, it will be
//placed at the end of the group of equal items (highest index)
pub trait InsertSorted<T> {
	fn partition_point<P>(&self, pred: P) -> usize
	where
		P: FnMut(&T) -> bool;

	fn insert(&mut self, index: usize, element: T);

	//replicates c++ multiset insertion, where
	fn insert_sorted_by_key<K, F>(&mut self, new_item: T, mut key_fn: F)
	where
		K: Ord,
		F: FnMut(&T) -> K,
	{
		let new_key = key_fn(&new_item);
		let index = self.partition_point(|existing_item| key_fn(existing_item) <= new_key);

		self.insert(index, new_item);
	}
}

impl<T> InsertSorted<T> for Vec<T> {
	fn partition_point<P>(&self, pred: P) -> usize
	where
		P: FnMut(&T) -> bool,
	{
		self.as_slice().partition_point(pred)
	}

	fn insert(&mut self, index: usize, element: T) {
		self.insert(index, element);
	}
}

impl<T> InsertSorted<T> for VecDeque<T> {
	fn partition_point<P>(&self, pred: P) -> usize
	where
		P: FnMut(&T) -> bool,
	{
		self.partition_point(pred)
	}

	fn insert(&mut self, index: usize, element: T) {
		self.insert(index, element);
	}
}
