use crate::util::hash_table::DeterministicMap;
use crate::util::vec_ext;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};

///from https://github.com/wjakob/dset, changed to add connected component
///computation
pub struct DisjointSets {
	data: Vec<AtomicU64>,
}

impl DisjointSets {
	pub fn new(size: usize) -> Self {
		debug_assert!(size <= u32::MAX as usize);
		Self {
			data: (0..size).map(|i| AtomicU64::new(i as u64)).collect(),
		}
	}

	pub fn find(&self, id: usize) -> usize {
		self.find_impl(DisjointSets::to_index(id)) as usize
	}

	pub fn unite(&self, id1_in: usize, id2_in: usize) -> usize {
		let mut id1 = DisjointSets::to_index(id1_in);
		let mut id2 = DisjointSets::to_index(id2_in);
		loop {
			id1 = self.find_impl(id1);
			id2 = self.find_impl(id2);

			if id1 == id2 {
				return id1 as usize;
			}

			let mut r1 = self.rank(id1);
			let mut r2 = self.rank(id2);

			if r1 > r2 || (r1 == r2 && id1 < id2) {
				mem::swap(&mut r1, &mut r2);
				mem::swap(&mut id1, &mut id2);
			}

			let mut old_entry = ((r1 as u64) << 32) | (id1 as u64);
			let mut new_entry = ((r1 as u64) << 32) | (id2 as u64);

			if self.data[id1 as usize]
				.compare_exchange(old_entry, new_entry, Ordering::SeqCst, Ordering::SeqCst)
				.is_err()
			{
				continue;
			}

			if r1 == r2 {
				old_entry = ((r2 as u64) << 32) | (id2 as u64);
				new_entry = (((r2 as u64) + 1) << 32) | (id2 as u64);
				/* Try to update the rank (may fail, retry if rank = 0) */
				if self.data[id2 as usize]
					.compare_exchange(old_entry, new_entry, Ordering::SeqCst, Ordering::SeqCst)
					.is_err() && r2 == 0
				{
					continue;
				}
			}

			break;
		}

		id2 as usize
	}

	pub fn connected_components(&self) -> (Vec<i32>, usize) {
		let mut components = unsafe { vec_ext::uninit(self.data.len()) };
		let mut lonely_nodes = 0;
		let mut to_label: DeterministicMap<u32, i32> = DeterministicMap::new();
		for i in 0..self.data.len() {
			// we optimize for connected component of size 1
			// no need to put them into the hashmap
			let i_parent = self.find_impl(DisjointSets::to_index(i as usize));
			if self.rank(i_parent) == 0 {
				components[i] = to_label.len() as i32 + lonely_nodes as i32;
				lonely_nodes += 1;
				continue;
			}
			if let Some(value) = to_label.get(&i_parent) {
				components[i] = *value;
			} else {
				let s = to_label.len() as u32 + lonely_nodes as u32;
				to_label.entry(i_parent).or_insert(s as i32);
				components[i] = s as i32;
			}
		}
		return (components, to_label.len() + lonely_nodes);
	}

	fn rank(&self, id: u32) -> u32 {
		((self.data[id as usize].load(Ordering::SeqCst) >> 32) as u32) & 0x7FFFFFFF
	}

	fn parent(&self, id: u32) -> u32 {
		self.data[id as usize].load(Ordering::SeqCst) as u32
	}

	fn to_index(id: usize) -> u32 {
		debug_assert!(id <= u32::MAX as usize);
		id as u32
	}

	fn find_impl(&self, mut id: u32) -> u32 {
		while id != self.parent(id) {
			let value = self.data[id as usize].load(Ordering::SeqCst);
			let new_parent = self.parent(value as u32);
			let new_value = (value & 0xFFFFFFFF00000000) | (new_parent as u64);
			if value != new_value {
				/* Try to update parent (may fail, that's ok) */
				#[allow(unused_must_use)]
				self.data[id as usize].compare_exchange_weak(
					value,
					new_value,
					Ordering::SeqCst,
					Ordering::SeqCst,
				);
			}

			id = new_parent;
		}

		id
	}
}
