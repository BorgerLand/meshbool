use crate::util::hash_table::DeterministicMap;
use std::mem;

///from https://github.com/wjakob/dset, changed to add connected component
///computation
pub struct DisjointSets {
	data: Vec<u64>,
}

impl DisjointSets {
	pub fn new(size: usize) -> Self {
		debug_assert!(size <= u32::MAX as usize);
		Self {
			data: (0..size).map(|i| i as u64).collect(),
		}
	}

	pub fn find(&mut self, id: usize) -> usize {
		self.find_impl(DisjointSets::to_index(id)) as usize
	}

	pub fn unite(&mut self, id1_in: usize, id2_in: usize) -> usize {
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

			if self.data[id1 as usize] != old_entry {
				continue;
			}
			self.data[id1 as usize] = new_entry;

			if r1 == r2 {
				old_entry = ((r2 as u64) << 32) | (id2 as u64);
				new_entry = (((r2 as u64) + 1) << 32) | (id2 as u64);
				/* Try to update the rank (may fail, retry if rank = 0) */
				if self.data[id2 as usize] == old_entry {
					self.data[id2 as usize] = new_entry;
				} else if r2 == 0 {
					continue;
				}
			}

			break;
		}

		id2 as usize
	}

	pub fn connected_components(&mut self) -> (Vec<i32>, usize) {
		let mut lonely_nodes = 0;
		let mut to_label: DeterministicMap<u32, i32> = DeterministicMap::new();
		let components = (0..self.data.len())
			.map(|i| {
				// we optimize for connected component of size 1
				// no need to put them into the hashmap
				let i_parent = self.find_impl(DisjointSets::to_index(i as usize));
				if self.rank(i_parent) == 0 {
					let component = to_label.len() as i32 + lonely_nodes as i32;
					lonely_nodes += 1;
					component
				} else if let Some(value) = to_label.get(&i_parent) {
					*value
				} else {
					let s = to_label.len() as u32 + lonely_nodes as u32;
					to_label.entry(i_parent).or_insert(s as i32);
					s as i32
				}
			})
			.collect();
		return (components, to_label.len() + lonely_nodes);
	}

	fn rank(&self, id: u32) -> u32 {
		((self.data[id as usize] >> 32) as u32) & 0x7FFFFFFF
	}

	fn parent(&self, id: u32) -> u32 {
		self.data[id as usize] as u32
	}

	fn to_index(id: usize) -> u32 {
		debug_assert!(id <= u32::MAX as usize);
		id as u32
	}

	fn find_impl(&mut self, mut id: u32) -> u32 {
		while id != self.parent(id) {
			let value = self.data[id as usize];
			let new_parent = self.parent(value as u32);
			let new_value = (value & 0xFFFFFFFF00000000) | (new_parent as u64);
			if value != new_value {
				/* Try to update parent (may fail, that's ok) */
				self.data[id as usize] = new_value;
			}

			id = new_parent;
		}

		id
	}
}
