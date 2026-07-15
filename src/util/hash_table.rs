use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::ops::{Deref, DerefMut};

//wrappers around hashmap/hashset to remove the nondeterministic build hasher
#[derive(Clone, Debug)]
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

impl<K, V> IntoIterator for DeterministicMap<K, V> {
	type Item = <HashMap<K, V> as IntoIterator>::Item;
	type IntoIter = <HashMap<K, V> as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		self.0.into_iter()
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

impl<K> FromIterator<K> for DeterministicSet<K>
where
	K: Eq + Hash,
{
	fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
		Self(HashSet::from_iter(iter))
	}
}

impl<K> IntoIterator for DeterministicSet<K> {
	type Item = <HashSet<K> as IntoIterator>::Item;
	type IntoIter = <HashSet<K> as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		self.0.into_iter()
	}
}

impl<K> DeterministicSet<K> {
	pub fn new() -> Self {
		Self(HashSet::with_hasher(BuildHasherDefault::default()))
	}
}
