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

impl<K, V, const N: usize> From<[(K, V); N]> for DeterministicMap<K, V>
where
	K: Eq + Hash,
{
	fn from(arr: [(K, V); N]) -> Self {
		Self(HashMap::from_iter(arr))
	}
}

impl<K, V> FromIterator<(K, V)> for DeterministicMap<K, V>
where
	K: Eq + Hash,
{
	fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
		Self(HashMap::from_iter(iter))
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
