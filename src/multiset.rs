use intrusive_collections::{KeyAdapter, RBTree, RBTreeLink, intrusive_adapter};
use std::ptr::NonNull;

///A node holding a key and value, with links + key + value all inline in a
///single `Box` allocation (mirroring the layout of a C++ red-black tree node).
pub struct Node<K> {
	link: RBTreeLink,
	key: K,
}

intrusive_adapter!(NodeAdapter<K> = Box<Node<K>>: Node<K> { link => RBTreeLink });

impl<'a, K: 'a> KeyAdapter<'a> for NodeAdapter<K> {
	type Key = &'a K;
	fn get_key(&self, node: &'a Node<K>) -> Self::Key {
		&node.key
	}
}

///A thin wrapper around a pointer to a specific entry of a MultiSet, which
///guarantees stable pointers
pub struct Handle<K>(NonNull<Node<K>>);
impl<K> Copy for Handle<K> {}
impl<K> Clone for Handle<K> {
	fn clone(&self) -> Self {
		*self
	}
}

///A multiset with C++ `std::multiset` semantics: duplicate keys are allowed and
///kept in sorted order, with equal keys retained in insertion order. For duplicate
///keys, equal elements are currently traversed in insertion order (a newly
///inserted equal key is placed to the right of existing equal keys, and in-order
///traversal is stable across rebalancing). This is a property of the current
///implementation, not a documented guarantee of the crate, but this property is
///necessary for matching the behavior of the C++ standard. Updating the crate
///should be met with scrutiny for this reason.
///
///I am not 100% sure this is a good idea for performance due to lack of cache
///locality. The previous commit was using a manually sorted vec, which was
///slightly faster, but Samples.Sponge4 was failing
///(Expected: (sponge.NumDegenerateTris()) <= (8), actual: 24 vs 8).
///Possibly the stable pointer guarantee as opposed to removing any identical key
///is load bearing.
pub struct MultiSet<K: 'static> {
	tree: RBTree<NodeAdapter<K>>,
}

impl<K: 'static> MultiSet<K> {
	pub fn new() -> Self {
		MultiSet {
			tree: RBTree::new(NodeAdapter::new()),
		}
	}

	///Inserts a `(key, value)` pair. Never overwrites; equal keys coexist and
	///are inserted after any existing equal keys. The returned handle is valid
	///until the entry is removed.
	///
	///Ordering is not taken from `K: Ord`. Instead it is supplied by `less_than`, a
	///per-method callback-based comparator function. The same comparison logic must
	///be used across calls in order to maintain a consistent meaning as to what
	///constitutes as "ordered". This hack is necessary because the ear clipper is
	///designed in a way that violently violates Rust aliasing rules.
	pub fn insert(&mut self, less_than: impl Fn(&K, &K) -> bool, key: K) -> Handle<K> {
		let cursor = self.tree.insert(
			|a, b| less_than(*a, *b),
			Box::new(Node {
				link: RBTreeLink::new(),
				key,
			}),
		);

		let ptr = cursor.get().unwrap() as *const Node<K> as *mut Node<K>;
		Handle(unsafe { NonNull::new_unchecked(ptr) })
	}

	///Iterates over all `(key, value)` pairs in ascending order.
	///For keys that compare equally, maintains insertion order.
	pub fn iter(&self) -> impl Iterator<Item = &K> {
		self.tree.iter().map(|n| &n.key)
	}

	///Removes and returns the smallest-key and oldest-inserted element, or
	///`None` if empty.
	pub fn pop_front(&mut self) -> Option<K> {
		self.tree.front_mut().remove().map(|n| n.key)
	}

	///Removes the exact element named by `handle`, returning its owned key.
	///
	///# Safety
	///`handle` must have come from [`insert`](Self::insert) on *this* set and
	///the element must not have already been removed. Violating this is
	///undefined behavior.
	pub unsafe fn remove(&mut self, handle: Handle<K>) -> K {
		let mut cursor = unsafe { self.tree.cursor_mut_from_ptr(handle.0.as_ptr()) };
		let node = cursor.remove().unwrap();
		node.key
	}
}
