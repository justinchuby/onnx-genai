//! A generational-free slot arena keyed by a typed index.
//!
//! [`Graph`](crate::Graph) stores nodes and values in `Arena`s. Removal leaves
//! a tombstone and recycles the slot on the next insert, so live ids stay
//! stable across unrelated mutations. Ids are **not** generational: reusing a
//! removed id after its slot has been recycled will alias the new occupant.
//! Optimization passes are expected to drop stale ids when they remove nodes.

use std::marker::PhantomData;

/// A type usable as an arena key: a newtype around a `u32` index.
pub trait ArenaKey: Copy {
    /// Build a key from a raw slot index.
    fn from_raw(raw: u32) -> Self;
    /// The raw slot index of this key.
    fn to_raw(self) -> u32;
}

/// A slot map: dense `Vec` storage with `O(1)` insert/remove/lookup by key.
#[derive(Clone, Debug)]
pub struct Arena<K: ArenaKey, T> {
    slots: Vec<Option<T>>,
    free: Vec<u32>,
    len: usize,
    _marker: PhantomData<K>,
}

impl<K: ArenaKey, T> Default for Arena<K, T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }
}

impl<K: ArenaKey, T> Arena<K, T> {
    /// An empty arena.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of allocated slots, including tombstones.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Insert `value`, returning its freshly allocated key.
    pub fn insert(&mut self, value: T) -> K {
        self.insert_with(|_| value)
    }

    /// Insert a value that needs to know its own key (e.g. a node that stores
    /// its own [`NodeId`](crate::NodeId)).
    pub fn insert_with(&mut self, make: impl FnOnce(K) -> T) -> K {
        let key = match self.free.pop() {
            Some(idx) => K::from_raw(idx),
            None => {
                let idx = self.slots.len() as u32;
                self.slots.push(None);
                K::from_raw(idx)
            }
        };
        self.slots[key.to_raw() as usize] = Some(make(key));
        self.len += 1;
        key
    }

    /// Whether `key` refers to a live entry.
    pub fn contains(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Borrow the entry for `key`, if live.
    pub fn get(&self, key: K) -> Option<&T> {
        self.slots.get(key.to_raw() as usize).and_then(Option::as_ref)
    }

    /// Mutably borrow the entry for `key`, if live.
    pub fn get_mut(&mut self, key: K) -> Option<&mut T> {
        self.slots
            .get_mut(key.to_raw() as usize)
            .and_then(Option::as_mut)
    }

    /// Remove and return the entry for `key`, recycling the slot.
    pub fn remove(&mut self, key: K) -> Option<T> {
        let slot = self.slots.get_mut(key.to_raw() as usize)?;
        let taken = slot.take();
        if taken.is_some() {
            self.free.push(key.to_raw());
            self.len -= 1;
        }
        taken
    }

    /// Iterate over `(key, &value)` for all live entries, in ascending key order.
    pub fn iter(&self) -> impl Iterator<Item = (K, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|v| (K::from_raw(i as u32), v)))
    }

    /// Iterate over the keys of all live entries.
    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|_| K::from_raw(i as u32)))
    }

    /// Iterate over `&value` for all live entries.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(Option::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct K(u32);
    impl ArenaKey for K {
        fn from_raw(raw: u32) -> Self {
            K(raw)
        }
        fn to_raw(self) -> u32 {
            self.0
        }
    }

    #[test]
    fn insert_get_remove() {
        let mut a: Arena<K, &str> = Arena::new();
        assert!(a.is_empty());
        let x = a.insert("x");
        let y = a.insert("y");
        assert_eq!(a.len(), 2);
        assert_eq!(a.get(x), Some(&"x"));
        assert_eq!(a.get(y), Some(&"y"));
        assert_eq!(a.remove(x), Some("x"));
        assert!(!a.contains(x));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn slots_are_recycled() {
        let mut a: Arena<K, u32> = Arena::new();
        let x = a.insert(1);
        a.remove(x);
        let z = a.insert(2);
        // freed slot reused
        assert_eq!(x.to_raw(), z.to_raw());
        assert_eq!(a.get(z), Some(&2));
    }

    #[test]
    fn insert_with_sees_own_key() {
        let mut a: Arena<K, K> = Arena::new();
        let k = a.insert_with(|self_key| self_key);
        assert_eq!(a.get(k), Some(&k));
    }

    #[test]
    fn iter_skips_tombstones() {
        let mut a: Arena<K, u32> = Arena::new();
        let x = a.insert(10);
        let _y = a.insert(20);
        a.remove(x);
        let collected: Vec<u32> = a.values().copied().collect();
        assert_eq!(collected, vec![20]);
    }
}
