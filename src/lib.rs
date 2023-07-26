#![deny(missing_docs)]
//! A crate providing an undirected graph struct

use std::{marker::PhantomData, ptr::NonNull};

/// A nodeof an undirected graph
struct Node<T> {
    elem: T,
}

// /// An edge of a weighted undirected graph
struct Edge<T, U> {
    start: Option<NonNull<Node<T>>>,
    end: Option<NonNull<Node<T>>>,
    value: U,
}

/// The weighted undirected graph itself
pub struct WUGraph<T, U> {
    nodes: Vec<Option<NonNull<Node<T>>>>,
    edges: Vec<Edge<T, U>>,
    _node_type: PhantomData<T>, // Phantom data so it is known that the type of the nodes is T
    _edge_type: PhantomData<T>, // Phantom data so it is know that the type of the edges is U
}

impl<T, U> Default for WUGraph<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

/// an immutable (read-only) iterator over nodes
pub struct IterNodes<'a, T> {
    nodes: std::slice::Iter<'a, Option<NonNull<Node<T>>>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over nodes
}

/// an mutable (read-only) iterator over nodes
pub struct IterMutNodes<'a, T> {
    nodes: std::slice::IterMut<'a, Option<NonNull<Node<T>>>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over nodes
}

/// an immutable (read-only) iterator over edges
pub struct IterEdges<'a, T, U> {
    edges: std::slice::Iter<'a, Edge<T, U>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over nodes
}

/// an mutable (read-only) iterator over edges
pub struct IterMutEdges<'a, T, U> {
    edges: std::slice::IterMut<'a, Edge<T, U>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over nodes
}

impl<T, U> WUGraph<T, U> {
    /// Constructor for an empty weighted undirected graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            _node_type: PhantomData,
            _edge_type: PhantomData,
        }
    }
    /// Constructor to creat a graph of unlinked nodes
    /// Should be impl FromIterator ?
    pub fn from_nodes<V: IntoIterator<Item = T>>(nodes: V) -> Self {
        Self {
            nodes: nodes
                .into_iter()
                .map(|n| unsafe {
                    Some(NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                        elem: n,
                    }))))
                })
                .collect(),
            edges: Vec::new(),
            _node_type: PhantomData,
            _edge_type: PhantomData,
        }
    }

    /// Add an edge to a graph
    pub fn add_edge(
        &mut self,
        edge_value: U,
        node_1: usize,
        node_2: usize,
    ) -> Result<(), EdgeError> {
        if node_1 == node_2 {
            return Err(EdgeError::SameNode);
        }
        let n_1 = *self.nodes.get(node_1).ok_or(EdgeError::NodeOutOfRange)?;
        let n_2 = *self.nodes.get(node_2).ok_or(EdgeError::NodeOutOfRange)?;
        let edge = Edge {
            start: n_1,
            end: n_2,
            value: edge_value,
        };
        self.edges.push(edge);
        Ok(())
    }

    /// Add nodes to the graph
    pub fn add_nodes<V: IntoIterator<Item = T>>(&mut self, nodes: V) {
        let nodes_to_add: &mut Vec<Option<NonNull<Node<T>>>> = &mut nodes
            .into_iter()
            .map(|n| unsafe {
                Some(NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                    elem: n,
                }))))
            })
            .collect();
        self.nodes.append(nodes_to_add)
    }

    /// return an immutable iterator over nodes
    pub fn iter_nodes(&self) -> IterNodes<T> {
        IterNodes {
            nodes: self.nodes.iter(),
            _boo: PhantomData,
        }
    }

    /// return an mutable iterator over nodes
    pub fn iter_mut_nodes(&mut self) -> IterMutNodes<T> {
        IterMutNodes {
            nodes: self.nodes.iter_mut(),
            _boo: PhantomData,
        }
    }

    /// return an immutable iterator over nodes
    pub fn iter_edges(&self) -> IterEdges<T, U> {
        IterEdges {
            edges: self.edges.iter(),
            _boo: PhantomData,
        }
    }

    /// return an mutable iterator over nodes
    pub fn iter_mut_edges(&mut self) -> IterMutEdges<T, U> {
        IterMutEdges {
            edges: self.edges.iter_mut(),
            _boo: PhantomData,
        }
    }

    /// Drop all edges in the graph
    pub fn drop_all_edges(&mut self) {
        self.edges.truncate(0)
    }

    /// Drop all nodes in the graph
    pub fn drop_all_nodes(&mut self) {
        // first drop all edges as cannot be
        // any edges in a graph with no nodes
        self.drop_all_edges();
        // let box memory management handle the dropping of the raw pointers
        self.nodes.drain(..).for_each(|node_pointer| {
            node_pointer.map(|node| unsafe { Box::from_raw(node.as_ptr()) });
        })
    }
}

impl<T, U> Drop for WUGraph<T, U> {
    fn drop(&mut self) {
        self.drop_all_nodes()
    }
}

impl<'a, T> Iterator for IterNodes<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.nodes.by_ref().flatten().next() {
            //if let Some(node) = node_option {
            // SAFETY: We know that the node is not null because it is Some
            let node_ref = unsafe { node.as_ref() };
            return Some(&node_ref.elem);
            //}
        }
        None
    }
}

impl<'a, T> Iterator for IterMutNodes<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.nodes.by_ref().flatten().next() {
            //if let Some(node) = node_option {
            // SAFETY: We know that the node is not null because it is Some
            let node_ref = unsafe { node.as_mut() };
            return Some(&mut node_ref.elem);
            //}
        }
        None
    }
}

impl<'a, T, U> Iterator for IterEdges<'a, T, U> {
    type Item = (&'a T, &'a T, &'a U);

    fn next(&mut self) -> Option<Self::Item> {
        for edge in self.edges.by_ref() {
            if let (Some(node_1), Some(node_2)) = (edge.start, edge.end) {
                // SAFETY: We know that the node is not null because it is Some
                let node_1_ref = unsafe { node_1.as_ref() };
                let node_2_ref = unsafe { node_2.as_ref() };
                return Some((&node_1_ref.elem, &node_2_ref.elem, &edge.value));
            }
        }
        None
    }
}

impl<'a, T, U> Iterator for IterMutEdges<'a, T, U> {
    type Item = (&'a mut T, &'a mut T, &'a mut U);

    fn next(&mut self) -> Option<Self::Item> {
        for edge in self.edges.by_ref() {
            if let (Some(mut node_1), Some(mut node_2)) = (edge.start, edge.end) {
                // SAFETY: We know that the node is not null because it is Some
                let node_1_ref = unsafe { node_1.as_mut() };
                let node_2_ref = unsafe { node_2.as_mut() };
                return Some((&mut node_1_ref.elem, &mut node_2_ref.elem, &mut edge.value));
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
/// An error regarding edge creation
pub enum EdgeError {
    /// The node is out of range of the present nodes
    NodeOutOfRange,
    /// the edge is connecting a node to itself
    SameNode,
}

#[cfg(test)]
mod test {
    use super::WUGraph;
    #[test]
    fn basics() {
        let points = vec![1, 2, 3];
        let mut graph: WUGraph<i32, &str> = WUGraph::from_nodes(points);
        // we can add an edge
        graph.add_edge("from 0 to 2", 0, 2).unwrap();
        println!("here");
        // check immutable iterators
        graph.iter_nodes().for_each(|n| println!("{n}"));
        graph
            .iter_edges()
            .for_each(|(start, end, value)| println!("Edge {value} connects {start} and {end}"));
        // check mutable iterators
        graph.iter_mut_nodes().for_each(|n| *n -= 1);
        // check assignment
        graph.iter_nodes().for_each(|n| println!("{n}"));
        // panic!()
    }
}
