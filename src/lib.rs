#![deny(missing_docs)]
#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style
)]

//! A crate providing an undirected graph struct

use std::{marker::PhantomData, ptr::NonNull};

/// A node of an undirected graph
struct Node<T, U> {
    elem: T,
    connections: Vec<Option<NonNull<Edge<T, U>>>>,
}

/// An edge of a weighted undirected graph
struct Edge<T, U> {
    start: Option<NonNull<Node<T, U>>>,
    end: Option<NonNull<Node<T, U>>>,
    value: U,
}

/// The weighted undirected graph itself
pub struct SimpleGraph<T, U> {
    nodes: Vec<Option<NonNull<Node<T, U>>>>,
    edges: Vec<Option<NonNull<Edge<T, U>>>>,
    _node_type: PhantomData<T>, // Phantom data so it is known that the type of the nodes is T
    _edge_type: PhantomData<U>, // Phantom data so it is know that the type of the edges is U
}

impl<T, U> Default for SimpleGraph<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

/// an immutable (read-only) iterator over nodes returning solely the contents of the node
pub struct IterNodes<'a, T, U> {
    nodes: std::slice::Iter<'a, Option<NonNull<Node<T, U>>>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over nodes
}

/// an immutable (read-only) iterator over nodes returning the contents of the node and a Vec containing the contents of aconnecting edges
pub struct IterNodesEdge<'a, T, U>(IterNodes<'a, T, U>);

/// an immutable (read-only) iterator over nodes returning the contents of the node and a Vec containing the contents of adjacent nodes
pub struct IterNodesAdj<'a, T, U>(IterNodes<'a, T, U>);

/// an mutable (read+write) iterator over nodes returning solely the contents of the node
pub struct IterMutNodes<'a, T, U> {
    nodes: std::slice::IterMut<'a, Option<NonNull<Node<T, U>>>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over nodes
}

/// an mutable (read+write) iterator over nodes returning the contents of the node and a Vec containing the contents of connecting edges
pub struct IterMutNodesEdge<'a, T, U>(IterMutNodes<'a, T, U>);

/// an mutable (read+write) iterator over nodes returning the contents of the node and a Vec containing the contents of adjacent nodes
pub struct IterMutNodesAdj<'a, T, U>(IterMutNodes<'a, T, U>);

/// an immutable (read-only) iterator over edges returning each edge and the contents of the nodes they connect
pub struct IterEdges<'a, T, U> {
    edges: std::slice::Iter<'a, Option<NonNull<Edge<T, U>>>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over edges
}

/// an mutable (read-write) iterator over edges returning each edge and the contents of the nodes they connect
pub struct IterMutEdges<'a, T, U> {
    edges: std::slice::IterMut<'a, Option<NonNull<Edge<T, U>>>>, // the contents of the iterator
    _boo: PhantomData<&'a T>, // lifetime annotation to ensure the graph last as long as the iterator over edges
}

impl<T, U> SimpleGraph<T, U> {
    //---------------------Constructors------------------------------------------
    /// Constructor for an empty weighted undirected graph
    /// ```
    /// # use graph::SimpleGraph;
    /// let graph:SimpleGraph<i32,f64> = SimpleGraph::new();
    /// ```
    #[must_use] // must use as if not assigned to anything produces a value with no use
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            _node_type: PhantomData,
            _edge_type: PhantomData,
        }
    }

    /// Constructor to creat a graph of unlinked nodes
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// ```
    pub fn from_nodes<V: IntoIterator<Item = T>>(nodes: V) -> Self {
        // Should be impl FromIterator ?
        Self {
            nodes: nodes
                .into_iter()
                .map(|n| unsafe {
                    Some(NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                        elem: n,
                        connections: Vec::new(),
                    }))))
                })
                .collect(),
            edges: Vec::new(),
            _node_type: PhantomData,
            _edge_type: PhantomData,
        }
    }

    //-----------------------------------Add Elements----------------------------------------------
    /// Add an edge to a graph
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 2).unwrap();
    /// ```
    /// # Errors
    /// `SameNode`: connecting a node to itself is not allowed in a simple graph
    /// ```should_panic
    /// # use graph::SimpleGraph;
    /// # let nodes = [1, 2, 3];
    /// # let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 0).unwrap();
    /// ```
    ///
    /// `NodeOutOfRange`: attempting to connect to a node not in the graph
    /// ```should_panic
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 3).unwrap();
    /// ```
    ///
    /// `MultipleConnection`: attempting to connect the same two nodes by more than 1 edge
    /// ```should_panic
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 2).unwrap();
    /// graph.add_edge(0.2, 0, 2).unwrap();
    /// ```
    pub fn add_edge(
        &mut self,
        edge_value: U,
        node_1: usize,
        node_2: usize,
    ) -> Result<(), EdgeError> {
        if node_1 == node_2 {
            return Err(EdgeError::SameNode);
        }
        if self.get_edge(node_1, node_2).is_some() {
            return Err(EdgeError::MultipleConnection);
        }
        let n_1 = *self.nodes.get(node_1).ok_or(EdgeError::NodeOutOfRange)?;
        let n_2 = *self.nodes.get(node_2).ok_or(EdgeError::NodeOutOfRange)?;
        let edge = Edge {
            start: n_1,
            end: n_2,
            value: edge_value,
        };
        let edge_pointer = unsafe { Some(NonNull::new_unchecked(Box::into_raw(Box::new(edge)))) };
        // push into the big edge list first
        self.edges.push(edge_pointer);

        // These must exist as we've checked for their existence just above
        if let (Some(mut node_1), Some(mut node_2)) = (n_1, n_2) {
            // SAFETY: We know that the node is not null because it is Some
            let node_1 = unsafe { node_1.as_mut() };
            let node_2 = unsafe { node_2.as_mut() };
            node_1.connections.push(edge_pointer);
            node_2.connections.push(edge_pointer);
        }

        Ok(())
    }

    /// Add nodes to the graph
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_nodes([4, 5]);
    /// ```
    pub fn add_nodes<V: IntoIterator<Item = T>>(&mut self, nodes: V) {
        let nodes_to_add: &mut Vec<Option<NonNull<Node<T, U>>>> = &mut nodes
            .into_iter()
            .map(|n| unsafe {
                Some(NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                    elem: n,
                    connections: Vec::new(),
                }))))
            })
            .collect();
        self.nodes.append(nodes_to_add);
    }

    //------------------------------------get elements---------------------------------------

    /// get an edge between node 1 and node 2 if it exists
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 2);
    /// assert_eq!( graph.get_edge(0, 2), Some(&0.2f64));
    /// assert_eq!( graph.get_edge(0, 1), None); // none as there is no edge
    /// assert_eq!( graph.get_edge(0, 3), None); // none as node 3 does not exist
    /// ```
    #[must_use]
    pub fn get_edge(&self, node_1: usize, node_2: usize) -> Option<&U> {
        let edge = self._get_edge(node_1, node_2);
        if let Some(edge) = edge {
            return Some(unsafe { &(edge.as_ref()).value });
        }
        None
    }

    /// get a mutable reference to an edge between node 1 and node 2 if it exists
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 2);
    /// assert_eq!( graph.get_edge_mut(0, 2), Some(&mut 0.2f64));
    /// assert_eq!( graph.get_edge_mut(0, 1), None); // none as there is no edge
    /// assert_eq!( graph.get_edge_mut(0, 3), None) // none as node 3 does not exist
    /// ```
    #[must_use]
    pub fn get_edge_mut(&mut self, node_1: usize, node_2: usize) -> Option<&mut U> {
        let edge = self._get_edge(node_1, node_2);
        if let Some(mut edge) = edge {
            return Some(unsafe { &mut (edge.as_mut()).value });
        }
        None
    }

    /// get a node by index
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// assert_eq!( graph.get_node(0), Some(&1i32));
    /// ```
    pub fn get_node(&mut self, node: usize) -> Option<&T> {
        if let Some(Some(node)) = self.nodes.get(node) {
            let node = unsafe { &(node.as_ref()).elem };
            Some(node)
        } else {
            None
        }
    }

    /// get a mutable node by index
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// assert_eq!( graph.get_node_mut(0), Some(&mut 1i32));
    /// ```
    pub fn get_node_mut(&mut self, node: usize) -> Option<&mut T> {
        if let Some(Some(mut node)) = self.nodes.get(node) {
            let node = unsafe { &mut (node.as_mut()).elem };
            Some(node)
        } else {
            None
        }
    }

    /// get an edge between node 1 and node 2 if it exists
    fn _get_edge(&self, node_1: usize, node_2: usize) -> Option<NonNull<Edge<T, U>>> {
        if let (Some(Some(node_1)), Some(node_2)) = (self.nodes.get(node_1), self.nodes.get(node_2))
        {
            let node_ref = unsafe { node_1.as_ref() };
            let edge: Option<NonNull<Edge<T, U>>> = node_ref
                .connections
                .iter()
                .flatten()
                .find(|edge| {
                    (unsafe { &(edge.as_ref()).start } == node_2)
                        | (unsafe { &(edge.as_ref()).end } == node_2)
                })
                .copied();
            if let Some(edge) = edge {
                return Some(edge);
            }
        }
        None
    }

    //------------------------------------Drop elements---------------------------------

    /// remove the edge between node 1 and 2 if it exists returning the value inside
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(0.2, 0, 2);
    /// assert_eq!( graph.drop_edge(0, 2), Some(0.2f64));
    /// assert_eq!( graph.get_edge_mut(0, 2), None); // The edge no longer exists
    /// ```
    pub fn drop_edge(&mut self, node_1: usize, node_2: usize) -> Option<U> {
        let edge_to_drop = self._get_edge(node_1, node_2);
        if let (Some(Some(mut node_1)), Some(Some(mut node_2))) =
            (self.nodes.get(node_1), self.nodes.get(node_2))
        {
            // SAFETY: We know that the node is not null because it is Some
            // mutate one at a time
            let node_1 = unsafe { node_1.as_mut() };
            node_1.connections.retain(|e| *e != edge_to_drop);
            let node_2 = unsafe { node_2.as_mut() };
            node_2.connections.retain(|e| *e != edge_to_drop);
        }
        self.edges.retain(|e| *e != edge_to_drop);
        if let Some(edge) = edge_to_drop {
            // this exists to take ownesrhip of the memory now
            // on the heap and be dropped and so deallocate that memory
            let box_scope = unsafe { Box::from_raw(edge.as_ptr()) };
            Some(box_scope.value)
        } else {
            None
        }
    }

    /// remove a specific edge
    fn _drop_edge(&mut self, edge_to_drop: Option<NonNull<Edge<T, U>>>) -> Option<U> {
        if let Some(edge_ref) = edge_to_drop {
            let edge = unsafe { edge_ref.as_ref() };
            if let (Some(mut node_1), Some(mut node_2)) = (edge.start, edge.end) {
                let node_1 = unsafe { node_1.as_mut() };
                node_1.connections.retain(|e| *e != edge_to_drop);
                let node_2 = unsafe { node_2.as_mut() };
                node_2.connections.retain(|e| *e != edge_to_drop);
            }
            self.edges.retain(|e| *e != edge_to_drop);
            // take ownership to drop value from scope now all pointers are dropped
            let box_scope = unsafe { Box::from_raw(edge_ref.as_ptr()) };
            Some(box_scope.value)
        } else {
            None
        }
    }

    /// drop a node based on index returning the contents
    ///
    /// will reindex other nodes
    ///
    /// time complexity propotional to number of connecting edges
    ///
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// assert_eq!( graph.drop_node(0), Some(1i32));
    /// assert_eq!( graph.get_node(0), Some(&2i32)); // other nodes reindexed
    /// assert_eq!( graph.get_node(2), None); // one fewer node
    /// ```
    pub fn drop_node(&mut self, node: usize) -> Option<T> {
        if let Some(node_ref) = self.nodes.get(node) {
            self._drop_node(*node_ref)
        } else {
            None
        }
    }

    // internal method for dropping a node
    fn _drop_node(&mut self, node: Option<NonNull<Node<T, U>>>) -> Option<T> {
        if let Some(mut node_ref) = node {
            let node = unsafe { node_ref.as_mut() };
            // delete all edges connecting the node
            for edge in node.connections.drain(..) {
                self._drop_edge(edge);
            }
            self.nodes.retain(|n| n != &Some(node_ref));
            // take ownership and drop
            let box_scope = unsafe { Box::from_raw(node_ref.as_ptr()) };
            Some(box_scope.elem)
        } else {
            None
        }
    }

    /// filter nodes based on a predicate on the value of the node
    /// drop nodes fulfilling a property
    ///
    /// returns a vec of the contents of dropped nodes
    ///
    /// will reindex other nodes
    ///
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3, 4, 5];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// let dropped = graph.drop_nodes_by(|val| (val > &2) & (val < &5));
    /// assert_eq!( dropped, [3, 4]);
    /// assert_eq!( graph.get_node(0), Some(&1i32));
    /// assert_eq!( graph.get_node(1), Some(&2i32));
    /// assert_eq!( graph.get_node(2), Some(&5i32)); // other nodes reindexed
    /// ```
    pub fn drop_nodes_by<Pred: Fn(&T) -> bool>(&mut self, predicate: Pred) -> Vec<T>
    where
        T: Copy,
    {
        let mut dropped: Vec<T> = Vec::new();
        let todrop: Vec<Option<NonNull<Node<T, U>>>> = self
            .nodes
            .iter()
            .filter(|node_ref| {
                if let Some(node) = node_ref {
                    unsafe { predicate(&(node.as_ref()).elem) }
                } else {
                    false
                }
            })
            .copied()
            .collect();

        for node_to_drop in todrop {
            let dropped_node = self._drop_node(node_to_drop);
            if let Some(val) = dropped_node {
                dropped.push(val);
            }
        }
        dropped
    }

    /// filter nodes based on a predicate on the value of the edge
    /// drop edges fulfilling a property
    ///
    /// returns a vec of the contents of dropped edges
    ///
    /// ```
    /// # use graph::SimpleGraph;
    /// let nodes = [1, 2, 3, 4, 5];
    /// let mut graph:SimpleGraph<i32,f64> = SimpleGraph::from_nodes(nodes);
    /// graph.add_edge(3.4, 0, 2).unwrap();
    /// graph.add_edge(4., 0, 3).unwrap();
    /// graph.add_edge(4.7, 0, 1).unwrap();
    /// let dropped = graph.drop_edges_by(|val| (val > &3.5) & (val < &4.5));
    /// assert_eq!( dropped, [4.]);
    /// assert_eq!( graph.get_edge(0, 2), Some(&3.4f64));
    /// assert_eq!( graph.get_edge(0, 3), None); // this edge has been filtered out
    /// assert_eq!( graph.get_edge(0, 1), Some(&4.7f64));
    /// ```
    pub fn drop_edges_by<Pred: Fn(&U) -> bool>(&mut self, predicate: Pred) -> Vec<U>
    where
        U: Copy,
    {
        let mut dropped: Vec<U> = Vec::new();
        let todrop: Vec<Option<NonNull<Edge<T, U>>>> = self
            .edges
            .iter()
            .filter(|edge_ref| {
                if let Some(edge) = edge_ref {
                    unsafe { predicate(&(edge.as_ref()).value) }
                } else {
                    false
                }
            })
            .copied()
            .collect();

        for edge_to_drop in todrop {
            let dropped_edge = self._drop_edge(edge_to_drop);
            if let Some(val) = dropped_edge {
                dropped.push(val);
            }
        }
        dropped
    }

    //-------------------------------------Get Iterators------------------------------------------------
    /// return an immutable iterator over nodes
    #[must_use]
    pub fn iter_nodes(&self) -> IterNodes<T, U> {
        IterNodes {
            nodes: self.nodes.iter(),
            _boo: PhantomData,
        }
    }

    /// return an immutable iterator over nodes
    /// returning node contents and a vec of the
    /// value of connected edges
    #[must_use]
    pub fn iter_nodes_edges(&self) -> IterNodesEdge<T, U> {
        IterNodesEdge(IterNodes {
            nodes: self.nodes.iter(),
            _boo: PhantomData,
        })
    }

    /// return an immutable iterator over nodes
    /// returning node contents and a vec of the
    /// value of adjacent nodes
    #[must_use]
    pub fn iter_nodes_adj(&self) -> IterNodesAdj<T, U> {
        IterNodesAdj(IterNodes {
            nodes: self.nodes.iter(),
            _boo: PhantomData,
        })
    }

    /// return an mutable iterator over nodes
    pub fn iter_mut_nodes(&mut self) -> IterMutNodes<T, U> {
        IterMutNodes {
            nodes: self.nodes.iter_mut(),
            _boo: PhantomData,
        }
    }

    /// return an mutable iterator over nodes and a vec of the adjacent edges
    pub fn iter_mut_nodes_edges(&mut self) -> IterMutNodesEdge<T, U> {
        IterMutNodesEdge(IterMutNodes {
            nodes: self.nodes.iter_mut(),
            _boo: PhantomData,
        })
    }

    /// return an mutable iterator over nodes and a vec of the adjacent nodes
    pub fn iter_mut_nodes_adj(&mut self) -> IterMutNodesAdj<T, U> {
        IterMutNodesAdj(IterMutNodes {
            nodes: self.nodes.iter_mut(),
            _boo: PhantomData,
        })
    }

    /// return an immutable iterator over edges
    #[must_use]
    pub fn iter_edges(&self) -> IterEdges<T, U> {
        IterEdges {
            edges: self.edges.iter(),
            _boo: PhantomData,
        }
    }

    /// return an mutable iterator over edges
    pub fn iter_mut_edges(&mut self) -> IterMutEdges<T, U> {
        IterMutEdges {
            edges: self.edges.iter_mut(),
            _boo: PhantomData,
        }
    }

    //----------------------Destructors----------------------------------------

    /// Drop all edges in the graph
    pub fn drop_all_edges(&mut self) {
        // if there are no edges, no nodes should report any
        // therefore drop the list of connections from each node
        if let Some(node) = self.nodes.iter_mut().by_ref().flatten().next() {
            // SAFETY: We know that the node is not null because it is Some
            let node = unsafe { node.as_mut() };
            node.connections.truncate(0);
        }

        self.edges.drain(..).for_each(|edge_pointer| {
            edge_pointer.map(|node| unsafe { Box::from_raw(node.as_ptr()) });
        });
    }

    /// Drop all nodes in the graph
    pub fn drop_all_nodes(&mut self) {
        // first drop all edges as cannot be
        // any edges in a graph with no nodes
        self.drop_all_edges();
        // let box memory management handle the dropping of the raw pointers
        self.nodes.drain(..).for_each(|node_pointer| {
            node_pointer.map(|node| unsafe { Box::from_raw(node.as_ptr()) });
        });
    }
}

impl<T, U> Drop for SimpleGraph<T, U> {
    fn drop(&mut self) {
        self.drop_all_nodes();
    }
}

impl<'a, T, U> Iterator for IterNodes<'a, T, U> {
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

impl<'a, T, U> Iterator for IterNodesEdge<'a, T, U> {
    type Item = (&'a T, Vec<&'a U>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.0.nodes.by_ref().flatten().next() {
            //if let Some(node) = node_option {
            // SAFETY: We know that the node is not null because it is Some
            let node_ref = unsafe { node.as_ref() };
            let edges: Vec<&U> = node_ref
                .connections
                .iter()
                .filter_map(|edge| edge.map(|edge| unsafe { &(edge.as_ref()).value }))
                .collect();
            return Some((&node_ref.elem, edges));
            //}
        }
        None
    }
}

impl<'a, T, U> Iterator for IterNodesAdj<'a, T, U> {
    type Item = (&'a T, Vec<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.0.nodes.by_ref().flatten().next() {
            //if let Some(node) = node_option {
            // SAFETY: We know that the node is not null because it is Some
            let node_ref = unsafe { node.as_ref() };
            let adj: Vec<&T> = node_ref
                .connections
                .iter()
                .filter_map(|edge| {
                    edge.map(|edge| {
                        // if the start is the current node
                        if unsafe { (edge.as_ref()).start == Some(*node) } {
                            // return the end
                            unsafe { (edge.as_ref()).end }
                                .map(|adj_node| unsafe { &(adj_node.as_ref().elem) })
                        } else {
                            // else return the start
                            unsafe { (edge.as_ref()).start }
                                .map(|adj_node| unsafe { &(adj_node.as_ref().elem) })
                        }
                    })
                })
                .flatten()
                .collect();
            return Some((&node_ref.elem, adj));
            //}
        }
        None
    }
}

impl<'a, T, U> Iterator for IterMutNodes<'a, T, U> {
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

impl<'a, T, U> Iterator for IterMutNodesEdge<'a, T, U> {
    type Item = (&'a mut T, Vec<&'a mut U>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.0.nodes.by_ref().flatten().next() {
            //if let Some(node) = node_option {
            // SAFETY: We know that the node is not null because it is Some
            // This needs to be checked
            let node_ref = unsafe { node.as_mut() };
            let edges: Vec<&mut U> = node_ref
                .connections
                .iter_mut()
                .filter_map(|edge| edge.map(|mut edge| unsafe { &mut (edge.as_mut()).value }))
                .collect();
            return Some((&mut node_ref.elem, edges));
            //}
        }
        None
    }
}

impl<'a, T, U> Iterator for IterMutNodesAdj<'a, T, U> {
    type Item = (&'a mut T, Vec<&'a mut T>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.0.nodes.by_ref().flatten().next() {
            //if let Some(node) = node_option {
            // SAFETY: We know that the node is not null because it is Some
            let node_ref = unsafe { node.as_mut() };
            let adj: Vec<&mut T> = node_ref
                .connections
                .iter_mut()
                .filter_map(|edge| {
                    edge.map(|edge| {
                        // if the start is the current node
                        if unsafe { (edge.as_ref()).start == Some(*node) } {
                            // return the end
                            unsafe { (edge.as_ref()).end }
                                .map(|mut adj_node| unsafe { &mut (adj_node.as_mut().elem) })
                        } else {
                            // else return the start
                            unsafe { (edge.as_ref()).start }
                                .map(|mut adj_node| unsafe { &mut (adj_node.as_mut().elem) })
                        }
                    })
                })
                .flatten()
                .collect();
            return Some((&mut node_ref.elem, adj));
            //}
        }
        None
    }
}

impl<'a, T, U> Iterator for IterEdges<'a, T, U> {
    type Item = (&'a T, &'a T, &'a U);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(edge_ref) = self.edges.by_ref().flatten().next() {
            let edge = unsafe { edge_ref.as_ref() };
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
        if let Some(edge_ref) = self.edges.by_ref().flatten().next() {
            let edge = unsafe { edge_ref.as_mut() };
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

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
/// An error regarding edge creation
pub enum EdgeError {
    /// The node is out of range of the present nodes
    NodeOutOfRange,
    /// The edge is connecting a node to itself
    SameNode,
    /// A multiply connnected pair of nodes
    MultipleConnection,
}

#[cfg(test)]
mod test {
    use crate::EdgeError;

    use super::SimpleGraph;
    #[test]
    fn basics() {
        let points = vec![1, 2, 3];
        let mut graph: SimpleGraph<i32, &str> = SimpleGraph::from_nodes(points);
        // we can add an edge
        graph.add_edge("from 0 to 2", 0, 2).unwrap();
        let e_v = graph.get_edge(2, 0).unwrap();
        println!("Edge can be got back {e_v}");
        // check immutable iterators
        graph.iter_nodes().for_each(|n| println!("{n}"));
        graph
            .iter_edges()
            .for_each(|(value, start, end)| println!("Edge {value} connects {start} and {end}"));
        // check mutable iterators
        graph.iter_mut_nodes().for_each(|n| *n -= 1);
        // check assignment
        graph.iter_nodes().for_each(|n| println!("{n}"));
        // panic!()
        graph.iter_nodes_edges().for_each(|(n, edges)| {
            println!("{n} is connected to");
            edges.iter().for_each(|edge| println!("    {edge}"))
        })
    }

    #[test]
    // Should not be possible to connect a node to itself
    fn edge_to_same() {
        let points = vec![1, 2, 3];
        let mut graph: SimpleGraph<i32, &str> = SimpleGraph::from_nodes(points);
        // we can add an edge
        assert_eq!(
            graph.add_edge("from 0 to itself", 0, 0),
            Err(EdgeError::SameNode)
        );
    }

    #[test]
    // Should not be possible to connect two nodes by more than one edge
    fn two_edges() {
        let points = vec![1, 2, 3];
        let mut graph: SimpleGraph<i32, &str> = SimpleGraph::from_nodes(points);
        // we can add an edge
        graph.add_edge("from 0 to 2", 0, 2).unwrap();
        assert_eq!(
            graph.add_edge("from 0 to 2", 2, 0),
            Err(EdgeError::MultipleConnection)
        );
    }

    #[test]
    // check ability to get an edge between two nodes specified by index
    fn get_edge() {
        let points = vec![1, 2, 3];
        let mut graph: SimpleGraph<i32, &str> = SimpleGraph::from_nodes(points);
        graph.add_edge("from 0 to 2", 0, 2).unwrap();
        assert_eq!(graph.get_edge(1, 0), None);
    }

    #[test]
    // check ability to drop an edge between two nodes specified by index
    fn drop_edge() {
        let points = vec![1, 2, 3];
        let mut graph: SimpleGraph<i32, &str> = SimpleGraph::from_nodes(points);
        graph.add_edge("from 0 to 2", 0, 2).unwrap();
        graph.add_edge("from 0 to 1", 0, 1).unwrap();
        graph.add_edge("from 2 to 1", 2, 1).unwrap();
        let dropped_edge = graph.drop_edge(0, 2);
        assert_eq!(dropped_edge, Some("from 0 to 2"));
        assert_eq!(graph.get_edge(0, 2), None);
    }

    #[test]
    // check ability to drop an node specified by index
    fn drop_node() {
        let points = vec![1, 2, 3, 4, 5];
        let mut k5: SimpleGraph<i32, f64> = SimpleGraph::from_nodes(points);
        k5.add_edge(0.1, 0, 1).unwrap();
        k5.add_edge(0.2, 0, 2).unwrap();
        k5.add_edge(0.3, 0, 3).unwrap();
        k5.add_edge(0.4, 0, 4).unwrap();

        k5.add_edge(1.2, 1, 2).unwrap();
        k5.add_edge(1.3, 1, 3).unwrap();
        k5.add_edge(1.4, 1, 4).unwrap();

        k5.add_edge(2.3, 2, 3).unwrap();
        k5.add_edge(2.4, 2, 4).unwrap();

        k5.add_edge(3.4, 3, 4).unwrap();

        assert_eq!(k5.drop_node(2), Some(3));
        assert_eq!(
            k5.iter_nodes().map(|v| *v).collect::<Vec<i32>>(),
            [1, 2, 4, 5]
        )
    }

    #[test]
    // check ability to drop an node specified by filter
    fn drop_node_filter() {
        let points = vec![1, 2, 3, 4, 5];
        let mut k5: SimpleGraph<i32, f64> = SimpleGraph::from_nodes(points);
        k5.add_edge(0.1, 0, 1).unwrap();
        k5.add_edge(0.2, 0, 2).unwrap();
        k5.add_edge(0.3, 0, 3).unwrap();
        k5.add_edge(0.4, 0, 4).unwrap();

        k5.add_edge(1.2, 1, 2).unwrap();
        k5.add_edge(1.3, 1, 3).unwrap();
        k5.add_edge(1.4, 1, 4).unwrap();

        k5.add_edge(2.3, 2, 3).unwrap();
        k5.add_edge(2.4, 2, 4).unwrap();

        k5.add_edge(3.4, 3, 4).unwrap();

        assert_eq!(k5.drop_nodes_by(|val| val > &3), [4, 5]);
        assert_eq!(k5.iter_nodes().map(|v| *v).collect::<Vec<i32>>(), [1, 2, 3])
    }

    #[test]
    fn iter_adjacent() {
        let points = vec![0, 1, 2, 3, 4];
        let mut graph: SimpleGraph<i32, ()> = SimpleGraph::from_nodes(points);
        graph.add_edge((), 0, 2).unwrap();
        graph.add_edge((), 0, 3).unwrap();
        graph.add_edge((), 4, 2).unwrap();
        graph.add_edge((), 3, 1).unwrap();
        graph.iter_nodes_adj().for_each(|(node, adj)| {
            println!("Node {node} is connected to");
            adj.into_iter().for_each(|a| println!("        {a}"))
        })
    }

    #[test]
    fn iter_adjacent_mut() {
        struct BFSNode {
            city: String,
            roads: Option<i32>,
        }
        let frankfurt = BFSNode {
            city: "Frankfurt".to_string(),
            roads: Some(0),
        };
        let mannheim = BFSNode {
            city: "Mannheim".to_string(),
            roads: None,
        };
        let wurzburg = BFSNode {
            city: "Wurzburg".to_string(),
            roads: None,
        };
        let stuttgart = BFSNode {
            city: "Stuttgart".to_string(),
            roads: None,
        };
        let kassel = BFSNode {
            city: "Kassel".to_string(),
            roads: None,
        };
        let karlsruhe = BFSNode {
            city: "Karlsruhe".to_string(),
            roads: None,
        };
        let erfurt = BFSNode {
            city: "Erfurt".to_string(),
            roads: None,
        };
        let nurnberg = BFSNode {
            city: "Nurnberg".to_string(),
            roads: None,
        };
        let augsburg = BFSNode {
            city: "Augsburg".to_string(),
            roads: None,
        };
        let munchen = BFSNode {
            city: "Munchen".to_string(),
            roads: None,
        };
        let cities = vec![
            frankfurt, mannheim, wurzburg, stuttgart, karlsruhe, erfurt, nurnberg, kassel,
            augsburg, munchen,
        ];
        let mut graph: SimpleGraph<BFSNode, i32> = SimpleGraph::from_nodes(cities);
        graph.add_edge(85, 0, 1).unwrap(); // frankfurt to mannheim
        graph.add_edge(217, 0, 2).unwrap(); // frankfurt to wurzburg
        graph.add_edge(173, 0, 7).unwrap(); // frankfurt to kassel
        graph.add_edge(80, 1, 4).unwrap(); // mannheim to karlsruhe
        graph.add_edge(186, 2, 5).unwrap(); // wurzburg to erfurt
        graph.add_edge(103, 2, 6).unwrap(); // wurzburg to nurnberg
        graph.add_edge(183, 6, 3).unwrap(); // nurnberg to stuttgart
        graph.add_edge(250, 4, 8).unwrap(); //karlsruhe to ausburg
        graph.add_edge(84, 8, 9).unwrap(); // ausberg to munchen
        graph.add_edge(167, 6, 9).unwrap(); // nurnberg to munchen
        graph.add_edge(502, 7, 9).unwrap(); // kassel to munchen

        // do a breadth first search to see the number of roads to each city from Frankfurt
        let mut roads = 0;
        loop {
            let mut changed = 0;
            graph
                .iter_mut_nodes_adj()
                .filter(|(node, _)| node.roads == Some(roads))
                .for_each(|(_, mut adj_nodes)| {
                    adj_nodes
                        .iter_mut()
                        .filter(|a| a.roads.is_none())
                        .for_each(|a| {
                            a.roads = Some(roads + 1);
                            changed += 1;
                        });
                });
            roads += 1;
            if changed == 0 {
                break;
            }
        }
        // graph
        //     .iter_nodes()
        //     .for_each(|node| println!("{} takes {:?} roads", node.city, node.roads));

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Frankfurt")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(0)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Mannheim")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(1)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Wurzburg")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(1)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Kassel")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(1)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Karlsruhe")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(2)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Nurnberg")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(2)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Erfurt")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(2)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Munchen")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(2)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Augsburg")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(3)]
        );

        assert_eq!(
            graph
                .iter_nodes()
                .filter(|node| node.city == "Stuttgart")
                .map(|node| node.roads)
                .collect::<Vec<Option<i32>>>(),
            vec![Some(3)]
        );
    }
}
