"""
Implementation of a graph dataset generator function.
"""

###########
# IMPORTS #
###########

import dgl
import itertools as it
import networkx as nx
import numpy as np
import torch

from .random_graph_generator import Random_Graph_Generator
from dgl.data import DGLDataset

#############
# FUNCTIONS #
#############

def concatenate_graph_dataset(
    X: list,
    y: list,
    path_graph: str = "graph_files/graph.graphml",
    path_pagerank: str = "graph_files/graph_PageRanks.txt"
) -> tuple:
    """
    Concatenate a selection of networkx graphs into a single
    large graph to be used as singular input to a GNN.

    Parameters
    ----------
    X : list of nx.classes.graph.DiGraph
        List of single, generated graphs
    y : list of floats
        list of dictionary corresponding to the PageRanks 
        of each node
    path_graph : string
        Path where to save the concatenation of graphs as
        a .graphml file
    path_pagerank : string
        Path where to save the list of graph node PageRank
        values
        
    Returns
    -------
    Tuple (nx.classes.graph.DiGraph, list)
        Single networkx graph containing each generated note
        with their PageRank as label and feature, and a
        separate list of the PageRank values
    """
    # Initializes the output variables
    graphs    = X[0]
    pageranks = list(y[0].values())
    # Aggregates the generated graphs and PageRank values
    for idx in range(1, len(X)):
        graphs     = nx.disjoint_union(graphs, X[idx])
        pageranks += list(y[idx].values())
    # Saves the graphs and PageRank
    nx.write_graphml(graphs, path_graph)
    with open(path_pagerank, 'w') as file:
        for row in pageranks:
            file.write(str(row)+'\n')
    return graphs, pageranks

def generate_graph_dataset(
        probability_range: list, 
        node_range: list,
        n_erdos: int,
        n_scale_free: int
) -> tuple:
    """
    Generate a dataset of Erdos and Scale-Free graphs.

    Parameters
    ----------
    probability_range : list of floats
        Range of probabilities to generate Erdos graphs
    node_range : list of integers
        Range of nodes to generate graphs
    n_erdos : integer
        Number of erdos graphs to generate per case
    n_scale_free : integer
        Number of scale-free graphs to generate per case
        
    Returns
    -------
    Tuple of lists
        Tuple of graphs X and corresponding PageRank value y
    """
    X, y = [], []
    # Generates the erdos graphs
    print("Generating Erdos graphs with parameters:")
    for p, n in it.product(probability_range, node_range):
        generator = Random_Graph_Generator("erdos")
        generator.graphs_generate(n_erdos, n, p)
        generator.graphs_page_rank_compute()
        genX, geny = generator.graphs_retrieve()
        X += genX
        y += geny
        print(f"\tCompleted -- n={n}, p={p}")

    # Generates the scale-free graphs
    print("\nGenerating Scale-Free graphs with parameter:")
    for n in node_range:
        generator = Random_Graph_Generator("scale-free")
        generator.graphs_generate(n_scale_free, n)
        generator.graphs_page_rank_compute()
        genX, geny = generator.graphs_retrieve()
        X += genX
        y += geny
        print(f"\tCompleted -- n={n}")
    return X, y

def load_concatenated_graph_dataset(
    path_graph: str = "graph_files/graph.graphml",
    path_pagerank: str = "graph_files/graph_PageRanks.txt"
) -> tuple:
    """
    Loads the concatenation of networkx graphs saved as a single
    large graph to be used as singular input to a GNN.

    Parameters
    ----------
    path_graph : string
        Path where the save of the concatenation of graphs as
        a .graphml file is located
    path_pagerank : string
        Path where where the save of the list of graph node PageRank
        values is located
        
    Returns
    -------
    Tuple (nx.classes.graph.DiGraph, list)
        Single networkx graph containing each generated note
        with their PageRank as label and feature, and a
        separate list of the PageRank values
    """
    graphs = nx.read_graphml(path_graph)
    pageranks = []
    with open(path_pagerank, 'r') as file:
        for row in file:
            pageranks.append(float(row))
    return graphs, pageranks

###########
# CLASSES #
###########

class PageRankDataset(DGLDataset):
    """
    Implementation of a Custom DGL dataset
    """
    def __init__(
            self, 
            graph: nx.classes.graph.Graph, 
            pageranks: list, 
            train_size: float = 0.7, 
            val_size: float = 0.15,
            use_noise: bool = True
    ) -> None:
        """
        Parameters
        ----------
        graph : nx.classes.graph.DiGraph
            Single networkx graph
        pageranks : list of floats
            list of PageRank values of each node of the graph
        train_size : float
            training size in %
        val_size : float
            validation size in %
        noise : bool
            Indicates whether to use a 1d noise or a torch.eye
            matrix as feature
        """
        # Process the input graph as a DGL graph attributes
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        n_nodes = graph.number_of_nodes()
        self.graph = dgl.from_networkx(graph)
        # Records the pagerank values as both features and labels
        pageranks =  torch.from_numpy(np.array(pageranks)).float()
        if not use_noise:
            self.graph.ndata['feat'] = torch.eye(len(pageranks))
        else:
            noise = torch.from_numpy(np.random.rand(len(pageranks), 1)).float()
            self.graph.ndata['feat'] = noise
        self.graph.ndata["label"] = pageranks
        # Saves the training, and validation sizes
        self.train_size = train_size
        self.val_size = val_size
        # Names the dataset
        super().__init__(name='page_rank_graph')
        
    def process(self):
        """
        Preprocesses the dataset as part of the PageRankDataset
        class.
        """
        ### Computes the Train/Val/Test Split
        # Counts the number of nodes and creates a randomized
        # split between the three sets
        indexes       = list(range(self.graph.number_of_nodes()))
        n_nodes       = len(indexes)
        n_train       = int(n_nodes * self.train_size)
        n_val         = int(n_nodes * self.val_size)
        np.random.shuffle(indexes)
        train_indexes = indexes[:n_train]
        val_indexes   = indexes[n_train:n_train + n_val]
        test_index    = indexes[n_train + n_val:]
        # Creates a Boolean mask
        train_mask    = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask      = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask     = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_indexes] = True
        val_mask[val_indexes] = True
        test_mask[test_index] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask']   = val_mask
        self.graph.ndata['test_mask']  = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1