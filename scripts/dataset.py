"""
Implementation of a graph dataset generator function.
"""

import itertools as it

from .random_graph_generator import Random_Graph_Generator

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
    tuple
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
