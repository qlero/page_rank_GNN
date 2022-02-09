"""
Function implementations to generate single graph examples. 
and plot them
"""


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

from networkx import erdos_renyi_graph, scale_free_graph
from networkx import pagerank, draw_networkx
from networkx import multipartite_layout
from networkx import shell_layout
from networkx import spectral_layout
from networkx import spiral_layout
from networkx import spring_layout

def generate_erdos_graph(
        n: int, 
        p: float
) -> nx.classes.graph.Graph:
    """
    Generates an Erdos' graph.
    
    Parameters
    ----------
    n : integer
        Number of nodes in each graph
    p : float, optional
        Probability for edge creation
        
    Returns 
    -------
    g : nx.classes.graph.Graph
        Generated graph
    """
    g      = erdos_renyi_graph(n, p)
    g.name = f"Erdos Graph with parameters n={n}, p={p}"
    return g
    
def generate_scale_free_graph(
        n: int
) -> nx.classes.graph.Graph:
    """
    Generates an Scale-Free graph.
    
    Parameters
    ----------
    n : integer
        Number of nodes in each graph
        
    Returns 
    -------
    g : nx.classes.graph.Graph
        Generated graph
    """
    g      = scale_free_graph(n)
    g.name = f"Scale-Free Graph with parameter n={n}"
    return g
    
def plot_graph(
    graph: nx.classes.graph.Graph,
    layout: str = "spring_layout"
) -> None:
    """
    Plots a graph nodes and edges with PageRank labels.
    
    Parameters
    ----------
    graph : networkx.classes.graph.Graph
        simple graph instance generated via the networkx
        library
    layout : string, optional
        defines which node position layout is chosen.    
    """
    # Computes the type of layout to be used for plotting
    if layout == "shell":
        pos      = shell_layout(graph)
        subtitle = "concentric circles"
    elif layout == "spectral":
        pos      = spectral_layout(graph)
        subtitle = "eigenvectors of the graph Laplacian"
    elif layout == "spiral":
        pos      = spiral_layout(graph)
        subtitle = "a spiral layout"    
    else:
        pos      = spring_layout(graph)
        subtitle = "Fruchterman-Reingold force-directed algo."
    # Computes custom labels with PageRank values
    labels = {key: f"{key}\n\nPR: {round(rank, 2)}"
              for key, rank in pagerank(graph).items()}
    # Plots the graph
    plt.figure(figsize=(8,8))
    plt.title(f"Plot of {graph.name} with PageRank values\n" + 
              f"Node position computed with {subtitle}")
    draw_networkx(
        graph,
        node_color = list(pagerank(graph).values()),
        pos        = pos,
        labels     = labels,
        font_size  = 10,
        cmap       = cm.YlGn
    )
    plt.show()