"""
Function implementations to generate single graph examples. 
and plot them
"""

###########
# IMPORTS #
###########

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import os
import shutil
import sys

from networkx import erdos_renyi_graph, scale_free_graph
from networkx import pagerank, draw_networkx
from networkx import multipartite_layout
from networkx import shell_layout
from networkx import spectral_layout
from networkx import spiral_layout
from networkx import spring_layout

#############
# FUNCTIONS #
#############

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
    g      = erdos_renyi_graph(n, p, directed=True)
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
    
def override_torch_gnn_library(
        skip_download: bool = True
):
    """
    git clones and overrides the library torch_gnn
    (https://github.com/mtiezzi/torch_gnn.git) with
    a new script for regression GNN    
    """
    path     = "./torch_gnn/"
    reg_file = "./scripts/regression_gnn_wrapper.py"
    lib_path = "https://github.com/mtiezzi/torch_gnn.git"
    if not skip_download:
        # Deletes a potentially existing archive
        if os.path.exists(path):
            shutil.rmtree(path)
        # Downloads the archive
        os.system(f"git clone {lib_path}")
        # Finds the python script in the cloned archive
        # Retrieves their name
        files = list(filter(lambda x: x[-2:] == "py", os.listdir(path)))
        names = list(map(lambda x: x[:-3], files))
        # Overrides the content of the script by updating
        # the import dependencies path and statements
        for file in files:
            with open(f"{path}{file}", "r") as f:
                data = f.read()
                for name in names:
                    data = data.replace(
                        f"import {name}", 
                        f"from . import {name}"
                    )
                    data = data.replace(
                        f"from {name}", 
                        f"from .{name}"
                    )
                    data = data.replace(
                        f"from . import networkx", 
                        "import networkx"
                    )
            with open(f"{path}{file}", "w") as f:
                f.write(data)
    # Copies regression file in torch_gnn root
    shutil.copyfile(reg_file, path+"regression_gnn_wrapper.py")
    
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
    plt.figure(figsize=(15,8))
    plt.title(f"Plot of PageRank values\n" + 
              f"Node position computed with {subtitle}")
    draw_networkx(
        graph,
        node_color = list(pagerank(graph).values()),
        pos        = pos,
        labels     = labels,
        font_size  = 10,
        cmap       = cm.YlGn,
        arrows     = True
    )
    plt.show()