# Content

This repository covers the implementation of a regression task using Graph Neural Network implementations (GNN with relaxation, GNN with Convolutional Layer, GNN with SAGE Layer) to learn the PageRank algorithm. 

**Sources**:
- GNN with relaxation: [The Graph Neural Network Model](https://persagen.com/files/misc/scarselli2009graph.pdf)
- GCN: [Semi-Supervised Classification with Graph Convolutional Networks, T. N. Kipf, M. Welling](https://arxiv.org/abs/1609.02907)
- SAGE: [Semi-Supervised Classification with Graph Convolutional Networks by T. N. Kipf, M. Welling] (https://arxiv.org/pdf/1706.02216.pdf)

### Folder structure

.
├── assets
│   └── *images used in `page_rank_gnn.ipynb`*
├── graph_files 
│   └── *save files for .graphml graphs*
├── presentation
│   └── *open `index.html` in browser for slides*
├── scripts
│   ├── dataset.py
│   ├── functions.py
│   ├── graph_networks.py
│   ├── page_rank.py
│   ├── random_graph_generator.py
│   └── regression_gnn_wrapper.py
├── README.md
└── page_rank_gnn.ipynb

#### dataset.py

Contains classes and functions to construct and concatenate `networkx` and `dgl` graphs.

<u>Classes:</u> PageRankDataset

<u>Functions:</u> concatenate_graph_dataset, generate_graph_dataset

#### functions.py

Contains useful functions for single graph generation, importing and setting up the `torch_gnn` library, plotting `networkx` graph

<u>Functions:</u> `generate_erdos_graph`, `generate_scale_free_graph`, `override_torch_gnn_library`, `plot_graph`

<u>Note:</u> the `override_torch_gnn_library` function git clone the `torch_gnn` library and performs string replacements for the contained methods to be imported as-is in a jupyter notebook.

#### graph_networks.py

Contains classes and functions to build and use the GCN and SAGE GNN.

<u>Classes:</u> `PageRankModelingWithGNN`, `GCN`

<u>Functions:</u> `evaluate`

#### page_rank.py

Contains a simple implementation of the PageRank algorithm (adapted from the Wikipedia page)

<u>Functions:</u> `page_rank`

#### random_graph_generator.py

Contains the class to generate directed random graphs.

<u>Classes:</u> `Random_Graph_Generator`

#### regression_gnn_wrapper.py

Contains the implementation of a wrapper in the style of `torch_gnn` for a regression task with a GNN with relaxation.

<u>Classes:</u> `GNNWrapperLight`, `RegressionGNNWrapper`, `PageRankModelingWithRelaxationGNN`

### Notes

<u>Possible tweak:</u> standardizing the function and class naming convention from camelCase to snake_case.
