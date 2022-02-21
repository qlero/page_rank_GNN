# Content

This repository covers the implementation of a regression task using Graph Neural Network implementations (GNN with relaxation, GNN with Convolutional Layer, GNN with SAGE Layer) to learn the PageRank algorithm. 

**Sources**:
- GNN with relaxation: [The Graph Neural Network Model by Scarselli F., Gori M. et al.](https://persagen.com/files/misc/scarselli2009graph.pdf)
- GCN: [Semi-Supervised Classification with Graph Convolutional Networks, Kipf T.N., Welling M.](https://arxiv.org/abs/1609.02907)
- SAGE: [Inductive Representation Learning on Large Graphs by Hamilton W.L., et al.](https://arxiv.org/pdf/1706.02216.pdf)

## Folder structure

```
.
├── assets
│   └── note:	images used in page_rank_gnn.ipynb
├── graph_files 
│   └── note: 	save files for .graphml graphs
├── presentation
│   └── note:	open index.html in browser for slides
├── scripts
│   ├── dataset.py
│   ├── functions.py
│   ├── graph_networks.py
│   ├── page_rank.py
│   ├── random_graph_generator.py
│   └── regression_gnn_wrapper.py
├── README.md
└── page_rank_gnn.ipynb
```

### dataset.py

Contains classes and functions to construct and concatenate `networkx` and `dgl` graphs.

***Classes***: PageRankDataset

***Functions***: concatenate_graph_dataset, generate_graph_dataset

### functions.py

Contains useful functions for single graph generation, importing and setting up the `torch_gnn` library, plotting `networkx` graph

***Functions:*** `generate_erdos_graph`, `generate_scale_free_graph`, `override_torch_gnn_library`, `plot_graph`

***Note:*** the `override_torch_gnn_library` function git clone the `torch_gnn` library and performs string replacements for the contained methods to be imported as-is in a jupyter notebook.

### graph_networks.py

Contains classes and functions to build and use the GCN and SAGE GNN.

***Classes:*** `PageRankModelingWithGNN`, `GCN`

***Functions:*** `evaluate`

### page_rank.py

Contains a simple implementation of the PageRank algorithm (adapted from the Wikipedia page)

***Functions:*** `page_rank`

### random_graph_generator.py

Contains the class to generate directed random graphs.

***Classes:*** `Random_Graph_Generator`

### regression_gnn_wrapper.py

Contains the implementation of a wrapper in the style of `torch_gnn` for a regression task with a GNN with relaxation.

***Classes:*** `GNNWrapperLight`, `RegressionGNNWrapper`, `PageRankModelingWithRelaxationGNN`

### Notes

***Possible tweak:*** standardizing the function and class naming convention from camelCase to snake_case.
