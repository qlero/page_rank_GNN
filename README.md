# Content

This repository covers the implementation of a regression task using Graph Neural Network implementations (GNN with relaxation, GNN with Convolutional Layer, GNN with SAGE Layer) to learn the PageRank algorithm. 

**Main Sources**:
- GNN with relaxation: [Scarselli, F., Gori, M., et al.: The Graph Neural Network Model (2009)](https://persagen.com/files/misc/scarselli2009graph.pdf)
- GCN: [Kipf, T.N., Welling, M.: Semi-Supervised Classification with Graph Convolutional Networks (2017)](https://arxiv.org/abs/1609.02907)
- SAGE: [Hamilton W.L., et al.: Inductive Representation Learning on Large Graphs (2017)](https://arxiv.org/pdf/1706.02216.pdf)
- `Networkx`: [NetworkX Developers, Aric Hagberg, Dan Schult, Pieter Swart](https://github.com/networkx/networkx)

- `torch_gnn`: [Tiezzi, M., Marra, G., et al.: A Lagrangian Approach to Information Propagation in Graph Neural Networks. ECAI2020. (2020)](https://github.com/mtiezzi/torch_gnn)
- `dgl`: [Wang, M., el al.: Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks. (2019)](https://github.com/qlero/page_rank_GNN)

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
