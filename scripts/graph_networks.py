"""
Implementations of Graph Neural Networks
"""

###########
# IMPORTS #
###########

import dgl
import dgl.nn as dglnn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import plot_graph
from matplotlib import cm

####################
# GLOBAL VARIABLES #
####################

global SAGE_description 
global GraphConv_description

SAGE_description = """
The model was declared with GraphSAGE layers (2) which were introduced in:
> Inductive Representation Learning on Large Graphs
> by W.L. Hamiton, R. Ying, and J. Leskovec
> https://arxiv.org/pdf/1706.02216.pdf

The GraphSAGE Algorithm trains weight matrices instead of relying on 
embedding tables. It allows adding and removing nodes directly via the 
weight matrix. Nodes embeddings do not need to be recomputed.

```
\begin{align}
    h_{\mathcal{N}(i)}^{(l+1)} &= 
        \text{aggregate}
        \left(
            \{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}
        \right) h_{i}^{(l+1)}
    \\
    &= \sigma 
        \left(
            W \cdot \mathrm{concat} (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) 
        \right) h_{i}^{(l+1)} \\
    &= \mathrm{norm}(h_{i}^{l})
\end{align}
```

See also: https://docs.dgl.ai/api/python/nn.pytorch.html#sageconv
"""

GraphConv_description = """
The model was declared with GraphConv layers (2) which were introduced in:
> Semi-Supervised Classification with Graph Convolutional Networks
> by T. N. Kipf, M. Welling
> https://arxiv.org/pdf/1706.02216.pdf

Graph convolution is defined as follows:

```
h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})
```

Where:
    - `\mathcal{N}(i)` is the set of neighbors of node `i`
    - `c_{ji}` is the product of the square root of node degrees
       `c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`)
    - `\sigma` is an activation function.

See also: https://docs.dgl.ai/api/python/nn.pytorch.html#graphconv

"""

#############
# FUNCTIONS #
#############

def evaluate(
        model: nn.Module, 
        graph: dgl.graph, 
        features: torch.Tensor, 
        labels: torch.Tensor, 
        mask: torch.Tensor
) -> tuple:
    """
    Evaluator function for a Pytorch model.
    
    Parameters
    ----------
    model : nn.Module 
        Torch nn module (e.g. custom SAGE class
        see below)
    graph : dgl.heterograph.DGLHeteroGraph 
        networkx graph formatted as a heterograph
        from the DGL library
    features : torch.Tensor
        Torch tensor of features (Page Ranks)
    labels : torch.Tensor
        Torch tensor of features (Page Ranks)
    mask : torch.Tensor
        Boolean mask for the validation or test
        sets
    
    Returns
    -------
    Tuples of:
        - val/test set predicted PageRank values
        - val/test set true PageRank values
        - Average MSE loss
    """
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask].reshape(-1, 1)
        correct = F.mse_loss(logits, labels)
    return logits, labels, correct.item() / len(labels)    

###########
# CLASSES #
###########

class PageRankModelingWithGNN():
    """
    Implementation of a GNN training and evaluation run.
    """
    def __init__(
            self, 
            graph: dgl.graph, 
            hidden_feature_size: int, 
            n_epochs: int, 
            model: str = "SAGE",
            optimizer: str = "Adam",
            print_description: bool = False
    ) -> None:
        """
        Initializes the PageRank Modeling with GNN.
        
        Parameters
        ----------
        graph : dgl.heterograph.DGLHeteroGraph
            networkx graph formatted as a heterograph
            from the DGL library
        hidden_feature_size : integer
            number of hidden features to be used by 
            the underlying model
        n_epochs : integer
            number of training epochs
        model : str, optional
            underlying model name
        print_description : boolean, optional
            Prints the description of the used GNN
        """
        # Declares the modelization run's attributes
        self.raw_graph         = graph
        # graph = dgl.add_self_loop(graph) # avoids lone loops
        self.graph             = graph
        self.node_features     = graph.ndata['feat']
        self.node_labels       = graph.ndata['label']
        self.train_mask        = graph.ndata['train_mask']
        self.valid_mask        = graph.ndata['val_mask']
        self.test_mask         = graph.ndata['test_mask']
        self.n_hidden_features = hidden_feature_size
        self.n_features        = graph.ndata['feat'].shape[1]
        self.n_labels          = 1
        self.n_epochs          = n_epochs
        self.loss_per_epoch    = []
        self.model_type        = model
        self.true_labels       = graph.ndata['label'][self.test_mask]
        self.predictions       = None
        self.p                 = None
        # Declares the internal model
        if model == "SAGE":
            if print_description: print(SAGE_description)
            self.model = SAGE(
                in_feats  = self.n_features, 
                hid_feats = self.n_hidden_features, 
                out_feats = self.n_labels
        )
        else: 
            if print_description: print(GraphConv_description)
            self.model = GCN(
                in_feats  = self.n_features, 
                hid_feats = self.n_hidden_features, 
                out_feats = self.n_labels
        )
        # Declares the optimizer
        if optimizer == "Adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif optimizer == "AdamW":
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        elif optimizer == "Adamax":
            self.opt = torch.optim.Adamax(self.model.parameters(), lr=1e-3)
        else:
            self.opt = torch.optim.Adadelta(self.model.parameters(), lr=1e-3)
    
    def run(
            self, 
            print_node_results: bool = False,
            print_graph_results: bool = False
    ) -> None:
        """
        Performs the training sequence for the underlying GNN model
        
        Parameters
        ----------
        print_node_results : boolean, optional
            Indicates whether to print a table comparing predictions
            with true PageRanks
        print_graph_results : boolean, optional
            Indicates whether to print two graphs, one with the true
            PageRank values and one with the Estimated ones
        """
        ################
        # TRAINING RUN #
        ################
        old_val_loss = float("inf")
        counter = 0
        for epoch in range(self.n_epochs):
            self.model.train()
            # Forward propagation on all nodes
            logits       = self.model(self.graph, self.node_features)
            train_logits = logits[self.train_mask]
            train_labels = self.node_labels[self.train_mask].reshape(-1, 1)
            self.p = logits
            # Computes the epoch loss
            loss = F.mse_loss(train_logits, train_labels)
            # backward propagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # records the loss
            self.loss_per_epoch.append(loss.item()) 
            # Accuracy
            _, _, val_loss = evaluate(
                self.model,
                self.graph,
                self.node_features,
                self.node_labels,
                self.valid_mask
            )
            if val_loss < old_val_loss:
                old_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter == 10:
                    print("#### VALIDATION LOSS NO IMPROVEMENT IN 10 EPOCHS ####")
                    print(f"#### EARLY STOPPING, epoch: {epoch}             ####")
                    break
            train_loss = round(loss.item(), 6)
            val_loss = round(val_loss, 6)
            if epoch%50==0: 
                print(f"Epoch {epoch},\tloss: {train_loss};\t\tval loss {val_loss}")
        ###############
        # TESTING RUN #
        ###############
        predictions, _, test_loss = evaluate(
            self.model,
            self.graph,
            self.node_features,
            self.node_labels,
            self.test_mask
        )
        self.predictions = predictions
        test_loss = round(test_loss, 6)
        print(f"Test loss: {test_loss}")
        ######################
        # VISUALIZATION STEP #
        ######################
        # Displays the loss convergence
        plt.figure(figsize=(8,4))
        plt.plot(self.loss_per_epoch)
        plt.title(f"Loss convergence of {self.model_type} model") 
        plt.show()
        # if indicated, displays the difference between preds and truth
        preds = [x[0].tolist() if x[0]>0 else 0 for x in self.p.detach().numpy()]
        truth = [x.item() if x.item()>0 else 0 for x in list(self.node_labels)]
        if print_node_results:
            print("Node\t\tPredictions\tTrue PageRank\tDiff")
            print("Note: negative values set to 0")
            print("-"*60)
            for node, (p, t) in enumerate(zip(preds, truth)):
                if round(np.abs(p-t), 6) > 0.025: 
                    tag = " <- |true-pred| > 0.025 "
                else: 
                    tag = ""
                print(f"Node {node}: \t", 
                      round(p, 6), "\t", 
                      round(t, 6), "\t", 
                      str(round(np.abs(p-t), 6))+tag)
        # if indicated, displays the true graph and the estimated graph
        if print_graph_results:
            graph_truth = self.raw_graph.clone().to_networkx()
            truth = {k:truth[k] for k in range(len(truth))}
            nx.set_node_attributes(graph_truth, truth, "PageRank")
            graph_preds = self.raw_graph.clone().to_networkx()
            preds = {k:preds[k] for k in range(len(preds))}
            nx.set_node_attributes(graph_preds, preds, "PageRank")
            true_labels = {k:f"{k}\n\nPR: {round(truth[k], 3)}" 
                             for k in range(len(truth))}
            pred_labels = {k:f"{k}\n\nPR: {round(preds[k], 3)}" 
                             for k in range(len(preds))}
            print("\n\nPlot with true labels")
            print("---------------------")
            colors = [graph_truth.nodes()[x]["PageRank"] 
                      for x in graph_truth.nodes()]
            plt.figure(figsize=(9,9))
            plt.title("Plot of True PageRank values")
            nx.draw_networkx(
                graph_truth,
                labels=true_labels,
                pos=nx.spring_layout(graph_truth),
                node_color = colors,
                font_size  = 10,
                cmap       = cm.YlGn)
            plt.show()
            print("Plot with predicted labels")
            print("--------------------------")
            colors = [graph_preds.nodes()[x]["PageRank"] 
                      for x in graph_preds.nodes()]
            plt.figure(figsize=(9,9))
            plt.title("Plot of Predicted PageRank values\n" + \
                      "Note: negative predictions set to 0")
            nx.draw_networkx(
                graph_preds,
                labels=pred_labels,
                pos=nx.spring_layout(graph_preds),
                node_color = colors,
                font_size  = 10,
                cmap       = cm.YlGn) 
            
    def test(
            self, 
            new_graph: dgl.graph
    ) -> None:
        """
        Predicts the PageRank values of the node of a
        graph that was not seen by the model.
        
        Parameters
        ----------
        new_graph : dgl.heterograph.DGLHeteroGraph 
            networkx graph formatted as a heterograph
            from the DGL library
            
        Returns
        -------
        preds_PageRanks : list
            List of PageRank predictions
        true_PageRanks : list
            List of PageRank true values
        """
        # Predicts and computes the loss
        new_graph = dgl.add_self_loop(new_graph) # avoids lone loops
        preds_PageRanks, true_PageRanks, loss = evaluate(
            self.model,
            new_graph,
            new_graph.ndata['feat'].float(),
            new_graph.ndata['label'].float(),
            new_graph.ndata['test_mask']
        )
        print(f"Average test loss: {loss}")
        # Displays the distribution of errors and mean square error
        error_diff = []
        mse_diff = []
        for idx, _ in enumerate(preds_PageRanks.squeeze()):
            pred = preds_PageRanks.squeeze()[idx].item()
            true = true_PageRanks[idx].item()
            error_diff.append(pred-true)
            mse_diff.append((pred-true)**2)
        plt.figure(figsize=(10, 5))
        sns.kdeplot(error_diff)
        plt.title("Distribution of Error between PageRank " + \
                  f"prediction and true value with {self.model_type} model")
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.hist(mse_diff, bins=50)
        plt.title("Distribution of Mean-Squared Error between PageRank " + \
                  f"prediction and true value with {self.model_type} model")
        plt.show()
        return preds_PageRanks, true_PageRanks

class SAGE(nn.Module):
    """
    Implementation of a two-layer SAGE Convolutional GNN
    """
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        # h = torch.tanh(h)
        return h
    
class GCN(nn.Module):
    """
    Implementation of a two-layer Convolutional GNN
    """
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid_feats, norm="right", allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid_feats, out_feats, norm="right", allow_zero_in_degree=True)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        # h = torch.tanh(h)
        return h