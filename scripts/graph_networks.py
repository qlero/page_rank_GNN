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
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

####################
# GLOBAL VARIABLES #
####################

global SAGE_description 

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
            model: str = "SAGE"
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
        """
        # Declares the modelization run's attributes
        graph = dgl.add_self_loop(graph) # avoids lone loops
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
            print(SAGE_description)
            self.model = SAGE(
                in_feats  = self.n_features, 
                hid_feats = self.n_hidden_features, 
                out_feats = self.n_labels
        )
        else: ###########################
            self.model = GCN(
                in_feats  = self.n_features, 
                hid_feats = self.n_hidden_features, 
                out_feats = self.n_labels
        )
        # Declares the optimizer
        self.opt = torch.optim.Adam(self.model.parameters()) #, lr=1e-6)
    
    def train(self):
        """
        Performs the training sequence for the underlying GNN model
        """
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
            if epoch%50==0: 
                # Accuracy
                _, _, val_loss = evaluate(
                    self.model,
                    self.graph,
                    self.node_features,
                    self.node_labels,
                    self.valid_mask
                )
                train_loss = round(train_loss.item(), 5)
                val_loss = round(val_loss, 5)
                print(f"Epoch {epoch}, loss: {train_loss}; val loss {val_loss}")
            if epoch == self.n_epochs-1:
                predictions, _, test_loss = evaluate(
                    self.model,
                    self.graph,
                    self.node_features,
                    self.node_labels,
                    self.test_mask
                )
                self.predictions = predictions
                test_loss = round(test_loss, 5)
                print(f"Epoch {epoch}, test loss: {test_loss}")
        # Displays the loss convergence
        plt.figure(figsize=(8,4))
        plt.plot(self.loss_per_epoch)
        plt.title(f"Loss convergence of {self.model_type} model") 
        plt.show()
            
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
    Implementation of a two-layer Convolutional GNN
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
        h = F.tanh(h)
        return h
    
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid_feats)
        self.conv2 = dglnn.GraphConv(hid_feats, out_feats)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h