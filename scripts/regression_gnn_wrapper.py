"""
Implementation of a Graph Neural Network with Relaxation.
"""

###########
# IMPORTS #
###########

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import Accuracy, prepare_device
from .pygnn import GNN
from .dataloader import from_EN_to_GNN
from matplotlib import cm

####################
# GLOBAL VARIABLES #
####################

global RELAX_description 

RELAX_description = """
The model is a GNN based on a diffusion mechanism and relaxation.
A graph is processed by a set of nodes linked according to the
graph adjacency matrix/list of edges. The model updates the nodes' 
states by using an algorithm that shares information between 
(adjacent) nodes until an equilibrium state is reached. 

The output of the model is computed locally at each node based
on the node state.

The diffusion mechanism is constrained to ensure that a unique stable
equilibrium exists.

See also: https://persagen.com/files/misc/scarselli2009graph.pdf
"""

###########
# CLASSES #
###########

class GNNWrapperLight:
    class Config:
        def __init__(self):
            # Setup parameters
            self.device       = None
            self.use_cuda     = None
            self.dataset_path = None
            self.log_interval = None
            self.task_type    = "semisupervised"

            # GNN hyperparameters
            self.lrw    = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations        = None
            self.n_nodes     = None
            self.state_dim   = None
            self.label_dim   = None
            self.output_dim  = None
            self.graph_based = False
            self.activation  = nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims  = None

    def __init__(self, config: Config):
        # Initializes GNN
        self.config       = config
        self.optimizer    = None
        self.criterion    = None
        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None

    def __call__(self, dset, state_net=None, out_net=None):
        # handles the dataset info
        self._data_loader(dset)
        self.gnn = GNN(
            self.config, 
            state_net, 
            out_net
        ).to(self.config.device)
        # Calls object functions (see below)
        self._criterion() 
        self._optimizer()
        self._accuracy()

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type=self.config.task_type)
        self.ValidAccuracy = Accuracy(type=self.config.task_type)
        self.TestAccuracy  = Accuracy(type=self.config.task_type)

    def _criterion(self):
        self.criterion = nn.MSELoss()
        
    def _data_loader(self, dset):  
        # handles dataset data and metadata
        self.dset              = dset.to(self.config.device)
        self.config.label_dim  = self.dset.node_label_dim
        self.config.n_nodes    = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _optimizer(self, opti = None):
        if opti == "AdamW":
            func = optim.AdamW
        elif opti == "Adamax":
            func = optim.Adamax
        elif opti == "Adadelta":
            func = optim.Adadelta
        elif opti == "SGD":
            func = optim.SGD
        else:
            func = optim.Adam
        self.optimizer = func(
            self.gnn.parameters(), 
            lr=self.config.lrw
        )

    def predict(self, edges, agg_matrix, node_labels):
        # To implement
        pass
        
    def train_step(self, epoch):
        # To implement
        pass

    def test_step(self, epoch):
        # To implement
        pass
    
class RegressionGNNWrapper(GNNWrapperLight):
    class Config:
        def __init__(self):
            # Setup parameters
            self.device       = None
            self.use_cuda     = None
            self.dataset_path = None
            self.log_interval = None
            self.task_type    = "semisupervised"

            # GNN hyperparameters
            self.lrw    = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations        = None
            self.n_nodes     = None
            self.state_dim   = None
            self.label_dim   = None
            self.output_dim  = None
            self.graph_based = False
            self.activation  = nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims  = None

    def __init__(self, config: Config):
        super().__init__(config)

    def _data_loader(self, dset):  
        # Handles dataset data and metadata
        self.dset              = dset.to(self.config.device)
        self.config.label_dim  = self.dset.node_label_dim
        self.config.n_nodes    = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type="semisupervised")
        self.ValidAccuracy = Accuracy(type="semisupervised")
        self.TestAccuracy  = Accuracy(type="semisupervised")
        
    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)
    
    def train_step(self, epoch):
        # Sets GNN to training mode
        self.gnn.train()
        # Retrieves training data
        data = self.dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        if self.config.graph_based:
            output, iterations = self.gnn(
                data.edges, 
                data.agg_matrix, 
                data.node_labels, 
                graph_agg=data.graph_node
            )
        else:
            output, iterations = self.gnn(
                data.edges, 
                data.agg_matrix, 
                data.node_labels
            )
        # Performs backpropagation
        loss = self.criterion(
            output[data.idx_train], 
            data.targets[data.idx_train].reshape(-1, 1).float()
        )
        loss.backward()
        # Updates weights and prints status
        self.optimizer.step()
        with torch.no_grad():  # Accuracy computation
            self.TrainAccuracy.update(
                output, 
                data.targets,
                idx = data.idx_train
            )
            if epoch % self.config.log_interval == 0:
                print(f"Epoch -- {epoch}")
                msg  = f"\tTraining -- Mean Loss: {str(np.round(loss.detach().numpy(), 4))}"
                msg += f", Iterations: {iterations}"
                print(msg)
        return output    

    def test_step(self, epoch, step="val"):
        # Sets GNN to training mode
        self.gnn.eval()
        # Retrieves training data
        data = self.dset
        self.optimizer.zero_grad()
        if step == "val":
            self.ValidAccuracy.reset()
        else:
            self.TestAccuracy.reset()
        # output computation
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(
                    data.edges, 
                    data.agg_matrix, 
                    data.node_labels, 
                    graph_agg=data.graph_node
                )
            else:
                output, iterations = self.gnn(
                    data.edges, 
                    data.agg_matrix, 
                    data.node_labels
                )
            # Performs backpropagation
            if step == "val":
                idx = data.idx_valid
            else:
                idx = data.idx_test
            loss = self.criterion(
                output[idx], 
                data.targets[idx].reshape(-1, 1).float()
            )
            if step == "val":
                self.ValidAccuracy.update(
                    output, 
                    data.targets,
                    idx = idx
                )
            else:
                self.TestAccuracy.update(
                    output, 
                    data.targets,
                    idx = idx
                )
            if epoch % self.config.log_interval == 0:
                msg  = "\tValidation --" if step=="val" else "\tTest --"
                msg += f" Mean Loss: {str(np.round(loss.detach().numpy(), 4))}"
                msg += f", Iterations: {iterations}"
                print(msg)
        return output

class PageRankModelingWithRelaxationGNN():
    """
    Implementation of a GNN training and evaluation run.
    """
    def __init__(
            self,
            nodes: np.ndarray,
            edges: np.ndarray,
            labels: np.ndarray,
            train_mask: np.ndarray,
            val_mask: np.ndarray,
            test_mask: np.ndarray,
            optimizer: str = "Adam",
            state_dimensions: int = 5,
            number_of_epochs: int = 1000,
            max_iterations: int = 50,
            convergence_threshold: float = 0.01,
            learning_rate: float = 0.001,
            print_description: bool = False
    ) -> None:
        """
        Initializes the PageRank Modeling with a GNN
        with Relaxation.

        Parameters
        ----------
        nodes : np.ndarray
            List of nodes.
        edges : np.ndarray
            List of edges (directed).
        labels : np.ndarray
            List of each node's label.
        train_mask : np.ndarray
            Boolean mask of the training set.
        val_mask : np.ndarray
            Boolean mask of the validation set.
        test_mask : np.ndarray
            Boolean mask of the test set.
        state_dimensions : integer, optional
            Number of hidden state dimensions.
        number_of_epochs : integer, optional
            Number of epochs to train for.
        max_iterations : integer, optional
            Maximum number of relaxation iterations.
        convergence_threshold : float, optional
            Convergence threshold for relaxation.
        learning_rate : float, optional
            Learning rate.
        print_description : boolean, optional
            Prints a short description of the process.
        """
        if print_description: print(RELAX_description)
        # Declares output variables
        self.list_out_train = []
        self.list_out_val   = []
        self.list_out_test  = []
        self.dset           = None
        self.model          = None
        self.nodes          = nodes.numpy()
        self.node_matrix    = torch.eye(len(labels)).numpy()
        self.edges          = edges
        self.true_labels    = labels
        self.epochs         = number_of_epochs
        # Constructs the nodes
        # Declares the dataset and dataloader
        self.dset = from_EN_to_GNN(
            self.edges, 
            self.node_matrix, 
            self.true_labels, 
            aggregation_type = "sum", 
            sparse_matrix    = True
        )
        self.dset.idx_train = train_mask
        self.dset.idx_valid = val_mask
        self.dset.idx_test  = test_mask
        # Modifies the dataset target because the
        # function dataloader.from_EN_to_GNN performs
        # the operation:
        #  > torch.tensor(labels, dtype=torch.long)
        # which sets float32 values (e.g. regression 
        # targets) to 0
        self.dset.targets = torch.Tensor(labels)
        # Declares the GNN wrapper configuration
        cfg = GNNWrapperLight.Config()
        cfg.use_cuda = False
        cfg.device   = prepare_device(n_gpu_use=0)
        cfg.epochs                = number_of_epochs
        cfg.max_iterations        = max_iterations
        cfg.convergence_threshold = convergence_threshold
        cfg.log_interval          = max_iterations*2
        cfg.state_transition_hidden_dims = [5,]
        cfg.output_function_hidden_dims  = [5]
        cfg.state_dim                    = state_dimensions
        cfg.graph_based = False
        cfg.task_type   = "semisupervised"
        cfg.lrw = learning_rate 
        # Declares the model
        self.model = RegressionGNNWrapper(cfg)
        self.model(self.dset)  # dataset initalized into the GNN
        # Updates the Optimizer
        self.model._optimizer(optimizer)

    def run(
            self,
            print_node_results: bool = False,
            print_graph_results: bool = False
    ) -> None:
        """
        Performs the training sequence for the underlying 
        GNN model with Relaxation.
        
        Parameters
        ----------
        print_node_results : boolean, optional
            Indicates whether to print a table comparing predictions
            with true PageRanks
        print_graph_results : boolean, optional
            Indicates whether to print two graphs, one with the true
            PageRank values and one with the Estimated ones
        """
        for epoch in range(1, self.epochs+1):
            out     = self.model.train_step(epoch)
            out_val = self.model.test_step(epoch, step="val")
            self.list_out_train.append(out)
            self.list_out_val.append(out_val)
        out_test = self.model.test_step(epoch, step="test")
        self.list_out_test.append(out_test)
        # Creates a Pandas DataFrame of results
        if print_node_results:
            output_results = np.column_stack((
                np.round(np.column_stack((
                    self.list_out_test[-1].detach().numpy(),
                    self.true_labels
                )),2), 
                self.dset.idx_train, 
                self.dset.idx_valid, 
                self.dset.idx_test
            ))
            print("\nResults")
            output_results = pd.DataFrame(
                output_results,
                columns = [
                    "Predictions", 
                    "Ground Truth", 
                    "Train", 
                    "Validation", 
                    "Test"
                ]
            )
            print(output_results)
        # Reconstructs a graph body
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        truth_g = g.copy()
        preds_g = g.copy()
        truth = {k:self.true_labels[k] 
                 for k in range(len(self.true_labels))}
        preds = {k:out_test[k].detach().numpy().item()
                 for k in range(len(out_test))}
        nx.set_node_attributes(truth_g, truth, "PageRank")
        nx.set_node_attributes(preds_g, preds, "PageRank")
        true_labels = {k:f"{k}\n\nPR: {np.round(truth[k].item(), 3)}" 
                         for k in range(len(truth))}
        pred_labels = {k:f"{k}\n\nPR: {np.round(preds[k], 3)}" 
                         for k in range(len(preds))}
        if print_graph_results:
            print("\n\nPlot with true labels")
            print("---------------------")
            colors = [truth_g.nodes()[x]["PageRank"] 
                      for x in truth_g.nodes()]
            plt.figure(figsize=(12,5))
            plt.title("Plot of True PageRank values")
            nx.draw_networkx(
                truth_g,
                labels=true_labels,
                pos=nx.kamada_kawai_layout(truth_g),
                node_color = colors,
                font_size  = 10,
                cmap       = cm.YlGn,
                arrows     = True
            )
            plt.show()
            print("Plot with predicted labels")
            print("--------------------------")
            colors = [preds_g.nodes()[x]["PageRank"] 
                      for x in preds_g.nodes()]
            plt.figure(figsize=(12,5))
            plt.title("Plot of Predicted PageRank values\n" + \
                      "Note: negative predictions set to 0")
            nx.draw_networkx(
                preds_g,
                labels=pred_labels,
                pos=nx.kamada_kawai_layout(preds_g),
                node_color = colors,
                font_size  = 10,
                cmap       = cm.YlGn)
            plt.show()
        print("Ground Truth vs. Predictions")
        print("--------------------------")
        plt.figure(figsize=(12,5))
        plt.title("Ground Truth vs. Predictions\n" + \
                  "(in increasing order of PageRank value)")
        truths = np.array(list(truth.values()))
        preds  = np.array(list(preds.values()))
        stack  = np.column_stack((truths, preds))
        stack  = np.sort(stack, axis=0)
        plt.scatter(
            np.linspace(1, stack.shape[0], stack.shape[0]), 
            stack[:,0]
        )
        plt.scatter(
            np.linspace(1, stack.shape[0], stack.shape[0]), 
            stack[:,1]
        )
        plt.legend(["True PageRank", "Predicted PageRank"])
        plt.show()