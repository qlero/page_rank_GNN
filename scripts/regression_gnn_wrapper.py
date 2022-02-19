import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import Accuracy
from .pygnn import GNN

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

    def _optimizer(self):
        self.optimizer = optim.Adam(
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
    
class RegressionGNNWrapper(GNNWrapper):
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
            msg  = "\tValidation --" if step=="val" else "\tTest --"
            msg += f" Mean Loss: {str(np.round(loss.detach().numpy(), 4))}"
            msg += f", Iterations: {iterations}"
            print(msg)
        return output

