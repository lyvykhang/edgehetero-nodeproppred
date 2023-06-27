import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, n_layers, n_features, n_classes, n_hidden):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(n_features, n_hidden, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        for _ in range(n_layers - 2):
            self.convs.append(
                GCNConv(n_hidden, n_hidden, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.convs.append(GCNConv(n_hidden, n_classes, cached=True))
    

    @staticmethod
    def reset_parameters(model):
        """Used to reset model parameters between folds/runs. 
        Assumes the input model has already been hetero-transformed.
        """
        for conv in model.convs:
            for k in conv.keys():
                conv[k].reset_parameters()
        for bn in model.bns:
            for k in bn.keys():
                bn[k].reset_parameters()
            
            
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=1)
