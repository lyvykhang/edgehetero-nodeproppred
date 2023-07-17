import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, JumpingKnowledge


class GCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
			
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
 
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=-1)

 
class GCNJKNet(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout, mode):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels, num_layers=num_layers)
        if mode == 'cat':
            self.lin = torch.nn.Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, adj_t):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            xs += [x]

        x = self.jump(xs)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)


class SGC(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=num_layers,
                            cached=True)

    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)

        return F.log_softmax(x, dim=1)
