import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

class GNN(torch.nn.Module):

    def __init__(self, GNNtype, Flist, MLPlist, softmax=False, Klist = None):

        super(GNN, self).__init__()
        self.type = GNNtype
        self.Flist = Flist
        self.L = len(Flist)
        self.MLPlist = MLPlist 
        self.Lmlp = len(MLPlist)
        self.softmax = softmax
        self.Klist = Klist

        self.layers = nn.ModuleList()
        self.MLPlayers = nn.ModuleList()

        for i in range(self.L-1):
            if self.type == 'sage':
                self.layers.append(SAGEConv(Flist[i],Flist[i+1]))
            elif self.type == 'gcn':
                self.layers.append(GCNConv(Flist[i],Flist[i+1]))

        for i in range(self.Lmlp-1):
            self.MLPlayers.append(nn.Linear(MLPlist[i],MLPlist[i+1]))
            
    def forward(self, data):

        y, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for i, layer in enumerate(self.layers):
            y = layer(y, edge_index=edge_index)
            y = F.relu(y)

        for i, layer in enumerate(self.MLPlayers):
            y = layer(y)
        
        if self.softmax == True:
            y = F.log_softmax(y, dim=1)

        return y