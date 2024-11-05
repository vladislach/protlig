import torch
import copy
from torch import nn
from torch.nn import ModuleList

from torch_geometric.data import Data
from torch_geometric.utils import scatter



class Net(nn.Module):
    """
    This model processes protein and small molecule embeddings and uses attention to weigh
    amino acids in the protein sequence followed by message-passing to iteratively refine
    the small molecule's embeddings for downstream prediction tasks.
    """

    def __init__(self):
        super().__init__()
        self.num_convs = 3
        
        self.x_embed = nn.Sequential(
            nn.Linear(49, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        queries = []
        for _ in range(8):
            queries.append(nn.Sequential(
                nn.Linear(68, 320),
                nn.ReLU(),
                nn.Linear(320, 640),
                nn.ReLU(),
                nn.Linear(640, 1280)
            ))
        self.queries = ModuleList(queries)
        
        values = []
        for _ in range(8):
            values.append(nn.Sequential(
                nn.Linear(1280, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            ))
        self.values = ModuleList(values)

        self.mlp_upd = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.mlp_readout = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        data = copy.deepcopy(data)
        data.x = self.x_embed(data.x)

        for _ in range(self.num_convs):
            msgs = torch.cat([data.x[data.edge_index[0]], data.x[data.edge_index[1]], data.edge_attr], dim=1)
            qs = torch.stack([query(msgs) for query in self.queries], dim=1)
            attn = qs @ data.protein_embed.T
            attn = torch.softmax(attn, dim=1).unsqueeze(-1)
            vs = torch.stack([value(data.protein_embed) for value in self.values]).unsqueeze(0)
            msgs = vs * attn
            msgs = msgs.sum(dim=-2).view(data.num_edges, -1)
            h = scatter(msgs, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce='add')
            x = torch.cat([data.x, h], dim=1)
            data.x = self.mlp_upd(x)
        
        pred = scatter(data.x, data.batch, dim=0, dim_size=data.num_graphs, reduce='add')
        pred = self.mlp_readout(pred)
        return pred

