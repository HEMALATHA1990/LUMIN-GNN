import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class H2GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = nn.Linear(
            hidden_channels * 2,
            out_channels
        )

        self.dropout = dropout

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        h1 = self.conv1(
            x,
            edge_index
        )

        h1 = F.relu(h1)

        h1 = F.dropout(
            h1,
            p=self.dropout,
            training=self.training
        )

        h2 = self.conv2(
            h1,
            edge_index
        )

        h2 = F.relu(h2)

        h2 = F.dropout(
            h2,
            p=self.dropout,
            training=self.training
        )

        h = torch.cat(
            [h1, h2],
            dim=1
        )

        out = self.classifier(h)

        return out