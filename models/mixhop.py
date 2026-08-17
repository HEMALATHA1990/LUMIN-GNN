import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class MixHop(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5
    ):
        super().__init__()

        self.lin0 = nn.Linear(
            in_channels,
            hidden_channels
        )

        self.lin1 = nn.Linear(
            in_channels,
            hidden_channels
        )

        self.lin2 = nn.Linear(
            in_channels,
            hidden_channels
        )

        self.classifier = nn.Linear(
            hidden_channels * 3,
            out_channels
        )

        self.dropout = dropout

    def forward(self, data):

        x = data.x

        A = to_dense_adj(
            data.edge_index
        )[0].to(x.device)

        A = A + torch.eye(
            A.size(0),
            device=x.device
        )

        deg = A.sum(1)

        deg_inv = deg.pow(-1)
        deg_inv[torch.isinf(deg_inv)] = 0

        D = torch.diag(
            deg_inv
        )

        A = D @ A

        x0 = self.lin0(x)
        x1 = self.lin1(A @ x)
        x2 = self.lin2(A @ A @ x)

        x = torch.cat(
            [x0, x1, x2],
            dim=1
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.classifier(x)

        return x