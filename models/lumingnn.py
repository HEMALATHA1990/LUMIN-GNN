import torch
import torch.nn as nn
import torch.nn.functional as F


class LUMINGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        K=3,
        dropout=0.5
    ):
        super().__init__()

        self.K = K
        self.dropout = dropout

        self.input_proj = nn.Linear(
            in_channels,
            hidden_channels
        )

        self.mean_proj = nn.Linear(
            hidden_channels,
            hidden_channels
        )

        self.var_proj = nn.Linear(
            hidden_channels,
            hidden_channels
        )

        self.beta = nn.Parameter(
            torch.ones(K)
        )

        self.classifier = nn.Linear(
            hidden_channels,
            out_channels
        )

    def propagate(
        self,
        x,
        edge_index
    ):
        row, col = edge_index

        deg = torch.bincount(
            row,
            minlength=x.size(0)
        ).float()

        deg_inv = deg.pow(-1)
        deg_inv[
            torch.isinf(deg_inv)
        ] = 0

        out = torch.zeros_like(x)

        out.index_add_(
            0,
            row,
            x[col] *
            deg_inv[row].unsqueeze(1)
        )

        return out

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        x = self.input_proj(x)
        x = F.relu(x)

        mu = self.mean_proj(x)

        log_var = self.var_proj(x)

        sigma = torch.exp(
            0.5 * log_var
        )

        eps = torch.randn_like(
            sigma
        )

        z = mu + eps * sigma

        gate = torch.sigmoid(
            -sigma.mean(
                dim=1,
                keepdim=True
            )
        )

        h = z * gate

        outs = []

        current = h

        for k in range(self.K):

            current = self.propagate(
                current,
                edge_index
            )

            outs.append(
                self.beta[k] *
                current
            )

        h = torch.stack(
            outs,
            dim=0
        ).sum(dim=0)

        h = F.relu(h)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training
        )

        out = self.classifier(h)

        return out, mu, log_var