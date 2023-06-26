import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, d_input, config, total_loc_num, if_residual_layer=True):
        super(FullyConnected, self).__init__()
        # the last fully connected layer

        fc_dim = d_input
        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, fc_dim)

        self.fc_loc = nn.Linear(fc_dim, total_loc_num)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            # the residual
            self.linear1 = nn.Linear(fc_dim, fc_dim * 2)
            self.linear2 = nn.Linear(fc_dim * 2, fc_dim)

            self.norm1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout1 = nn.Dropout(p=config.fc_dropout)
            self.fc_dropout2 = nn.Dropout(p=config.fc_dropout)

    def forward(self, out, user) -> Tensor:

        # with fc output
        if self.if_embed_user:
            out = out + self.emb_user(user)

        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm1(out + self._res_block(out))

        return self.fc_loc(out)

    def _res_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
        return self.fc_dropout2(x)
