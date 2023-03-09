import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
import torch


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]


class SemanticAttention(nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.num_head = num_head
        self.att_layers = nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
                nn.Sequential(
                    nn.Linear(in_channel, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1, bias=False))
            )

    def forward(self, z):
        w = self.att_layers[0](z).mean(0)
        beta = torch.softmax(w, dim=0)

        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)

            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp

        return output / self.num_head


class RGTLayer(nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channel + out_channel, in_channel),
            nn.Sigmoid()
        )

        self.activation = nn.ELU()
        self.transformer_list = nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(
                TransformerConv(in_channels=in_channel, out_channels=out_channel, heads=trans_heads, dropout=dropout,
                                concat=False))

        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        r"""
        feature: input node features
        edge_index: all edge index, shape (2, num_edges)
        edge_type: same as RGCNconv in torch_geometric
        num_rel: number of relations
        beta: return cross relation attention weight
        agg: aggregation type across relation embedding
        """

        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).flatten(1)
        a = self.gate(torch.cat((u, features), dim=1))

        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))).unsqueeze(1)

        for i in range(1, len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gate(torch.cat((u, features), dim=1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))
            semantic_embeddings = torch.cat((semantic_embeddings, output.unsqueeze(1)), dim=1)

            return self.semantic_attention(semantic_embeddings)


class RGTDetector(nn.Module):
    def __init__(self, numeric_num, cat_num, tweet_channel, des_channel, linear_channels,
                 out_channel, trans_head, semantic_head, dropout):
        super(RGTDetector, self).__init__()
        self.in_linear_numeric = nn.Linear(numeric_num, linear_channels // 4, bias=True)
        self.in_linear_bool = nn.Linear(cat_num, linear_channels // 4, bias=True)
        self.in_linear_tweet = nn.Linear(tweet_channel, linear_channels // 4, bias=True)
        self.in_linear_des = nn.Linear(des_channel, linear_channels // 4, bias=True)
        self.linear1 = nn.Linear(linear_channels, linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=2, in_channel=linear_channels, out_channel=out_channel,
                                   trans_heads=trans_head, semantic_head=semantic_head, dropout=dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=2, in_channel=linear_channels, out_channel=out_channel,
                                   trans_heads=trans_head, semantic_head=semantic_head, dropout=dropout)

        self.out1 = torch.nn.Linear(out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

    def forward(self, cat_features, prop_features, tweet_features, des_features, edge_index, edge_type):
        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))

        user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, user_features_des),
                                  dim=1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))
        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)
        return pred
