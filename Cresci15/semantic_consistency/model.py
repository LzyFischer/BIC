import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv, GATConv, RGCNConv, SAGEConv
from einops import rearrange


def get_batch_text(x):
    text = []
    padding_mask = []
    for item in x:
        length = item.shape[0]
        padding_mask.append(torch.tensor([0] * length + [1] * (200 - length), dtype=torch.bool))
        # item = torch.tensor(item, dtype=torch.float)
        padding = [torch.zeros(768, dtype=torch.float) for _ in range(length, 200)]
        if padding:
            padding = torch.stack(padding)
            item = torch.cat([item, padding], dim=0)
        text.append(item)
    padding_mask = torch.stack(padding_mask)
    text = torch.stack(text)
    return text, padding_mask


class TextModule(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, text, padding_mask):
        res1, res2 = self.attn(text, text, text, key_padding_mask=padding_mask)
        return res1, res2


class GraphModule(nn.Module):
    def __init__(self, hidden_dim, mode, num_heads, dropout):
        super().__init__()
        assert mode in ['gcn', 'gat', 'sage', 'rgcn']
        if mode == 'gcn':
            self.gnn = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        elif mode == 'gat':
            self.gnn = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        elif mode == 'rgcn':
            self.gnn = RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=2)
        elif mode == 'sage':
            self.gnn = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.mode = mode
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, rep, edge_index, edge_type, size):
        if self.mode == 'rgcn':
            res = self.gnn(rep, edge_index, edge_type)
        else:
            res = self.gnn(rep, edge_index)
        res = self.drop(res)
        users = []
        for i in range(size):
            neighbor_0 = edge_index[0][edge_index[1] == i].to('cpu').numpy()
            neighbor_1 = edge_index[1][edge_index[0] == i].to('cpu').numpy()
            neighbor = [i] + list(set(neighbor_1) | set(neighbor_0))
            neighbor = res[neighbor]
            neighbor, _ = self.attn(neighbor, neighbor, neighbor)
            users.append(neighbor[0])
        users = torch.stack(users)
        res = torch.cat([users, res[size:]], dim=0)
        return res


class TextEncoding(nn.Module):
    def __init__(self, hidden_dim, des_dim, sta_dim):
        super().__init__()
        self.des_fc = nn.Linear(des_dim, hidden_dim)
        self.sta_fc = nn.Linear(sta_dim, hidden_dim)

    def forward(self, des, sta):
        des = self.des_fc(des).unsqueeze(1)
        sta = self.sta_fc(sta)
        return torch.cat([des, sta], dim=1)


class GraphEncoding(nn.Module):
    def __init__(self, hidden_dim, num_dim, cat_dim, des_dim, twe_dim):
        super().__init__()
        self.num_fc = nn.Linear(num_dim, hidden_dim // 4)
        self.cat_fc = nn.Linear(cat_dim, hidden_dim // 4)
        self.des_fc = nn.Linear(des_dim, hidden_dim // 4)
        self.twe_fc = nn.Linear(twe_dim, hidden_dim // 4)

    def forward(self, num, cat, des, twe):
        num = self.num_fc(num)
        cat = self.cat_fc(cat)
        des = self.des_fc(des)
        twe = self.twe_fc(twe)
        return torch.cat([num, cat, des, twe], dim=-1)


class InterLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mode):
        super().__init__()
        self.text_model = TextModule(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.graph_model = GraphModule(hidden_dim=hidden_dim, num_heads=num_heads, mode=mode, dropout=dropout)
        self.inter = InterHead(hidden_dim=hidden_dim)

    def forward(self, text, padding_mask, graph, edge_index, edge_type, size, inter=True):
        text, padding_mask = text[:size], padding_mask[:size]
        des_mask = torch.zeros(text.shape[0], dtype=torch.bool, device=text.device).unsqueeze(1)
        padding_mask = torch.cat([des_mask, padding_mask], dim=1)
        text, attn_map = self.text_model(text, padding_mask)
        graph = self.graph_model(graph, edge_index, edge_type, size)
        if not inter:
            return text, graph, attn_map
        text_head = text[:, 0, :]
        graph_head = graph[:size]
        text_head, graph_head = self.inter(text_head, graph_head)
        text = torch.cat([text_head.unsqueeze(1), text[:, 1:, :]], dim=1)
        graph = torch.cat([graph_head, graph[size:]], dim=0)
        return text, graph, attn_map


class InterHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.text_fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.graph_fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, text, graph):
        text_sen = self.text_fc(text)
        graph_sen = self.graph_fc(graph)

        a_tt = torch.mul(text, text_sen).sum(-1)
        a_tg = torch.mul(text, graph_sen).sum(-1)
        a_gt = torch.mul(graph, text_sen).sum(-1)
        a_gg = torch.mul(graph, graph_sen).sum(-1)

        a_tt, a_tg = self.softmax(torch.stack([a_tt, a_tg])).split([1, 1], dim=0)
        a_gt, a_gg = self.softmax(torch.stack([a_gt, a_gg])).split([1, 1], dim=0)

        a_tt = a_tt.squeeze(0).unsqueeze(-1)
        a_tg = a_tg.squeeze(0).unsqueeze(-1)
        a_gt = a_gt.squeeze(0).unsqueeze(-1)
        a_gg = a_gg.squeeze(0).unsqueeze(-1)

        text = torch.mul(a_tt, text) + torch.mul(a_tg, graph)
        graph = torch.mul(a_gt, text) + torch.mul(a_gg, graph)
        return text, graph


class SemanticEncoding(nn.Module):
    def __init__(self, fixed_size, hidden_dim):
        super().__init__()
        self.fixed_size = fixed_size
        self.fc = nn.Linear(fixed_size ** 2, hidden_dim)

    def forward(self, x):
        b, w, h = x.shape
        p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
        p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
        x = nn.functional.pad(x, (0, p_h, 0, p_w))
        pool_size = (((w + self.fixed_size - 1) // self.fixed_size), ((h + self.fixed_size - 1) // self.fixed_size))
        pool = nn.MaxPool2d(pool_size, stride=pool_size)
        x = pool(x)
        x = rearrange(x, 'b w h -> b (w h)')
        x = self.fc(x)
        return x


class BIC(nn.Module):
    def __init__(self, hidden_dim, num_layers, des_dim, sta_dim, twe_dim, num_dim, cat_dim,
                 num_heads, fixed_size, graph_mode, semantic_mode, dropout):
        super().__init__()
        self.layers = num_layers
        assert graph_mode in ['rgcn', 'gcn', 'gat', 'sage']
        assert semantic_mode in ['last', 'first', 'cat', 'mean', 'none']
        self.graph_mode = graph_mode
        self.semantic_mode = semantic_mode
        self.text_encoding = TextEncoding(hidden_dim=hidden_dim, des_dim=des_dim, sta_dim=sta_dim)
        self.graph_encoding = GraphEncoding(hidden_dim=hidden_dim,
                                            des_dim=des_dim, twe_dim=twe_dim,
                                            num_dim=num_dim, cat_dim=cat_dim)
        self.inter = nn.ModuleList()
        for i in range(self.layers):
            self.inter.append(InterLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, mode=graph_mode))
        self.semantic_encoding = SemanticEncoding(fixed_size=fixed_size, hidden_dim=hidden_dim)
        if semantic_mode == 'cat':
            self.fc = nn.Linear((3 + self.layers - 1) * hidden_dim, hidden_dim)
        elif semantic_mode == 'none':
            self.fc = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.fc = nn.Linear(3 * hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 2)
        self.act_func = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, padding_mask, edge_index, edge_type, tweet, description, numerical, categorical, batch_size):
        text_code = self.text_encoding(description[:batch_size], text[:batch_size])
        graph_code = self.graph_encoding(numerical, categorical, description, tweet)
        res_text_code = text_code
        res_graph_code = graph_code
        text_code = self.dropout(self.act_func(text_code))
        graph_code = self.dropout(self.act_func(graph_code))
        semantic_codes = []
        for i in range(self.layers):
            text_code, graph_code, _ = self.inter[i](text_code, padding_mask,
                                                     graph_code, edge_index, edge_type, batch_size)
            semantic_codes.append(self.semantic_encoding(_))
        # text_code, graph_code, _ = self.inter[-1](text_code, padding_mask,
        #                                           graph_code, edge_index, edge_type, batch_size, inter=False)
        # semantic_codes.append(self.semantic_encoding(_))
        text_code = res_text_code + text_code
        graph_code = res_graph_code + graph_code
        text_code = self.dropout(self.act_func(text_code))
        graph_code = self.dropout(self.act_func(graph_code))
        # print(len(semantic_codes))
        if self.semantic_mode == 'none':
            rep = torch.cat([text_code[:batch_size, 0, :],
                             graph_code[:batch_size]], dim=-1)
        else:
            if self.semantic_mode == 'first':
                semantic_codes = self.dropout(self.act_func(semantic_codes[0]))
            elif self.semantic_mode == 'last':
                semantic_codes = self.dropout(self.act_func(semantic_codes[-1]))
            elif self.semantic_mode == 'cat':
                semantic_codes = self.dropout(self.act_func(torch.cat(semantic_codes, dim=-1)))
            else:
                semantic_codes = torch.mean(torch.stack(semantic_codes), dim=0)
                semantic_codes = self.dropout(self.act_func(semantic_codes))
            rep = torch.cat([text_code[:batch_size, 0, :],
                             graph_code[:batch_size],
                             semantic_codes[:batch_size]], dim=-1)
        rep = self.act_func(self.dropout(self.fc(rep[:batch_size])))
        return self.cls(rep)
