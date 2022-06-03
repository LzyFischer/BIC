import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import NeighborSampler
from src.utils import FixedPooling

class InteractModel(nn.Module):
    """
    """
    def __init__(self, num_property_dim=5, cat_property_dim=3, tweet_dim=768,
                 des_dim=768, input_dim=768, hidden_dim=768, output_dim=768, 
                 attention_dim=16, graph_num_heads=4, dropout=0.1, device='cuda:3'):
        super(InteractModel, self).__init__()
        self.device = device
        
        self.dropout = nn.Dropout(p=dropout)
        self.attention_dim = attention_dim
        
        self.text_linear = nn.Linear(input_dim, hidden_dim)
        """
        only used when input_dim != hidden_dim
        """

        self.num_linear = nn.Linear(num_property_dim, hidden_dim // 4)
        self.cat_linear = nn.Linear(cat_property_dim, hidden_dim // 4)
        self.tweet_linear = nn.Linear(tweet_dim, hidden_dim // 4)
        self.des_linear = nn.Linear(des_dim, hidden_dim // 4)
        
        self.Model_0 = RespectiveLayer(in_channels_for_graph=hidden_dim, in_channels_for_text=hidden_dim, out_channels=hidden_dim, attention_dim=attention_dim, graph_num_heads=graph_num_heads, device=self.device)
        self.Model_1 = RespectiveLayer(in_channels_for_graph=hidden_dim, in_channels_for_text=hidden_dim, out_channels=hidden_dim, attention_dim=attention_dim, graph_num_heads=graph_num_heads, device=self.device)
        self.Model_2 = RespectiveLayer(in_channels_for_graph=hidden_dim, in_channels_for_text=hidden_dim, out_channels=output_dim, attention_dim=attention_dim, graph_num_heads=graph_num_heads, device=self.device)
        
        self.InteractModel_0 = InteractLayer(in_channels=hidden_dim, out_channels=hidden_dim)
        self.InteractModel_1 = InteractLayer(in_channels=hidden_dim, out_channels=hidden_dim)
        
        self.attention_linear = nn.Linear(attention_dim * attention_dim * 2, output_dim // 3)
        self.user_feature_linear = nn.Linear(output_dim, output_dim // 3)
        self.title_linear = nn.Linear(output_dim, output_dim // 3) 

        self.final_linear = nn.Linear(output_dim, 2)
    
    def forward(self, text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index):
        """
        text: batch_size * 200+1 * 768
        """
        num_feature = self.num_linear(num_feature)
        cat_feature = self.cat_linear(cat_feature)
        tweet_feature = self.tweet_linear(tweet_feature)
        des_feature = self.des_linear(des_feature)
        all_user_feature = torch.cat((cat_feature, num_feature, tweet_feature, des_feature), dim=-1)
        
        # text = self.text_linear(text)

        text, all_user_feature, attention_graph_0 = self.Model_0(text, user_neighbor_index, 
                                                               all_user_feature, edge_index)
        text, all_user_feature = self.InteractModel_0(text, all_user_feature, user_neighbor_index)
        
        text, all_user_feature, attention_graph_1 = self.Model_1(text, user_neighbor_index, 
                                                               all_user_feature, edge_index)
        # text, all_user_feature = self.InteractModel_1(text, all_user_feature, user_neighbor_index)

        # text, all_user_feature, attention_graph_2 = self.Model_2(text, user_neighbor_index, 
        #                                                        all_user_feature, edge_index)
        title = text[:, 0]
        title = self.title_linear(title) 

        user_index = []
        for neighbor_index in user_neighbor_index:
            user_index.append(neighbor_index[0])
        user_feature = all_user_feature[user_index]
        user_feature = self.user_feature_linear(user_feature) 
        
        attention_vec_0 = attention_graph_0.view(attention_graph_0.shape[0], self.attention_dim * self.attention_dim)
        attention_vec_1 = attention_graph_1.view(attention_graph_1.shape[0], self.attention_dim * self.attention_dim)
        # attention_vec_2 = attention_graph_2.view(attention_graph_2.shape[0], self.attention_dim * self.attention_dim)
        
        attention_vec = torch.cat((attention_vec_0, attention_vec_1), dim=-1)
        attention_vec = self.attention_linear(attention_vec)
        
        final_input = torch.cat((attention_vec, title, user_feature), dim=-1)
        final_output = self.final_linear(final_input)
        
        return final_output
    
    
    
class RespectiveLayer(nn.Module):
    """
    assume LM & GM has same layer
    """
    def __init__(self, in_channels_for_graph=768, in_channels_for_text=768, out_channels=768, attention_dim=6, graph_num_heads=4, text_num_heads=4, device='cuda:4'):
        super(RespectiveLayer, self).__init__()
        self.device = device
        self.attention_dim = attention_dim
        
        self.dropout = nn.Dropout(0.5)
        self.GCN = GCNConv(in_channels=in_channels_for_graph, out_channels=out_channels)
        self.MultiAttn = MultiAttn(embed_dim=out_channels, num_heads=graph_num_heads)
        self.LModel = LModel(embed_dim=in_channels_for_text, num_heads=text_num_heads, dropout=0.1)
        self.FixedPooling = FixedPooling(fixed_size=self.attention_dim)
    
    def forward(self, user_text, user_neighbor_index, all_user_feature, edge_index):
        """
        user_neighbor_index: dict(n * 1) * num_batch
        all_user_feature: tensor(768 * 229580)
        text: tensor(768 * length) * num_bacth
        attention_graph = num_batch * dim * dim
        """
        user_index = []
        for neighbor_index in user_neighbor_index:
            user_index.append(neighbor_index[0])
        user_index = torch.tensor(user_index)
        subgraph_loader = NeighborSampler(edge_index=edge_index, node_idx=user_index, sizes=[-1], batch_size=len(user_index))

        text, attention = self.LModel(user_text)
   
        for _, _, adj in subgraph_loader:
            index = adj[0].to(self.device)
            all_user_feature = self.GCN(all_user_feature, index)
        # all_user_feature = self.GCN(all_user_feature, edge_index)
        all_user_feature = self.MultiAttn(user_neighbor_index, all_user_feature)
        attention_graph = self.FixedPooling(attention)
        
        return text, all_user_feature, attention_graph

    
class InteractLayer(nn.Module): 
    def __init__(self, in_channels=768, out_channels=768):
        super(InteractLayer, self).__init__()
        self.linear_text = nn.Linear(in_channels, out_channels)
        self.linear_graph = nn.Linear(in_channels, out_channels)
    
    def forward(self, text, all_user_feature, user_neighbor_index):
        
        assert len(user_neighbor_index) == len(text)
        user_index = []
        for neighbor_index in user_neighbor_index:
            user_index.append(neighbor_index[0])
        
        graph_ini = all_user_feature[user_index]
        text_ini, text_rest = text.split([1,200], dim=1)
        text_ini = text_ini.squeeze(1)
        
        text = self.linear_text(graph_ini)
        # softmax = nn.Softmax(dim=0)
        # a = torch.mul(text_ini, text_tmp).sum(dim=-1).unsqueeze(-1)
        # b = torch.mul(graph_ini, text_ini).sum(dim=-1).unsqueeze(-1)
        # a_b = torch.stack((a,b))
        # a_b = softmax(a_b)
        # a, b = a_b.split([1,1], dim=0)
        # a, b = a.squeeze(0), b.squeeze(0)

        # text = torch.mul(a, text_ini) + torch.mul(b, graph_ini)

        text = torch.cat((text.unsqueeze(1), text_rest), dim=1)
        
        graph = self.linear_graph(graph_ini)
        # c = torch.mul(graph_tmp, graph_ini).sum(dim=-1).unsqueeze(-1)
        # d = torch.mul(graph_ini, text_ini).sum(dim=-1).unsqueeze(-1)
        # c_d = torch.stack((c, d))
        # c_d = softmax(c_d)
        # c, d = c_d.split([1,1], dim=0)
        # c, d = c.squeeze(0), d.squeeze(0)
        # graph = torch.mul(c, graph_ini) + torch.mul(d, text_ini)

        for i in range(len(user_index)):
            all_user_feature[user_index[i]] = graph[i]
        
        return text, all_user_feature
    
    
class MultiAttn(nn.Module):
    """
    """
    def __init__(self, embed_dim=768, num_heads=4):
        super(MultiAttn, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, user_neighbor_index, all_user_feature):
        for user_index in user_neighbor_index:
            tmp_feature = all_user_feature[user_index].unsqueeze(0)
            tmp_feature, attention_weight = self.multihead_attention(tmp_feature, tmp_feature, tmp_feature) #batch个multi-attention
            all_user_feature[user_index[0]] = tmp_feature[0][0]
        return all_user_feature
    

class LModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.1, activation='LeakyReLU', 
                 norm_first=True, layer_norm_eps=1e-5):
        super(LModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                                         dropout=dropout, batch_first=True)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        if activation == 'SELU':
            self.activation = nn.SELU()
        
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, text_src):
        if self.norm_first:
            text, attention_weight = self._sa_block(self.norm1(text_src))
            text = text_src + text
            text = text + self._ff_block(self.norm2(text))
        else:
            text, attention_weight = self._sa_block(text_src)
            text = self.norm1(text_src + text)
            text = self.norm2(text + self._ff_block(text))
        return text, attention_weight
        
    def _sa_block(self, text):
        text, attention_weight = self.multihead_attention(text, text, text)
        text = self.dropout1(text)
        return text, attention_weight
    
    def _ff_block(self, text):
        text = self.linear2(self.dropout(self.activation(self.linear1(text))))
        text = self.dropout2(text)
        return text
