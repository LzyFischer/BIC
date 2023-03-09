import torch
import numpy as np
from model import get_batch_text, BIC
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


batch_size = 64
hidden_dim = 256
num_layers = 2
dropout = 0.5
fixed_size = 16

no_up_limit = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


label = torch.load('data/label.pt')
tweet = torch.load('data/tweet.pt')
description = torch.load('data/description.pt')
numerical = torch.load('data/numerical.pt')
categorical = torch.load('data/categorical.pt')
edge_index = torch.load('data/edge_index.pt')
edge_type = torch.load('data/edge_type.pt')
status = torch.load('data/text.pt')
status = status[:, :200, :]
# status = np.load('labeled_status.npy', allow_pickle=True)
uid = torch.arange(label.shape[0])
data = Data(x=uid, y=label, edge_index=edge_index, edge_type=edge_type)

loader = NeighborLoader(data,
                        num_neighbors=[256, 256, 256, 256],
                        batch_size=batch_size,
                        shuffle=False)


def get_sc(self, text, padding_mask, edge_index, edge_type, tweet, description, numerical, categorical, batch_size):
    text_code = self.text_encoding(description[:batch_size], text[:batch_size])
    graph_code = self.graph_encoding(numerical, categorical, description, tweet)
    res_text_code = text_code
    res_graph_code = graph_code
    text_code = self.dropout(self.act_func(text_code))
    graph_code = self.dropout(self.act_func(graph_code))
    semantic_codes = []
    semantic_matrix = []
    for i in range(self.layers):
        text_code, graph_code, _ = self.inter[i](text_code, padding_mask,
                                                 graph_code, edge_index, edge_type, batch_size)
        semantic_matrix.append(_)
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
        return semantic_codes, semantic_matrix


def forward_one_batch(batch, model):
    size = batch.batch_size
    batch_edge_index = batch.edge_index.to(device)
    batch_edge_type = batch.edge_type.to(device)
    batch_tweet = tweet[batch.x].to(device)
    batch_description = description[batch.x].to(device)
    batch_numerical = numerical[batch.x].to(device)
    batch_categorical = categorical[batch.x].to(device)
    batch_status = status[batch.x[:size]]
    text, padding_mask = get_batch_text(batch_status)
    text, padding_mask = text.to(device), padding_mask.to(device)
    return get_sc(model, text, padding_mask,
                  batch_edge_index, batch_edge_type,
                  batch_tweet, batch_description,
                  batch_numerical, batch_categorical, size)


if __name__ == '__main__':
    model = BIC(hidden_dim=hidden_dim, num_layers=num_layers, des_dim=768, sta_dim=768,
                twe_dim=768, num_dim=5, cat_dim=1,
                num_heads=4, fixed_size=fixed_size,
                graph_mode='rgcn', semantic_mode='last',
                dropout=dropout).to(device)
    model.load_state_dict(torch.load('model_states_new/98.69_98.97_98.25_99.70.model'))
    reps = []
    matrix0 = []
    matrix1 = []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=0):
            rep, matrix = forward_one_batch(batch, model)
            reps.append(rep)
            matrix0.append(matrix[0])
            matrix1.append(matrix[1])
    reps = torch.cat(reps, dim=0)
    matrix0 = torch.cat(matrix0, dim=0)
    matrix1 = torch.cat(matrix1, dim=0)
    print(reps.shape, matrix0.shape, matrix1.shape)
    torch.save(reps, 'Cresci_reps.pt')
    torch.save(matrix0, 'Cresci_matrix0.pt')
    torch.save(matrix1, 'Cresci_matrix1.pt')


