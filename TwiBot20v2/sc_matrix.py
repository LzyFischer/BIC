import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model import InteractModel
import json


batch_size = 64
dropout = 0.5
lr = 1e-4
weight_decay = 1e-5

no_up_limit = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_mask, val_mask, test_mask = [], [], []
split_list = pd.read_csv('data/split.csv')
label_data = pd.read_csv('data/label.csv')

users_index_to_uid = list(label_data['id'])
uid_to_users_index = {x: i for i, x in enumerate(users_index_to_uid)}
for _ in split_list[split_list['split'] == 'train']['id']:
    train_mask.append(uid_to_users_index[_])
for _ in split_list[split_list['split'] == 'val']['id']:
    val_mask.append(uid_to_users_index[_])
for _ in split_list[split_list['split'] == 'test']['id']:
    test_mask.append(uid_to_users_index[_])

all_text = torch.load('data/text.pt')
all_user_neighbor_index = json.load(open('data/user_neighbor_index.json'))
all_user_neighbor_index = [all_user_neighbor_index[str(i)] for i in range(len(all_user_neighbor_index))]
all_label = torch.load('data/node_label.pt')
all_num_feature = torch.load('data/num_properties_tensor.pt').to(device)
all_cat_feature = torch.load('data/cat_properties_tensor.pt').to(device)
all_tweet_feature = torch.load('data/tweets_tensor.pt').to(device)
all_des_feature = torch.load('data/des_tensor.pt').to(device)
all_edge_index = torch.load('data/edge_index.pt').to(device)


class InterDataset(Dataset):
    def __init__(self, idx):
        self.idx = torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        return self.idx[index]


def get_sc(self, text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index):
    num_feature = self.num_linear(num_feature)
    cat_feature = self.cat_linear(cat_feature)
    tweet_feature = self.tweet_linear(tweet_feature)
    des_feature = self.des_linear(des_feature)
    all_user_feature = torch.cat((cat_feature, num_feature, tweet_feature, des_feature), dim=-1)

    text, all_user_feature, attention_graph_0, _0 = self.Model_0(text, user_neighbor_index,
                                                             all_user_feature, edge_index)
    text, all_user_feature = self.InteractModel_0(text, all_user_feature, user_neighbor_index)

    text, all_user_feature, attention_graph_1, _1 = self.Model_1(text, user_neighbor_index,
                                                             all_user_feature, edge_index)
    title = text[:, 0]
    title = self.title_linear(title)

    user_index = []
    for neighbor_index in user_neighbor_index:
        user_index.append(neighbor_index[0])
    user_feature = all_user_feature[user_index]
    user_feature = self.user_feature_linear(user_feature)

    semantic_matrix = [_0, _1]
    attention_vec_0 = attention_graph_0.view(attention_graph_0.shape[0], self.attention_dim * self.attention_dim)
    attention_vec_1 = attention_graph_1.view(attention_graph_1.shape[0], self.attention_dim * self.attention_dim)

    attention_vec = torch.cat((attention_vec_0, attention_vec_1), dim=-1)

    return attention_vec, semantic_matrix


def forward_one_batch(batch, model):
    text = all_text[batch].to(device)
    user_neighbor_index = [all_user_neighbor_index[item.item()] for item in batch]
    return get_sc(model, text, user_neighbor_index,
                  all_num_feature, all_cat_feature, all_tweet_feature, all_des_feature, all_edge_index)


if __name__ == '__main__':

    loader = DataLoader(InterDataset([i for i in range(11826)]), batch_size=batch_size, shuffle=False)

    model = InteractModel(device=device, dropout=dropout)
    model = model.to(device)

    model.load_state_dict(torch.load('model_states/87.91_89.17_86.49_92.03.model'))

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


