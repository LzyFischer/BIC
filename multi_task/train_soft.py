import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model_soft import InteractModel
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64
dropout = 0.5
lr = 1e-4
weight_decay = 1e-5
no_up_limit = 8


path = '../TwiBot20v2/data'
train_mask, val_mask, test_mask = [], [], []
split_list = pd.read_csv('{}/split.csv'.format(path))
label_data = pd.read_csv('{}/label.csv'.format(path))


users_index_to_uid = list(label_data['id'])
uid_to_users_index = {x: i for i, x in enumerate(users_index_to_uid)}
for _ in split_list[split_list['split'] == 'train']['id']:
    train_mask.append(uid_to_users_index[_])
for _ in split_list[split_list['split'] == 'val']['id']:
    val_mask.append(uid_to_users_index[_])
for _ in split_list[split_list['split'] == 'test']['id']:
    test_mask.append(uid_to_users_index[_])

all_text = torch.load('{}/text.pt'.format(path))
all_user_neighbor_index = json.load(open('{}/user_neighbor_index.json'.format(path)))
all_user_neighbor_index = [all_user_neighbor_index[str(i)] for i in range(len(all_user_neighbor_index))]
all_label = torch.load('{}/node_label.pt'.format(path))
all_num_feature = torch.load('{}/num_properties_tensor.pt'.format(path)).to(device)
all_cat_feature = torch.load('{}/cat_properties_tensor.pt'.format(path)).to(device)
all_tweet_feature = torch.load('{}/tweets_tensor.pt'.format(path)).to(device)
all_des_feature = torch.load('{}/des_tensor.pt'.format(path)).to(device)
all_edge_index = torch.load('{}/edge_index.pt'.format(path)).to(device)


class InterDataset(Dataset):
    def __init__(self, idx):
        self.idx = torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        return self.idx[index]


def forward_one_batch(model, batch):
    text = all_text[batch].to(device)
    user_neighbor_index = [all_user_neighbor_index[item.item()] for item in batch]
    return model(text, user_neighbor_index,
                 all_num_feature, all_cat_feature, all_tweet_feature, all_des_feature, all_edge_index)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        all_truth.append(all_label[batch])
        text_pred, graph_pred = forward_one_batch(model, batch)
        out = text_pred + graph_pred
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return [accuracy_score(all_truth, all_preds) * 100,
            f1_score(all_truth, all_preds) * 100,
            precision_score(all_truth, all_preds) * 100,
            recall_score(all_truth, all_preds) * 100]


def train_one_epoch(model, optimizer, loss_fn):
    for batch in tqdm(train_loader, ncols=0):
        optimizer.zero_grad()
        text_pred, graph_pred = forward_one_batch(model, batch)
        label = all_label[batch].to(device)
        model_loss_fn = nn.MSELoss()
        model_loss_1 = model_loss_fn(model.InteractModel_0.mlp_text.weight, model.InteractModel_0.mlp_graph.weight)
        model_loss_2 = model_loss_fn(model.InteractModel_0.mlp_text.bias, model.InteractModel_0.mlp_graph.bias)
        model_loss = model_loss_1 + model_loss_2
        loss = loss_fn(text_pred, label) + loss_fn(graph_pred, label) + 1e-2 * model_loss
        loss.backward()
        optimizer.step()


def train(save_path='model_states'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = InteractModel(device=device, dropout=dropout)
    model = model.to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_model = model.state_dict()
    best_acc = 0
    no_up = 0
    epoch = 0
    while True:
        train_one_epoch(model, optimizer, loss_fn)
        acc, f1, pre, rec = evaluate(model, val_loader)
        if acc > best_acc:
            no_up = 0
            best_acc = acc
            best_model = model.state_dict()
        else:
            no_up += 1
        if no_up == no_up_limit:
            break
        print('epoch {}, no up: {}, best acc: {}'.format(epoch, no_up, best_acc))
        epoch += 1
    # best_model = torch.load('model_states/87.91_89.17_86.49_92.03.model')
    model.load_state_dict(best_model)
    acc, f1, pre, rec = evaluate(model, test_loader)
    print('training done.')
    print('acc {:.2f} f1 {:.2f} pre {:.2f} rec {:.2f}'.format(acc, f1, pre, rec))
    torch.save(best_model, '{}/{:.2f}_{:.2f}_{:.2f}_{:.2f}.model'.format(save_path, acc, f1, pre, rec))


if __name__ == '__main__':
    train_dataset = InterDataset(train_mask)
    val_dataset = InterDataset(val_mask)
    test_dataset = InterDataset(test_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    while True:
        train('soft_states')
