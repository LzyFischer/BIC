import os.path

import torch
import numpy as np
from model_soft import get_batch_text, BIC
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


batch_size = 64
lr = 1e-3
weight_decay = 1e-5
hidden_dim = 256
num_layers = 2
dropout = 0.5
fixed_size = 16

no_up_limit = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = '../Cresci15/data'
label = torch.load('{}/label.pt'.format(path))
tweet = torch.load('{}/tweet.pt'.format(path))
description = torch.load('{}/description.pt'.format(path))
numerical = torch.load('{}/numerical.pt'.format(path))
categorical = torch.load('{}/categorical.pt'.format(path))
edge_index = torch.load('{}/edge_index.pt'.format(path))
edge_type = torch.load('{}/edge_type.pt'.format(path))
status = torch.load('{}/text.pt'.format(path))
status = status[:, :200, :]
# status = np.load('labeled_status.npy', allow_pickle=True)
uid = torch.arange(label.shape[0])
data = Data(x=uid, y=label, edge_index=edge_index, edge_type=edge_type)
train_mask = torch.load('{}/train.pt'.format(path))
val_mask = torch.load('{}/val.pt'.format(path))
test_mask = torch.load('{}/test.pt'.format(path))


train_loader = NeighborLoader(data,
                              num_neighbors=[256, 256, 256, 256],
                              batch_size=batch_size,
                              input_nodes=train_mask,
                              shuffle=True)
val_loader = NeighborLoader(data,
                            num_neighbors=[256, 256, 256, 256],
                            batch_size=batch_size,
                            input_nodes=val_mask,
                            shuffle=False)
test_loader = NeighborLoader(data,
                             num_neighbors=[256, 256, 256, 256],
                             batch_size=batch_size,
                             input_nodes=test_mask,
                             shuffle=False)


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
    return model(text, padding_mask,
                 batch_edge_index, batch_edge_type,
                 batch_tweet, batch_description,
                 batch_numerical, batch_categorical, size)


def train_one_epoch(model, optimizer, loss_fn):
    model.train()
    # ave_loss = 0
    for batch in tqdm(train_loader, ncols=0):
        optimizer.zero_grad()
        batch_label = batch.y.to(device)[:batch.batch_size]
        text_out, graph_out = forward_one_batch(batch, model)

        model_loss_fn = torch.nn.MSELoss()
        model_loss_1 = model_loss_fn(model.inter[0].inter.mlp_text.weight, model.inter[0].inter.mlp_graph.weight)
        model_loss_2 = model_loss_fn(model.inter[0].inter.mlp_text.bias, model.inter[0].inter.mlp_graph.bias)
        model_loss = model_loss_1 + model_loss_2
        loss = loss_fn(text_out, batch_label) + loss_fn(graph_out, batch_label) + 1e-2 * model_loss
        # ave_loss += loss.item()
        loss.backward()
        optimizer.step()
    # print(ave_loss / len(train_loader))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        all_truth.append(batch.y[:batch.batch_size])
        text_out, graph_out = forward_one_batch(batch, model)
        out = text_out + graph_out
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return [accuracy_score(all_truth, all_preds) * 100,
            f1_score(all_truth, all_preds) * 100,
            precision_score(all_truth, all_preds) * 100,
            recall_score(all_truth, all_preds) * 100]


def train(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = BIC(hidden_dim=hidden_dim, num_layers=num_layers, des_dim=768, sta_dim=768,
                twe_dim=768, num_dim=5, cat_dim=1,
                num_heads=4, fixed_size=fixed_size,
                graph_mode='rgcn', semantic_mode='last',
                dropout=dropout).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
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
    model.load_state_dict(best_model)
    acc, f1, pre, rec = evaluate(model, test_loader)
    print('training done.')
    print('acc {:.2f} f1 {:.2f} pre {:.2f} rec {:.2f}'.format(acc, f1, pre, rec))
    torch.save(best_model, '{}/{:.2f}_{:.2f}_{:.2f}_{:.2f}.model'.format(save_path, acc, f1, pre, rec))


if __name__ == '__main__':
    while True:
        train('soft_states')
