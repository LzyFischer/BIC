import torch
import numpy as np
from model import get_batch_text, BIC
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

no_up_limit = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


label = torch.load('../data/label.pt')
tweet = torch.load('../data/tweet.pt')
description = torch.load('../data/description.pt')
numerical = torch.load('../data/numerical.pt')
categorical = torch.load('../data/categorical.pt')
edge_index = torch.load('../data/edge_index.pt')
edge_type = torch.load('../data/edge_type.pt')
status = torch.load('../data/text.pt')
status = status[:, :200, :]
# status = np.load('labeled_status.npy', allow_pickle=True)
uid = torch.arange(label.shape[0])
data = Data(x=uid, y=label, edge_index=edge_index, edge_type=edge_type)
train_mask = torch.load('../data/train.pt')
val_mask = torch.load('../data/val.pt')
test_mask = torch.load('../data/test.pt')

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
        out = forward_one_batch(batch, model)
        loss = loss_fn(out, batch_label)
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
        out = forward_one_batch(batch, model)
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return [accuracy_score(all_truth, all_preds) * 100,
            f1_score(all_truth, all_preds) * 100,
            precision_score(all_truth, all_preds) * 100,
            recall_score(all_truth, all_preds) * 100]


def train():
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
    torch.save(best_model, 'model_states_new/{:.2f}_{:.2f}_{:.2f}_{:.2f}.model'.format(acc, f1, pre, rec))


if __name__ == '__main__':
    while True:
        train()
