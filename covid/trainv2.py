import torch
import os
from modelv2 import get_batch_text, BIC
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

batch_size = 256
lr = 1e-3
weight_decay = 1e-5
hidden_dim = 256
num_layers = 2
dropout = 0.0
fixed_size = 16

no_up_limit = 8
semantic_mode = 'last'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = 'data'
num_feature = torch.load('{}/num_properties_tensor.pt'.format(path))
cat_feature = torch.load('{}/cat_properties_tensor.pt'.format(path))
tweet_feature = torch.load('{}/tweets_tensor.pt'.format(path))
des_feature = torch.load('{}/des_tensor.pt'.format(path))
edge_index = torch.load('{}/edge_index.pt'.format(path))
edge_type = torch.load('{}/edge_type.pt'.format(path))
labels = torch.load('{}/node_label.pt'.format(path))
data = Data(torch.arange(0, 1000000), y=labels, edge_index=edge_index, edge_type=edge_type)

test_uid, test_tweet = torch.load('{}/test.pt'.format(path))

test_index = torch.arange(0, 1000000)
test_index[test_uid] = torch.arange(0, test_uid.shape[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def forward_one_batch(batch, model):
    size = batch.batch_size
    batch_edge_index = batch.edge_index.to(device)
    batch_edge_type = batch.edge_type.to(device)
    batch_tweet = tweet_feature[batch.x].to(device)
    batch_description = des_feature[batch.x].to(device)
    batch_numerical = num_feature[batch.x].to(device)
    batch_categorical = cat_feature[batch.x].to(device)
    index_uid = test_index[batch.x[:batch.batch_size]]
    batch_status = test_tweet[index_uid]
    text, padding_mask = get_batch_text(batch_status)
    text, padding_mask = text.to(device), padding_mask.to(device)
    return model(text, padding_mask,
                 batch_edge_index, batch_edge_type,
                 batch_tweet, batch_description,
                 batch_numerical, batch_categorical, size)


def train_one_epoch(model, optimizer, loss_fn):
    model.train()
    for batch in tqdm(train_loader, ncols=0):
        optimizer.zero_grad()
        batch_label = batch.y.to(device)[:batch.batch_size]
        out = forward_one_batch(batch, model)
        loss = loss_fn(out, batch_label)
        loss.backward()
        optimizer.step()


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


def train(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = BIC(hidden_dim=hidden_dim, num_layers=num_layers, des_dim=768, sta_dim=768,
                twe_dim=768, num_dim=5, cat_dim=3,
                num_heads=4, fixed_size=fixed_size,
                graph_mode='rgcn', semantic_mode=semantic_mode,
                dropout=dropout).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0
    no_up = 0
    epoch = 0
    while True:
        train_one_epoch(model, optimizer, loss_fn)
        acc, f1, pre, rec = evaluate(model, test_loader)
        if acc > best_acc:
            no_up = 0
            best_acc = acc
            torch.save(model.state_dict(), '{}/{:.2f}_{:.2f}_{:.2f}_{:.2f}.model'.format(save_path, acc, f1, pre, rec))
        else:
            no_up += 1
        if no_up == no_up_limit:
            break
        print('epoch {}, no up: {}, best acc: {}'.format(epoch, no_up, best_acc))
        epoch += 1


if __name__ == '__main__':
    uids = json.load(open('data/uids.json'))
    uids = {item: index for index, item in enumerate(uids)}
    covid_uids = json.load(open('covid_uids.json'))
    train_size = int(len(covid_uids) * 0.8)
    train_uids = torch.tensor([uids[item] for item in covid_uids[:train_size]], dtype=torch.long)
    test_uids = torch.tensor([uids[item] for item in covid_uids[train_size:]], dtype=torch.long)
    train_loader = NeighborLoader(data,
                                  num_neighbors=[-1] * 4,
                                  batch_size=batch_size,
                                  input_nodes=train_uids,
                                  shuffle=True)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[-1] * 4,
                                 batch_size=batch_size,
                                 input_nodes=test_uids,
                                 shuffle=False)
    while True:
        train('model_states_v2')
