import json
import random

from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from model import InteractModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


batch_size = 256
dropout = 0.5
lr = 1e-3
weight_decay = 1e-5
no_up_limit = 16


path = 'data'
num_feature = torch.load('{}/num_properties_tensor.pt'.format(path))
cat_feature = torch.load('{}/cat_properties_tensor.pt'.format(path))
tweet_feature = torch.load('{}/tweets_tensor.pt'.format(path))
des_feature = torch.load('{}/des_tensor.pt'.format(path))
edge_index = torch.load('{}/edge_index.pt'.format(path))
edge_type = torch.load('{}/edge_type.pt'.format(path))
labels = torch.load('{}/node_label.pt'.format(path))
data = Data(torch.arange(0, 1000000), edge_index=edge_index, edge_type=edge_type)

test_uid, test_tweet = torch.load('{}/test.pt'.format(path))

test_index = torch.arange(0, 1000000)
test_index[test_uid] = torch.arange(0, test_uid.shape[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def forward_one_batch(model, batch):
    batch_uid = batch.x  # the global id
    batch_edge_index = batch.edge_index
    index_uid = test_index[batch.x[:batch.batch_size]]  # the split id

    tweet = test_tweet[index_uid]
    description = des_feature[batch.x[:batch.batch_size]]
    text = torch.cat([description.unsqueeze(1), tweet], dim=1)

    num_batch = num_feature[batch_uid]
    cat_batch = cat_feature[batch_uid]
    tweet_batch = tweet_feature[batch_uid]
    des_batch = des_feature[batch_uid]

    neighbor_index = []
    for i in range(batch.batch_size):
        neighbor_0 = batch_edge_index[0][batch_edge_index[1] == i].numpy()
        neighbor_1 = batch_edge_index[1][batch_edge_index[0] == i].numpy()
        neighbor = [i] + list(set(neighbor_1) | set(neighbor_0))
        neighbor_index.append(neighbor)
    return model(text.to(device),
                 neighbor_index,
                 num_batch.to(device),
                 cat_batch.to(device),
                 tweet_batch.to(device),
                 des_batch.to(device),
                 batch_edge_index.to(device))


def train_one_epoch(model, optimizer, loss_fn):
    model.train()
    ave_loss = 0
    for batch in tqdm(train_loader, ncols=0):
        optimizer.zero_grad()
        batch_label = labels[batch.x[:batch.batch_size]].to(device)
        out = forward_one_batch(model, batch)
        loss = loss_fn(out, batch_label)
        ave_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(ave_loss / len(train_loader))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        all_truth.append(labels[batch.x[:batch.batch_size]])
        out = forward_one_batch(model, batch)
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return [accuracy_score(all_truth, all_preds) * 100,
            f1_score(all_truth, all_preds) * 100,
            precision_score(all_truth, all_preds) * 100,
            recall_score(all_truth, all_preds) * 100]


def train():
    model = InteractModel(dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            torch.save(model.state_dict(), 'model_states/{:.2f}_{:.2f}_{:.2f}_{:.2f}.model'.format(acc, f1, pre, rec))
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
                                  num_neighbors=[256, 256, 256, 256],
                                  input_nodes=train_uids,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256, 256, 256, 256],
                                 input_nodes=test_uids,
                                 batch_size=batch_size,
                                 shuffle=False)
    while True:
        train()
