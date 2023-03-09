import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from model import RGTDetector
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import json


data_path = '../../covid/data'
batch_size = 512
linear_channels = 128
out_channel = 128
dropout = 0.5
trans_head = 8
semantic_head = 8
lr = 1e-3
weight_decay = 3e-5

no_up_limit = 16

label = torch.load('{}/node_label.pt'.format(data_path))
tweet = torch.load('{}/tweets_tensor.pt'.format(data_path))
description = torch.load('{}/des_tensor.pt'.format(data_path))
numerical = torch.load('{}/num_properties_tensor.pt'.format(data_path))
categorical = torch.load('{}/cat_properties_tensor.pt'.format(data_path))
edge_index = torch.load('{}/edge_index.pt'.format(data_path))
edge_type = torch.load('{}/edge_type.pt'.format(data_path))
uid = torch.arange(label.shape[0])
data = Data(x=uid, y=label, edge_index=edge_index, edge_type=edge_type)


uids = json.load(open('../../covid/data/uids.json'))
uids = {item: index for index, item in enumerate(uids)}
covid_uids = json.load(open('../../covid/covid_uids.json'))
train_size = int(len(covid_uids) * 0.8)
train_uids = torch.tensor([uids[item] for item in covid_uids[:train_size]], dtype=torch.long)
test_uids = torch.tensor([uids[item] for item in covid_uids[train_size:]], dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, optimizer, loss_fn):
    model.train()
    for batch in tqdm(train_loader, ncols=0):
        optimizer.zero_grad()
        batch_label = batch.y[:batch.batch_size].to(device)
        out = model(categorical[batch.x].to(device),
                    numerical[batch.x].to(device),
                    tweet[batch.x].to(device),
                    description[batch.x].to(device),
                    batch.edge_index.to(device),
                    batch.edge_type.to(device))[:batch.batch_size]
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
        out = model(categorical[batch.x].to(device),
                    numerical[batch.x].to(device),
                    tweet[batch.x].to(device),
                    description[batch.x].to(device),
                    batch.edge_index.to(device),
                    batch.edge_type.to(device))[:batch.batch_size]
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return [accuracy_score(all_truth, all_preds) * 100,
            f1_score(all_truth, all_preds) * 100,
            precision_score(all_truth, all_preds) * 100,
            recall_score(all_truth, all_preds) * 100]


def train(save_path='model_states'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = RGTDetector(numeric_num=numerical.shape[-1],
                        cat_num=categorical.shape[-1],
                        tweet_channel=tweet.shape[-1],
                        des_channel=description.shape[-1],
                        linear_channels=linear_channels,
                        out_channel=out_channel,
                        trans_head=trans_head,
                        semantic_head=semantic_head,
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
    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 4,
                                  input_nodes=train_uids,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 4,
                                 input_nodes=test_uids,
                                 batch_size=batch_size,
                                 shuffle=False)
    while True:
        train('covid_states')




