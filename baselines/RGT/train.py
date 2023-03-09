import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from model import RGTDetector
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['TwiBot20', 'Cresci15']

if dataset_name == 'TwiBot20':
    data_path = '../../TwiBot20'
else:
    data_path = '../../Cresci15/data'

batch_size = 256
linear_channels = 128
out_channel = 128
dropout = 0.5
trans_head = 8
semantic_head = 8
lr = 1e-3
weight_decay = 3e-5

no_up_limit = 8

label = torch.load('{}/label.pt'.format(data_path))
tweet = torch.load('{}/tweet.pt'.format(data_path))
description = torch.load('{}/description.pt'.format(data_path))
numerical = torch.load('{}/numerical.pt'.format(data_path))
categorical = torch.load('{}/categorical.pt'.format(data_path))
edge_index = torch.load('{}/edge_index.pt'.format(data_path))
edge_type = torch.load('{}/edge_type.pt'.format(data_path))
uid = torch.arange(label.shape[0])
data = Data(x=uid, y=label, edge_index=edge_index, edge_type=edge_type)
print(data)

train_mask = torch.load('{}/train.pt'.format(data_path))
val_mask = torch.load('{}/val.pt'.format(data_path))
test_mask = torch.load('{}/test.pt'.format(data_path))

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
    val_loader = NeighborLoader(data,
                                num_neighbors=[256] * 4,
                                input_nodes=val_mask,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 4,
                                 input_nodes=test_mask,
                                 batch_size=batch_size,
                                 shuffle=False)
    proportions = [80]
    for p in proportions:
        size = int(len(train_mask) / 100 * p)
        train_loader = NeighborLoader(data,
                                      num_neighbors=[256] * 4,
                                      input_nodes=train_mask[:size],
                                      batch_size=batch_size,
                                      shuffle=True)
        train('TwiBot_model_states_{}'.format(p))


