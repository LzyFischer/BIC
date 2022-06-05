import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from src.model import InteractModel
from src.pytorchtools import EarlyStopping
#from ablation_study import model_only_graph, model_only_text, model_no_interact, model_mean_interact

"""
bot->0, human->1
"""

def record_highest(num_added, in_list):
    for i in range(len(in_list)):
        if num_added > in_list[i]:
            if i == 0:
                in_list[i] == num_added
            else:
                in_list[i-1] = in_list[i]
                in_list[i] = num_added

def eval(all_confusion):
    acc = (all_confusion[0][0] + all_confusion[1][1]) / np.sum(all_confusion)
    precision = all_confusion[1][1] / (all_confusion[1][1] + all_confusion[0][1])
    recall = all_confusion[1][1] / (all_confusion[1][1] + all_confusion[1][0])
    f1 = (2 * precision * recall) / (precision + recall)
    return acc, f1, precision, recall
    


path1 = Path('src/data')
path2 = Path('src/state_dict')

split = [[], [], []]
split_list = pd.read_csv(path1 / 'split.csv')
label = pd.read_csv(path1 / 'label.csv')

users_index_to_uid = list(label['id'])
uid_to_users_index = {x : i for i, x in enumerate(users_index_to_uid)}
for id in split_list[split_list['split'] == 'train']['id']:
    split[0].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'val']['id']:
    split[1].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'test']['id']:
    split[2].append(uid_to_users_index[id])

def my_collate(batch):
    text = torch.stack([item[0] for item in batch])
    user_neighbor_index = [item[1] for item in batch]
    user_label = torch.stack([item[2] for item in batch]).type(torch.LongTensor)
    num_feature = batch[0][3]
    cat_feature = batch[0][4]
    tweet_feature = batch[0][5]
    des_feature = batch[0][6]
    edge_index = batch[0][7]
    return [text, user_neighbor_index, user_label, num_feature, cat_feature, tweet_feature, des_feature, edge_index]

class InterDataset(Dataset):
    def __init__(self, name='train'):
        super(InterDataset, self).__init__()
        self.text = torch.load(path1 / 'text.pt') #11826
        self.user_neighbor_index = np.load(path1 / 'user_neighbor_index.npy', allow_pickle=True).tolist() # 11826
        self.user_label = torch.load(path1 / 'node_label.pt')
        self.num_feature = torch.load(path1 / 'num_properties_tensor.pt') #229580
        self.cat_feature = torch.load(path1 / 'cat_properties_tensor.pt') #229580
        self.tweet_feature = torch.load(path1 / 'tweets_tensor.pt')
        self.des_feature = torch.load(path1 / 'des_tensor.pt')
        self.edge_index = torch.load(path1 / 'edge_index.pt') #check if tensor
        
        if name == 'train':
            self.text = self.text[split[0]]
            self.user_neighbor_index = [self.user_neighbor_index[split[0][i]] for i in range(len(split[0]))]
            self.user_label = self.user_label[split[0]]
            self.length = len(self.user_label)
        if name == 'val':
            self.text = self.text[split[1]]
            self.user_neighbor_index = [self.user_neighbor_index[split[1][i]] for i in range(len(split[1]))]
            self.user_label = self.user_label[split[1]]
            self.length = len(self.user_label)
        if name == 'test':
            self.text = self.text[split[2]]
            self.user_neighbor_index = [self.user_neighbor_index[split[2][i]] for i in range(len(split[2]))]
            self.user_label = self.user_label[split[2]]
            self.length = len(self.user_label)
        
        # self.text = torch.randn(16, 200, 768) #11826
        # self.user_neighbor_index = {i:[i] for i in range(16)} # 11826
        # self.user_label = torch.LongTensor([1,0,1,0,1,1,1,0,1,1,0,0,1,0,1,1])
        # self.num_feature = torch.randn(16, 5) #229580
        # self.cat_feature = torch.randn(16, 3) #229580
        # self.edge_index = torch.tensor([[1, 2], [2, 1]])
        # self.length = len(self.text)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.text[index], self.user_neighbor_index[index], self.user_label[index], self.num_feature, self.cat_feature, self.tweet_feature, self.des_feature, self.edge_index

class InterTrainer:
    def __init__(self, train_loader, val_loader, test_loader, model=InteractModel, optimizer=torch.optim.Adam,
                 lr=1e-4, weight_decay=1e-5, scheduler=ReduceLROnPlateau, dropout=0.5, num_epochs=100, early_stopping_patience=20, 
                 state_dict_path='state_dict_only_graph.tar', device='cuda:1'):
    # def __init__(self, train_loader, val_loader, test_loader, optimizer=torch.optim.Adam, lr=1e-4,
    #             weight_decay=1e-5, dropout=0.2, num_epochs=10, device='cuda:2'):
        self.device = device
        self.state_dict_path = state_dict_path
        
        self.num_epochs = num_epochs
        self.model = model(device=self.device, dropout=dropout)
        # self.model = nn.TransformerEncoderLayer(768, 4)
        self.model.to(device)
        # self.test_linear = nn.Linear(768, 2).to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = scheduler(self.optimizer, mode='max', verbose=True, factor=0.1, patience=5, eps=1e-8)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        self.loss_func = nn.CrossEntropyLoss()

        self.highest_5_acc = [0,0,0,0,0]
        self.highest_5_f1 = [0,0,0,0,0]
        
    def train(self):
        max_acc, corres_f1, model_state_dict = 0, 0, None
        epoch_loss = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            batch_quantity = 0
            with tqdm(self.train_loader) as progress_bar:
                for batch in progress_bar:
                    text = batch[0].to(self.device)
                    user_neighbor_index = batch[1]
                    user_label = batch[2].to(self.device)
                    num_feature = batch[3].to(self.device)
                    cat_feature = batch[4].to(self.device)
                    tweet_feature = batch[5].to(self.device)
                    des_feature = batch[6].to(self.device)
                    edge_index = batch[7].to(self.device)
                    pred = self.model(text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index)
                    loss = self.loss_func(pred, user_label)
                    epoch_loss += loss.item()
                    batch_quantity += 1
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())

            epoch_loss /= batch_quantity
            print(f'epoch_loss={epoch_loss}')

            val_loss, val_acc = self.val()
            self.scheduler.step(val_acc)
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                a = open('result.txt', 'a')
                a.write(' '.join(list(map(lambda x:str(x), self.highest_5_acc))+['\n']))
                a.write(' '.join(list(map(lambda x:str(x), self.highest_5_f1))+['\n\n']))
                print("Early stopping")
                break

            test_acc, corres_f1, recall, precision, model_state_dict = self.test()

            
            print(f'Test_Accuracy: {test_acc}', end=' ')
            print(f'Precision:{precision}', end=' ')
            print(f'Recall:{recall}', end=' ')
            print(f'F1:{corres_f1}')
            

            if max_acc < val_acc:
                a = open('result.txt', 'a')
                a.write(str(test_acc) + '\n')
                a.write(str(corres_f1) + '\n\n')
                a.close()

                max_acc = val_acc
                if not os.path.exists(path2 / self.state_dict_path):
                    torch.save({'acc': test_acc, 'f1': corres_f1, 'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_state_dict': model_state_dict}, path2 / self.state_dict_path)
                else:
                    best_model = torch.load(path2 / self.state_dict_path)
                    if best_model['acc'] < max_acc:
                        torch.save({'acc': test_acc, 'f1': corres_f1, 'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dict}, path2 / self.state_dict_path)
    
    @torch.no_grad()                
    def val(self):
        self.model.eval()
        batch_quantity, epoch_loss = 0, 0
        all_confusion = np.zeros([2,2])
        for batch in self.val_loader:
            text = batch[0].to(self.device)
            user_neighbor_index = batch[1]
            user_label = batch[2].to(self.device)
            num_feature = batch[3].to(self.device)
            cat_feature = batch[4].to(self.device)
            tweet_feature = batch[5].to(self.device)
            des_feature = batch[6].to(self.device)
            edge_index = batch[7].to(self.device)
            pred = self.model(text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index)
            loss = self.loss_func(pred, user_label)
            pred, user_label = pred.argmax(dim=-1).cpu().numpy(), user_label.cpu().numpy()
            epoch_loss += loss
            batch_quantity += 1
            all_confusion += confusion_matrix(user_label, pred)

        epoch_loss /= batch_quantity
        acc, f1, precision, recall = eval(all_confusion)
        # print(f'Val_Accuracy: {acc}', end=' ')
        # print(f'Precision:{precision}', end=' ')
        # print(f'Recall:{recall}', end=' ')
        # print(f'F1:{f1}')
        print(f'val_loss:{epoch_loss}')

        a = open('result.txt', 'a')
        a.write(str(epoch_loss) + '\n')
        a.close()
        
        return epoch_loss, acc
    
    @torch.no_grad()                
    def test(self):
        self.model.eval()
        all_confusion = np.zeros([2,2])
        for batch in self.test_loader:
            text = batch[0].to(self.device)
            user_neighbor_index = batch[1]
            user_label = batch[2].numpy()
            num_feature = batch[3].to(self.device)
            cat_feature = batch[4].to(self.device)
            tweet_feature = batch[5].to(self.device)
            des_feature = batch[6].to(self.device)
            edge_index = batch[7].to(self.device)
            pred = self.model(text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index).argmax(dim=-1).cpu().numpy()
            all_confusion += confusion_matrix(user_label, pred)
        
        acc, f1, precision, recall = eval(all_confusion)

        record_highest(acc, self.highest_5_acc)
        record_highest(f1, self.highest_5_f1)
        
        model_state_dict = self.model.state_dict()
        
        return acc, f1, recall, precision, model_state_dict
                    
BATCH_SIZE = 64

if __name__ == '__main__':
    train_dataset = InterDataset('train')
    val_dataset = InterDataset('val')
    test_dataset = InterDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    
    for i in range(30):
        trainer = InterTrainer(train_loader, val_loader, test_loader, model=InteractModel, optimizer=torch.optim.RAdam,
                     lr=1e-4, weight_decay=1e-5, scheduler=ReduceLROnPlateau, dropout=0.5, num_epochs=30, early_stopping_patience=10, 
                     state_dict_path='state_dict.tar', device='cuda:1')
        trainer.train()
