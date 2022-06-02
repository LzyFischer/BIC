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
from src.case_model2 import InteractModel
from src.pytorchtools import EarlyStopping

path1 = Path('src/data')
path2 = Path('src/state_dict')
label = pd.read_csv(path1 / 'label.csv')
human_list = label[label.label == 'human'].index
bot_list = label[label.label == 'bot'].index

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
        self.length = len(self.user_label)

        if name == 'human':
            self.text = self.text[human_list]
            self.user_neighbor_index = [self.user_neighbor_index[human_list[i]] for i in range(len(human_list))]
            self.user_label = self.user_label[human_list]
            self.length = len(self.user_label)
        if name == 'bot':
            self.text = self.text[bot_list]
            self.user_neighbor_index = [self.user_neighbor_index[bot_list[i]] for i in range(len(bot_list))]
            self.user_label = self.user_label[bot_list]
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


class AttentionWeight:
    def __init__(self, all_loader=None, human_loader=None, bot_loader=None, model=InteractModel, dropout=0.5, 
                state_dict_path='state_dict.tar', device='cuda:0', result_path='result.txt'):
        self.device = device
        self.model_state_dict=torch.load(path2 / state_dict_path)['model_state_dict']
        self.result_path = result_path
        
        self.model = model(device=self.device, dropout=dropout)
        self.model.to(device)
        
        self.human_loader = human_loader
        self.bot_loader = bot_loader
        self.all_loader = all_loader
    
    def save_human(self):
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()
        att_weight_0, att_weight_1 = [], []
        for batch in self.human_loader:
            text = batch[0].to(self.device)
            user_neighbor_index = batch[1]
            user_label = batch[2].to(self.device)
            num_feature = batch[3].to(self.device)
            cat_feature = batch[4].to(self.device)
            tweet_feature = batch[5].to(self.device)
            des_feature = batch[6].to(self.device)
            edge_index = batch[7].to(self.device)
            attention_weight_0, attention_weight_1 = self.model(text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index)
            att_weight_0.append(attention_weight_0)
            att_weight_1.append(attention_weight_1)
        
        att_weight_0 = torch.cat(att_weight_0)
        att_weight_1 = torch.cat(att_weight_1)
    
    def save_bot(self):
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()
        att_weight_0, att_weight_1 = [], []
        for batch in self.bot_loader:
            text = batch[0].to(self.device)
            user_neighbor_index = batch[1]
            user_label = batch[2].to(self.device)
            num_feature = batch[3].to(self.device)
            cat_feature = batch[4].to(self.device)
            tweet_feature = batch[5].to(self.device)
            des_feature = batch[6].to(self.device)
            edge_index = batch[7].to(self.device)
            attention_weight_0, attention_weight_1 = self.model(text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index)
            att_weight_0.append(attention_weight_0)
            att_weight_1.append(attention_weight_1)
        
        att_weight_0 = torch.cat(att_weight_0)
        att_weight_1 = torch.cat(att_weight_1)
        
        torch.save(att_weight_0, 'att_weight_0_b.pt')
        torch.save(att_weight_1, 'att_weight_1_b.pt')

    def save_all(self):
        all_1, all_2, all_3, all_4 = [], [], [], []

        print(self.model_state_dict['attention_linear.weight'].shape)
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()
        att_weight_0, att_weight_1 = [], []
        for batch in tqdm(self.all_loader):
            text = batch[0].to(self.device)
            user_neighbor_index = batch[1]
            user_label = batch[2].to(self.device)
            num_feature = batch[3].to(self.device)
            cat_feature = batch[4].to(self.device)
            tweet_feature = batch[5].to(self.device)
            des_feature = batch[6].to(self.device)
            edge_index = batch[7].to(self.device)
            a1, a2, a3, a4 = self.model(text, user_neighbor_index, num_feature, cat_feature, tweet_feature, des_feature, edge_index)
            a1 = a1.detach().cpu()
            a2 = a2.detach().cpu()
            a3 = a3.detach().cpu()
            a4 = a4.detach().cpu()
            all_1.append(a1)
            all_2.append(a2)
            all_3.append(a3)
            all_4.append(a4)        
        all_1 = torch.cat(all_1)
        all_2 = torch.cat(all_2)
        all_3 = torch.cat(all_3)
        all_4 = torch.cat(all_4)
        
        torch.save(all_1, 'all_1.pt')
        torch.save(all_2, 'all_2.pt')
        torch.save(all_3, 'all_3.pt')
        torch.save(all_4, 'all_4.pt')



    

BATCH_SIZE = 16

if __name__ == '__main__':
#    human_dataset = InterDataset('human')
#    bot_dataset = InterDataset('bot')
    all_dataset = InterDataset('all')
    
#    human_loader = DataLoader(human_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)
#    bot_loader = DataLoader(bot_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)
    all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)

    A = AttentionWeight(all_loader=all_loader)

    A.save_all()
