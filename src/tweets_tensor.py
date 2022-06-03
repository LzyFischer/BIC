import os
import json
import torch
os.environ['CUDA_VISIBLE_DEVICE'] = '3'
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import *
pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
model = RobertaModel.from_pretrained(pretrained_weights)
feature_extractor = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=3, padding=True, truncation=True, max_length=50)

def padding_tweets(user_tweets_tensor):
    dif = 200 - len(user_tweets_tensor)
    if dif > 0:
        tmp = torch.zeros(dif, 768)
        user_tweets_tensor = torch.cat((user_tweets_tensor, tmp), dim=0)
    elif dif < 0:
        user_tweets_tensor = user_tweets_tensor.resize_(200, 768)
    return user_tweets_tensor

users_tweets = np.load('data/user_tweets_dict.npy', allow_pickle=True).tolist()

tweets_tensor = []
for i in tqdm(range(2000, 4000)):
    user_tweets_tensor = []
    try:
        for tweet in users_tweets[i]:
            tweet_tensor = torch.tensor(feature_extractor(tweet)).squeeze(0)
            tweet_tensor = torch.mean(tweet_tensor, dim=0) 
            user_tweets_tensor.append(tweet_tensor)
        user_tweets_tensor = torch.stack(user_tweets_tensor)
        user_tweets_tensor = padding_tweets(user_tweets_tensor)
        print(user_tweets_tensor.shape)
    except:
        user_tweets_tensor = torch.zeros(200, 768)
    tweets_tensor.append(user_tweets_tensor)
tweets_tensor = torch.stack(tweets_tensor)

path3 = Path('src/data')
torch.save(tweets_tensor, path3 / 'tweets_tensor_2000.pt')
