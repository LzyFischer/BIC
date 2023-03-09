import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
from pathlib import Path
import pandas as pd
import json

path1 = Path('20_22/data2/whr/TwiBot22-baselines/datasets/Twibot-22')
path2 = Path('lzy/bot-22/data-processing/processed_data')
f = open(path2 / 'print.txt', 'w')

user=pd.read_json(path1 / 'user.json')

each_user_tweets=json.load(open(path2 / "id_tweet.json",'r'))

# device = 'cuda:0'
feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base', device=1, padding=True, truncation=True,max_length=50, add_special_tokens = True)

def tweets_embedding():
        print('Running feature2 embedding')
        path=path2 / "tweets_tensor.pt"
        if True:
            tweets_list=[]
            for i in tqdm(range(0, 1000)):
                if i % 1000 == 0:
                    print(i, file=f)
                if len(each_user_tweets[str(i)])==0:
                    total_each_person_tweets=torch.zeros(20, 768)
                else:
                    total_each_person_tweets = []
                    for j in range(len(each_user_tweets[str(i)])):
                        each_tweet=each_user_tweets[str(i)][j]
                        if each_tweet is None:
                            total_word_tensor=torch.zeros(1, 768)
                        else:
                            each_tweet_tensor=torch.tensor(feature_extract(each_tweet)).detach()
                            for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                                if k==0:
                                    total_word_tensor=each_word_tensor
                                else:
                                    total_word_tensor+=each_word_tensor
                            total_word_tensor = (total_word_tensor / each_tweet_tensor.shape[1]).unsqueeze(0)
                        if j==20:
                            total_each_person_tweets = torch.cat(total_each_person_tweets).unsqueeze(0)
                            break
                        else:
                            total_each_person_tweets.append(total_word_tensor)
                    if len(each_user_tweets[str(i)]) < 20:
                        for l in range(20-len(total_each_person_tweets)):
                            total_each_person_tweets.append(torch.zeros(1, 768))
                        total_each_person_tweets = torch.cat(total_each_person_tweets).unsqueeze(0)                        
                tweets_list.append(total_each_person_tweets)
                if (i+1) % 50000 == 0:
                    tweet_tensor=torch.cat(tweets_list)
                    torch.save(tweet_tensor, path2 / f"tweets_tensor/tweets_tensor_{int(i / 50000)}.pt")
                    tweets_list = []
        else:
            tweets_tensor=torch.load(path)
        print('Finished')

# Des_embbeding()
tweets_embedding()