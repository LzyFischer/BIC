import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime as dt
import json
print('loading raw data')
path1=Path('20_22/data2/whr/TwiBot22-baselines/datasets/Twibot-20')
path2=Path('lzy/bot-detection/src/data')

node=pd.read_json(path1 / 'node.json')
user = node[node.id.str.contains('^u') == True]
edge=pd.read_csv(path1 / 'edge.csv')
user_idx=user['id']

uid_index={uid:index for index,uid in enumerate(user_idx.values)}

tweets = node[node.id.str.contains('^t') == True].reset_index(drop=True)
tweet_idx = tweets['id']
tid_index = {tid:index for index,tid in enumerate(tweet_idx.values)}

post = edge[edge.relation == 'post'].reset_index(drop=True)
post.loc[:,'source_id'] = list(map(lambda x:uid_index[x], post.source_id))
post.loc[:,'target_id'] = list(map(lambda x:tid_index[x], post.target_id))

# print('extracting labels and splits')
# split=pd.read_csv(path1 / "split.csv")
# label=pd.read_csv(path1 / "label.csv")


print("extracting each_user's tweets")
id_tweet={i:[] for i in range(len(user_idx))}

for i, uidx in tqdm(enumerate(post.source_id.values)):
    tidx=post.target_id[i]
    try:
        id_tweet[uidx].append(tweets.text[tidx])
    except KeyError:
        print('wrong')
        break
        continue
json.dump(id_tweet,open(path2 / 'id_tweet.json','w'))



