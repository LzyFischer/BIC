
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

starttime = datetime.now()

path0 = Path('20_22/data2/whr/TwiBot22-baselines/datasets/Twibot-22')
path1 = Path('lzy/bot-22/data-processing')

print_file = open(path1 / 'print.txt', 'w+')

edge = pd.read_csv(path0 / 'edge.csv')
endtime = datetime.now()
print("read edge time is"  + str(endtime - starttime), file=print_file)
starttime = datetime.now()
post = edge[edge.relation == 'post']
post = post.reset_index(drop=True)
endtime = datetime.now()
print('find post time is ' + str(endtime - starttime), file=print_file)
starttime = datetime.now()
user = pd.read_json(path0 / 'user.json')
endtime = datetime.now()
print('read user time is ' + str(endtime - starttime), file=print_file)
starttime = datetime.now()
label = pd.read_csv(path0 / 'label.csv')
endtime = datetime.now()
print('read label time is ' + str(endtime - starttime), file=print_file)

starttime = datetime.now()
user_index_to_uid = list(user.id)
uid_to_user_index = dict(map(reversed, enumerate(user_index_to_uid)))
post.source_id = list(map(lambda x: uid_to_user_index[x], post.source_id))
endtime = datetime.now()
print('shift index time is ' + str(endtime - starttime), file=print_file)

starttime = datetime.now()
tweet_0 = pd.read_json(path0 / "tweet_0.json")
endtime = datetime.now()
print('read tweet time is ' + str(endtime - starttime), file=print_file)

starttime = datetime.now()
user_tweet = {i:[] for i in range(len(label))}
user_tweet_count = {i: 0 for i in range(len(label))}
for i in tqdm(range(10)):
    user_index = post.source_id[i]
    if user_tweet_count[user_index] <= 200:
        user_tweet_count[user_index] += 1
        text = tweet_0[tweet_0.id == post.target_id[i]].text
        user_tweet[user_index].append(text)
endtime = datetime.now()
print("process time is " + str(endtime - starttime), file=print_file)

np.save(path1 / 'data/user_tweet.npy')
np.save(path1 / 'data/user_tweet_count.npy')