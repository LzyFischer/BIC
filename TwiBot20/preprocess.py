import json
import ijson
import pandas
import time
import os.path as osp
from tqdm import tqdm
import torch
import datetime
import numpy as np
# from transformers import pipeline
# from transformers import RobertaTokenizer, AutoConfig, AutoModel
#
#
# mode = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(mode, do_lower_case=True, model_max_length=64)
# config = AutoConfig.from_pretrained(mode)
# model = AutoModel.from_pretrained(mode)
#
# pipe = pipeline('feature-extraction', config=config, model=model,
#                 tokenizer=tokenizer, framework='pt', device=0,
#                 truncation=True, padding=True, add_special_tokens=True)


def get_description():
    path = '../dataset/Twibot-20'
    description = []
    data = json.load(open(osp.join(path, 'node.json')))
    for item in tqdm(data, ncols=0, total=229580):
        if item['id'].find('u') != 0:
            continue
        if item['description'] is None:
            item['description'] = ''
        out = pipe(item['description'])
        out = torch.tensor(out[0], dtype=torch.float).mean(dim=0)
        description.append(out.to('cpu'))
    description = torch.stack(description)
    print(description.shape)
    torch.save(description, 'description.pt')


def get_tweet():
    author_tweets_dict = json.load(open('author_tweets_dict.json'))
    uids = json.load(open('uids.json'))
    tweets_list = []
    for index, item in enumerate(tqdm(uids, ncols=0)):
        if item in author_tweets_dict:
            tweets = author_tweets_dict[item]
        else:
            tweets = []
        tweets = tweets[:200]
        out = pipe(tweets)
        out = [torch.tensor(item[0], dtype=torch.float).mean(dim=0) for item in out]
        for _ in range(len(out), 1):
            out.append(torch.zeros(768, dtype=torch.float))
        out = torch.stack(out)
        out = out.to('cpu')
        tweets_list.append(out)
    torch.save(tweets_list, 'tweet.pt')


def calc_activate_day(created_at):
    if created_at is None:
        return 0
    created_at = created_at.strip()
    mod = '%a %b %d %H:%M:%S %z %Y'
    begin = datetime.datetime.strptime(created_at, mod)
    end = datetime.datetime.strptime('2022 02 01 +0000', '%Y %m %d %z')
    diff = end - begin
    return diff.days


def get_profile():
    path = '../dataset/Twibot-20'
    data = ijson.items(open(osp.join(path, 'node.json')), 'item')
    numerical = []
    categorical = []
    func_n = lambda x: int(x) if x is not None else 0
    func_c = lambda x: int(x.strip() == 'True') if x is not None else 0
    for item in tqdm(data, ncols=0, total=229580):
        if item['id'].find('u') != 0:
            break
        num = [func_n(item['public_metrics']['followers_count']),
               func_n(item['public_metrics']['following_count']),
               func_n(item['public_metrics']['tweet_count']),
               func_n(item['public_metrics']['listed_count']),
               calc_activate_day(item['created_at']),
               len(item['username']) if item['username'] is not None else 0]
        cat = [func_c(item['protected']),
               func_c(item['verified']),
               int(item['profile_image_url'].find('default_profile_normal') != -1)
               if item['profile_image_url'] is not None else 0]
        numerical.append(num)
        categorical.append(cat)
    numerical = np.array(numerical)
    categorical = np.array(categorical)
    numerical = (numerical - numerical.mean(0, keepdims=True)) / numerical.std(0, keepdims=True)
    print(numerical)
    print(numerical.shape)
    print(categorical.shape)
    numerical = torch.tensor(numerical, dtype=torch.float)
    categorical = torch.tensor(categorical, dtype=torch.float)
    torch.save(numerical, 'numerical.pt')
    torch.save(categorical, 'categorical.pt')


def get_label():
    path = '../dataset/Twibot-20'
    data = pandas.read_csv(osp.join(path, 'label.csv'))
    uids = json.load(open('uids.json'))
    label_index = {item[1]: int(item[2] == 'bot') for item in data.itertuples()}
    label = []
    for item in uids:
        if item in label_index:
            label.append(label_index[item])
        else:
            label.append(2)
    label = torch.tensor(label, dtype=torch.long)
    torch.save(label, 'label.pt')


def get_split():
    path = '../dataset/Twibot-20'
    data = pandas.read_csv(osp.join(path, 'split.csv'))
    uids = json.load(open('uids.json'))
    uids = {item: index for index, item in enumerate(uids)}
    train = []
    val = []
    test = []
    for item in data.itertuples():
        if item[2] == 'train':
            train.append(uids[item[1]])
        elif item[2] == 'val':
            val.append(uids[item[1]])
        elif item[2] == 'test':
            test.append(uids[item[1]])
        else:
            continue
    train = torch.tensor(train, dtype=torch.long)
    val = torch.tensor(val, dtype=torch.long)
    test = torch.tensor(test, dtype=torch.long)
    torch.save(train, 'train.pt')
    torch.save(val, 'val.pt')
    torch.save(test, 'test.pt')


def combine():
    data = torch.load('tweet.pt')
    statue = []
    tweet = []
    for item in tqdm(data, ncols=0):
        tweet.append(item.mean(0))
        statue.append(item.numpy())
    tweet = torch.stack(tweet)
    statue = np.array(statue, dtype=object)
    print(tweet.shape)
    print(len(statue))
    torch.save(tweet, 'new_tweet.pt')
    np.save('status.npy', statue)


def get_edge():
    path = '../dataset/Twibot-20'
    data = pandas.read_csv(osp.join(path, 'edge.csv'))
    print(data.head())
    uids = json.load(open('uids.json'))
    uids = {item: index for index, item in enumerate(uids)}
    edge_index = []
    edge_type = []

    for item in data.itertuples():
        if item[2] == 'post':
            continue
        edge_index.append([uids[item[1]], uids[item[3]]])
        edge_type.append(int(item[2] == 'friend'))

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.permute(1, 0)
    print(edge_index.shape)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    print(edge_type.shape)
    torch.save(edge_index, 'edge_index.pt')
    torch.save(edge_type, 'edge_type.pt')


if __name__ == '__main__':
    get_profile()