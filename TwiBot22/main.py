import os
import os.path as osp
import ijson
import json
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import RobertaTokenizer, AutoConfig, AutoModel


mode = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(mode, do_lower_case=True, model_max_length=64)
config = AutoConfig.from_pretrained(mode)
model = AutoModel.from_pretrained(mode)
pipe = pipeline('feature-extraction', config=config, model=model,
                tokenizer=tokenizer, framework='pt', device=0,
                truncation=True, padding=True, add_special_tokens=True)


if __name__ == '__main__':
    author_tweets_dict = json.load(open('author_tweets_dict.json'))
    uids = json.load(open('uids.json'))
    with torch.no_grad():
        tweets_list = []
        for index, item in enumerate(tqdm(uids, ncols=0)):
            uid = item.replace('u', '')
            if uid not in author_tweets_dict:
                tweets = []
            else:
                tweets = author_tweets_dict[uid]
            tweets = tweets[:200]
            out = pipe(tweets)
            out = [torch.tensor(item[0], dtype=torch.float).mean(dim=0) for item in out]
            for _ in range(len(out), 1):
                out.append(torch.zeros(768, dtype=torch.float))
            out = torch.stack(out)
            out = out.to('cpu')
            tweets_list.append(out)
            if len(tweets_list) == 25000:
                file_id = index // 25000
                torch.save(tweets_list, 'tweet_{:02}.pt'.format(file_id))
                tweets_list = []

