import torch
from tqdm import tqdm
from transformers import pipeline
import json
import sys

each_user_tweets = json.load(open("lzy/bot-detection/src/data/id_tweet.json", 'r'))
pipe = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=1,
                padding=True, truncation=True, max_length=512)


def tweets_embedding():
    tweets_list = []
    for i in tqdm(range(0, len(each_user_tweets))):
        tweets = each_user_tweets[str(i)]
        tweets = [item for item in tweets if tweets is not None]
        tweets = tweets[:200]
        out = pipe(tweets)
        out = [torch.tensor(item[0], dtype=torch.float).mean(dim=0) for item in out]
        for _ in range(len(out), 200):
            out.append(torch.zeros(768, dtype=torch.float))
        out = torch.stack(out)
        tweets_list.append(out)
    
    outs = torch.stack(tweets_list)
    torch.save(outs, 'lzy/bot-detection/src/data/tweet_200.pt')
    tweets_list = []



if __name__ == '__main__':
    fp = open('lzy/bot-detection/process/log.txt', 'w')
    stderr = sys.stderr
    stdout = sys.stdout

    sys.stderr = fp
    sys.stdout = fp

    tweets_embedding()

    fp.close()
    sys.stderr = stderr
    sys.stdout = stdout