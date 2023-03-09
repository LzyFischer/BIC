import json
import pandas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


if __name__ == '__main__':
    uids = json.load(open('covid_uids.json'))
    label = pandas.read_csv('data/label.csv')

    train_size = int(len(uids) * 0.8)
    uids = uids[train_size:]

    scores = json.load(open('Twibot-22.json'))
    preds = {item['id']: int(item['english'] >= 0.75) for item in scores}

    all_truth = []
    all_preds = []

    for item in label.itertuples():
        if item[1].replace('u', '') in uids:
            all_preds.append(preds[item[1]] if item[1] in preds else 1)
            all_truth.append(int(item[2] == 'bot'))
    print(accuracy_score(all_truth, all_preds) * 100,
          f1_score(all_truth, all_preds) * 100,
          precision_score(all_truth, all_preds) * 100,
          recall_score(all_truth, all_preds) * 100)

