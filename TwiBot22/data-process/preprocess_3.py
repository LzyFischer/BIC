import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

path1 = Path('20_22/data2/whr/TwiBot22-baselines/datasets/Twibot-22')
path2 = Path('lzy/bot-22/data-processing')

# edge = pd.read_csv(path1 / 'edge.csv')
label = pd.read_csv(path1 / 'label.csv')
print(label.id.size)
# user = pd.read_json(path1 / 'user.json')
# user_index_to_uid = list(user.id)
# uid_to_index = dict(map(reversed, enumerate(user_index_to_uid)))

# following = edge[edge['relation'] == 'following']
# following = following.reset_index(drop=True)
# followers = edge[edge['relation'] == 'followers']
# followers = followers.reset_index(drop=True)

# following.source_id = list(map(lambda x: uid_to_index[x], following.source_id))
# following.target_id = list(map(lambda x: uid_to_index[x], following.target_id))
# followers.source_id = list(map(lambda x: uid_to_index[x], followers.source_id))
# followers.target_id = list(map(lambda x: uid_to_index[x], followers.target_id))

# neighbor = {i:[i] for i in range(label.id.size)}
# for i in tqdm(range(following.target_id.size)):
#     if following.source_id[i] < label.id.size:
#         neighbor[following.source_id[i]].append(following.target_id[i])
#     if following.target_id[i] < label.id.size:
#         neighbor[following.target_id[i]].append(following.source_id[i])
# for i in tqdm(range(followers.target_id.size)):
#     if followers.source_id[i]< label.id.size:
#         neighbor[followers.source_id[i]].append(followers.target_id[i])
#     if followers.target_id[i] < label.id.size:
#         neighbor[followers.target_id[i]].append(followers.source_id[i])

# np.save(path2 / 'neighbor.npy', neighbor)