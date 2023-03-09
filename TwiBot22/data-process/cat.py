import torch
from pathlib import Path
from tqdm import tqdm
path = Path('lzy/bot-22/src/data')

des = torch.load(path / 'des_tensor.pt')
split = [50000]*20
des = torch.split(des, split)
for i in tqdm(range(20)):
    tweet = torch.load(path / f'tweet_{i}.pt')
    text = torch.cat([des[i].unsqueeze(1), tweet], dim=1)
    torch.save(text, path / f'text_{i}.pt')