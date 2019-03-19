import os
import torch
import json
from data import Corpus
from model import SeNet
from torch.utils.data import DataLoader
from konlpy.tag import Okt
from gluonnlp.data import PadSequence
from tqdm import tqdm
from preprocessing import read_data, remove_na

with open('./config.json') as io:
    params = json.loads(io.read())
    print(params)

# restoring model
savepath = params['filepath'].get('ckpt')
ckpt = torch.load(savepath)

vocab = ckpt['vocab']

model = SeNet(num_classes=params['num_classes'], vocab=vocab)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# create dataset, dataloader
tagger = Okt()
padder = PadSequence(length=30)
tst_data = read_data(params['filepath'].get('tst'))
tst_data = remove_na(tst_data)
tst_dataset = Corpus(tst_data, vocab, tagger, padder)
tst_dataloader = DataLoader(tst_dataset, batch_size=128)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# evaluation
correct_count = 0
for x_mb, y_mb in tqdm(tst_dataloader):
    x_mb = x_mb.to(device)
    y_mb = y_mb.to(device)
    with torch.no_grad():
        y_mb_hat = model(x_mb)
        y_mb_hat = torch.max(y_mb_hat, 1)[1]
        correct_count += (y_mb_hat == y_mb).sum().item()

print('Acc : {:.2%}'.format(correct_count / len(tst_dataset)))
