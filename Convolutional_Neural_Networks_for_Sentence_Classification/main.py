import os
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from konlpy.tag import Okt
from data import Corpus
from model import SeNet
from gluonnlp.data import PadSequence
from tqdm import tqdm
from preprocessing import read_data, remove_na
from sklearn.model_selection import train_test_split

with open('./config.json') as io:
    params = json.loads(io.read())

with open(params['filepath'].get('vocab'), mode='rb') as io:
    vocab = pickle.load(io)

# creating model
model = SeNet(num_classes=params['num_classes'], vocab=vocab)

# creating dataset, dataloader
tagger = Okt()
padder = PadSequence(length=30)

# load data
data = read_data(params['filepath'].get('tr'))
data = remove_na(data)
tr_data, val_data = train_test_split(data, test_size=0.2)
print('data: {}, tr_data:{}, val_data:{}'.format(len(data), len(tr_data), len(val_data)))

# dataset
batch_size = params['training'].get('batch_size')
tr_dataset = Corpus(tr_data, vocab, tagger, padder)
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_dataset = Corpus(val_data, vocab, tagger, padder)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# training
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(params=model.parameters(), lr=params['training'].get('lr'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

epochs = params['training'].get('epochs')

for epoch in tqdm(range(epochs), desc='epochs'):
    avg_tr_loss = 0
    avg_val_loss = 0
    tr_step = 0
    val_step = 0

    model.train()
    for x_mb, y_mb in tqdm(tr_dataloader, desc='iters'):
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)
        score = model(x_mb)

        opt.zero_grad()
        tr_loss = loss_fn(score, y_mb)
        reg_term = torch.norm(model.fc.weight, p=2)
        tr_loss.add_(.5 * reg_term)
        tr_loss.backward()
        opt.step()

        avg_tr_loss += tr_loss.item()
        tr_step += 1
    else:
        avg_tr_loss /= tr_step

    model.eval()
    for x_mb, y_mb in tqdm(val_dataloader):
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)

        with torch.no_grad():
            score = model(x_mb)
            val_loss = loss_fn(score, y_mb)
            avg_val_loss += val_loss.item()
            val_step += 1
    else:
        avg_val_loss /= val_step

    tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, avg_tr_loss, avg_val_loss))

ckpt = {'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'vocab': vocab}

savepath = params['filepath'].get('ckpt')
torch.save(ckpt, savepath)
