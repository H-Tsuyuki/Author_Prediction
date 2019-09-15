import time
import random
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Rank 
from utils import DataIter

bat = progressbar.ProgressBar()

def train_epoch(model, loss_fn, train_iter, optim, device, epoch_i):
    ''' Epoch operation in training phase'''

    model.to(device)
    model.train()

    total_loss = 0
    total_acc = 0
    start = time.time()
    for i, batch in enumerate(train_iter):
        # prepare data
        title, doc, cand, t = batch["title"].to(device), batch["doc"].to(device), batch["cand"].to(device), batch["t"].to(device)

        # forward
        optim.zero_grad()
        y = model(title, doc, cand)
        t = t.squeeze(1)
        if random.random()*epoch_i > 10000000000000000000000000000000000000000000000000000000:
            gold = model.make_gold(cand, t)
            loss = F.kl_div(y, gold)
        else:
            loss = loss_fn(y, t)
        #from IPython.core.debugger import Pdb; Pdb().set_trace() 
        acc = (torch.max(y, 1)[1].data == t.data).sum().item()/len(title)
        # backward
        loss.backward()

        # update parameters
        optim.step()

        if i%100==0:
         print('  - (Training) {}  loss: {loss: 8.5f}, acc: {acc:3.3f} %, '\
              'elapse: {elapse:3.3f} sec'.format(
                  i,loss=loss, acc=100*acc,
                  elapse=(time.time()-start)))


def eval_epoch(model, loss_fn, valid_iter, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_acc = 0
    count = 0

    with torch.no_grad():
        for batch in valid_iter:
            count+=1
            # prepare data
            title, doc, cand, t = batch["title"].to(device), batch["doc"].to(device), batch["cand"].to(device), batch["t"].to(device)

            # forward
            y = model(title, doc, cand)
            t = t.squeeze(1)
            loss = loss_fn(y, t)
            acc = (torch.max(y, 1)[1].data == t.data).sum().item()/len(title)

            # note keeping
            total_loss += loss.item()
            total_acc += acc

    accuracy = total_acc/count
    loss = total_loss/count
    return loss, accuracy

def train(device):
    ''' Start training '''
    vocab = torch.load("vocab.pt")
    train_data = torch.load("train.pt")
    valid_data = torch.load("valid.pt")
    author_vocab = torch.load("author_vocab.pt")
    valid_iter = DataIter(valid_data, 32, False)
    model = Rank(len(author_vocab), len(vocab), device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss_fn = nn.CrossEntropyLoss()

    valid_accus = []
    for epoch_i in range(10):
        print('[ Epoch', epoch_i, ']')
        train_iter = DataIter(train_data, 64, True)

        train_epoch(model, loss_fn, train_iter, optim, device, epoch_i)

        start = time.time()
        valid_loss, valid_acc = eval_epoch(model, loss_fn, valid_iter, device)
        print('  - (Validation) loss: {loss: 8.5f}, acc: {acc:3.3f} %, '\
                'elapse: {elapse:3.3f} sec'.format(
                    loss=valid_loss, acc=100*valid_acc,
                    elapse=(time.time()-start)))

        valid_accus += [valid_acc]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch_i}

        #if valid_acc >= max(valid_accus):
        torch.save(checkpoint, "model.pt")
        #print('    - [Info] The checkpoint file has been updated.')

def main():
    device = torch.device('cuda:3')
    train(device)

if __name__ == '__main__':
    main()
