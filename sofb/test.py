import json
import time
import progressbar
import collections as cl

import torch
import torch.nn as nn

from model import Rank 
from utils import TestDataIter


bat = progressbar.ProgressBar()

def test(model, test_iter, device, vocab, author_vocab):
    ''' Epoch operation in evaluation phase '''
    with open("data/aozora_titles.json", 'r', encoding="utf-8") as f:
        title_dict = json.load(f)
        title2id = {v:k for k, v in title_dict.items()}
    with open("data/test_q.json", 'r', encoding="utf-8") as f:
        test_q = json.load(f)
        title_id = [q["title_id"] for q in test_q]

    model.to(device)
    model.eval()
    results=[]
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            if idx%100==0:
                print(idx)
            result=cl.OrderedDict()
            # prepare data
            title, doc, cand = batch["title"].to(device), batch["doc"].to(device), batch["cand"].to(device)
            L_content = batch["L_content"]
            # forward
            y = model(title, doc, cand)
            y = torch.Tensor(L_content).unsqueeze(1).to(device)*y 
            y = torch.sum(y,0)


            y = torch.sort(y, descending=True)[1].cpu()
            tmp = torch.index_select(batch["cand"][0], 0, y).long().tolist()           
            #title = "".join([vocab[i] for i in batch["title"][0].tolist()])          
            result["title_id"] = title_id[idx]          
            result["candidates"] = [author_vocab[i] for i in tmp]          
            results.append(result)
    return results

def train(device):
    ''' Start training '''
    vocab = torch.load("vocab.pt")
    test_data = torch.load("test.pt")
    author_vocab = torch.load("author_vocab.pt")
    test_iter = TestDataIter(test_data)
   
    idx2word = {v:k for k, v in vocab.items()}
    idx2author_id = {v:k for k, v in author_vocab.items()}

    checkpoint = torch.load("model.pt")
    model = Rank(len(author_vocab), len(vocab), device)
    model.load_state_dict(checkpoint["model"])
    results = test(model, test_iter, device, idx2word, idx2author_id)

    with open("tsuyuki00.json", "w") as f:
        json.dump(results,f, indent=2)
    

def main():
    device = torch.device('cuda:3')
    train(device)

if __name__ == '__main__':
    main()
