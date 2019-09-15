import random
import numpy as np

import torch
import torch.nn as nn


UNK = np.random.rand(1, 300)[0]
PAD = np.random.rand(1, 300)[0]

def get_ft(ft_path):
    word_vec = {}
    word_vec["<unk>"] = UNK.astype(np.float32)
    word_vec["<pad>"] = PAD.astype(np.float32)
    with open(ft_path) as f:
        for i, line in enumerate(f):
            if i==0: continue
            word, vec = line.split(' ', 1)
            if not word in word_vec:
                word_vec[word] = np.array(list(map(float, vec.split()))).astype(np.float32)
    
    torch.save(word_vec, "ftw_vec.pt")

#get_ft("/datac/tsuyuki/japanese_fasttext/cc.ja.300.vec")

def make_w_matrix():
    ft=torch.load("ftw_vec.pt")
    vocab=torch.load("vocab.pt")
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found=0
    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = ft[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = UNK
    print(words_found)
    torch.save(weights_matrix, "w_matrix.pt")

#make_w_matrix()

class DataIter(nn.Module):
    def __init__(self, data, batch_size, train):
        super(DataIter, self).__init__()
        self.data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        if train:
            print("Train Data Size: ",len(data))
            random.shuffle(self.data)
        else:
            print("Valid Data Size: ",len(data))
        print("total iter : ",len(data)//batch_size)
            
    
    def __iter__(self):
        for idx, batch in enumerate(self.data):
            batch_dict={}
            L_title = [len(i["title"]) for i in batch]
            L_content = [len(i["content"]) for i in batch]
            n_cand = [len(i["cand"]) for i in batch]
            tmp = [i["cand"] + [random.randint(0,869)] if len(i["cand"])!=max(n_cand) else i["cand"] for i in batch ]
            batch_dict["cand"] = torch.Tensor(tmp)
            batch_dict["t"] = torch.LongTensor([[idx for idx, a in enumerate(i["cand"]) if a==i["author"]] for i in batch ])
            batch_dict["title"] = torch.LongTensor([i["title"] + [1]*(max(L_title)-len(i["title"])) for i in batch])
            batch_dict["doc"] = torch.LongTensor([ i["content"] + [1]*(max(L_content)-len(i["content"])) for i in batch])
            yield batch_dict  

class TestDataIter(nn.Module):
    def __init__(self, data):
        super(TestDataIter, self).__init__()
        print("Number of Quest: ",len(data))
        print("total n_content : ",sum([len(i) for i in data]))
        self.data = data

    def __iter__(self):
        for idx, batch in enumerate(self.data):
            batch_dict={}
            L_title = [len(i["title"]) for i in batch]
            L_content = [len(i["content"]) for i in batch]
            n_cand = [len(i["cand"]) for i in batch]
            batch_dict["L_content"] = L_content
            tmp = [i["cand"] + [random.randint(0,869)] if len(i["cand"])!=max(n_cand) else i["cand"] for i in batch ]
            batch_dict["cand"] = torch.Tensor(tmp)
            batch_dict["title"] = torch.LongTensor([i["title"] + [1]*(max(L_title)-len(i["title"])) for i in batch])
            batch_dict["doc"] = torch.LongTensor([ i["content"] + [1]*(max(L_content)-len(i["content"])) for i in batch])
            yield batch_dict  
