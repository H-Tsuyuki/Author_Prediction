import json
import torch
import MeCab
import progressbar
import random
bar = progressbar.ProgressBar()

def read_data():
    with open("data/aozora_titles.json", 'r', encoding="utf-8") as f:
        title = json.load(f)
    with open("data/aozora_contents.json", "r", encoding="utf-8") as f:
        contents = json.load(f)
    with open("data/test_q.json", "r", encoding="utf-8") as f:
        test_quest = json.load(f)
    with open("data/test_aozora_data.json", 'r', encoding="utf-8") as f:
        test_title2content = json.load(f)

    return test_title2content, title, contents, test_quest, 

def preprocess(title2content, title_dict, contents_dict, quest, n):
    data = []

    for q in bar(quest, max_value=len(quest)):
        data_per_quest = []
        title_id = q["title_id"]
        title = title_dict[title_id]
        content_id = title2content[title_id]
        content_all = contents_dict[content_id].replace("\n","").replace("\r","").replace("\u3000","")
        if content_all == '':
            content_all = '<pad>'
        content_all = split_n_words(content_all, n)
        title = split_n_words(title, n)[0]
        cand = q["candidates"]
        for content in content_all:
            data_per_content = {"title":title,
                                "content":content,
                                "cand":cand}
            data_per_quest.append(data_per_content)
        data.append(data_per_quest)
        
    return data

def split_n_words(content_all, n):
    m=MeCab.Tagger()
    a = m.parse(content_all)
    words = [i.split("\t")[0] for i in a.split('\n')][:-2]
    nwords_list = list(split_list(words, n))
    return nwords_list

def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def to_idx(data, author_vocab, vocab):
    for i, q in enumerate(data):
        for j, d in enumerate(q):
            data[i][j]["cand"] = [author_vocab[a] for a in d["cand"]]
            data[i][j]["title"] = [vocab.get(w,0) for w in d["title"]]
            data[i][j]["content"] = [vocab.get(w,0) for w in d["content"]]
    return data

def main():
    all_idx, title, contents, quest = read_data()
    data = preprocess(all_idx, title, contents, quest, n=500)
    vocab = torch.load("vocab.pt")
    author_vocab = torch.load("author_vocab.pt")
    
    data = to_idx(data, author_vocab, vocab)
    torch.save(data, "test.pt")

if __name__=='__main__':
    main()
