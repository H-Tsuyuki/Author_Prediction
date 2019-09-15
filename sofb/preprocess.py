import json
import torch
import MeCab
import progressbar
import random
bar = progressbar.ProgressBar()

def read_data():
    with open("data/train_aozora_data.json", 'r', encoding="utf-8") as f:
        all_idx = json.load(f)
    with open("data/aozora_titles.json", 'r', encoding="utf-8") as f:
        title = json.load(f)
    with open("data/aozora_contents.json", "r", encoding="utf-8") as f:
        contents = json.load(f)
    with open("data/train_q.json", "r", encoding="utf-8") as f:
        quest = json.load(f)
#    with open("data/test_q.json", "r", encoding="utf-8") as f:
#        test_quest = json.load(f)
#    with open("data/test_aozora_data.json", 'r', encoding="utf-8") as f:
#            test_title2content = json.load(f)

    return all_idx, title, contents, quest#, test_quest, test_title2content

def preprocess(all_idx_dict, title_dict, contents_dict, quest, n):
    data = []
    sents = []
    authors = []

    for author_novels in bar(all_idx_dict, max_value=len(all_idx_dict)):
        author_id = author_novels["author_id"]
        authors.append(author_id) 
        for novel in author_novels["novels"]:
            title_id = novel["title_id"]
            content_id = novel["content_id"]

            title = title_dict[title_id]
            content_all = contents_dict[content_id].replace("\n","").replace("\r","").replace("\u3000","") 
            if content_all=='': 
                content_all='<pad>'
            content_all = split_n_words(content_all, n)
            title = split_n_words(title, n)[0]
            
            sents.extend(content_all) 
            
            for content in content_all:
                data_per_content = {"author":author_id,
                                    "title":title,
                                    "content":content}
                for q in quest:
                    if q["title_id"] == title_id:
                        assert q["answer"] == author_id, "Error"
                        data_per_content["cand"] = q["candidates"]
                if "cand" in data_per_content.keys():    
                    data.append(data_per_content)
                else:
                    continue
    return data, sents, authors

def split_n_words(content_all, n):
    m=MeCab.Tagger()
    a = m.parse(content_all)
    words = [i.split("\t")[0] for i in a.split('\n')][:-2]
    nwords_list = list(split_list(words, n))
    return nwords_list

def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def build_vocab_idx(word_insts, vocab_size):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {"<unk>": 0,
                "<pad>": 1,
        }

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1
    word_count = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
    ignored_word_count = 0
    for word, count in word_count:
        if word not in word2idx:
#            if count > min_word_count:
            word2idx[word] = len(word2idx)
            if len(word2idx)==vocab_size:
                break
            #else:
            #    ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)))
          #'each with minimum occurrence = {}'.format(min_word_count))
    #print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def build_author_idx(author):
    author2idx = {}
    for i, a in enumerate(author):
        author2idx[a] = i
    print('[Info] Trimmed author size = {},'.format(len(author2idx)))
    return author2idx


def to_idx(data, author_vocab, vocab):
    for i, d in enumerate(data):
        data[i]["author"] = author_vocab[d["author"]]
        data[i]["cand"] = [author_vocab[j] for j in d["cand"]]
        data[i]["title"] = [vocab.get(w,0) for w in d["title"]]
        data[i]["content"] = [vocab.get(w,0) for w in d["content"]]
    
    return data

def main():
    all_idx, title, contents, quest = read_data()
    data, sents, authors = preprocess(all_idx, title, contents, quest, n=500)
    vocab = build_vocab_idx(sents, 100000)
    author_vocab = build_author_idx(authors)
    
    data = to_idx(data, author_vocab, vocab)
    random.shuffle(data) 
    train, valid = data[:-1000], data[-1000:]
    torch.save(vocab, "vocab.pt")
    torch.save(author_vocab, "author_vocab.pt")
    torch.save(train, "train.pt")
    torch.save(valid, "valid.pt")

if __name__=='__main__':
    main()
