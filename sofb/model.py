import torch
import torch.nn as nn
import torch.nn.functional as F


class Rank(nn.Module):
    def __init__(self, author_vocab_size, vocab_size, device, emb_dim=512):
        super(Rank, self).__init__()
        self.fc1 = nn.Linear(emb_dim*2, emb_dim)        
#        self.bow_emb = BOW(10000, 4000, emb_dim, vocab_size, device)
        self.cnn_emb = CNN(emb_dim, vocab_size, device)
#        self.rnn_emb = BiLSTM_MAX(emb_dim, vocab_size, device)
        self.author_emb = Author(emb_dim, author_vocab_size, device)
        self.device = device
    def forward(self, title, doc, authors):

#        title_bow = self.bow_emb(title)
#        doc_bow = self.bow_emb(doc)
        title_cnn = self.cnn_emb(title)
        doc_cnn = self.cnn_emb(doc)
#        title_rnn = self.rnn_emb(title)
#        doc_rnn = self.rnn_emb(doc)
#        out = torch.cat([title_cnn, doc_cnn, title_rnn, doc_rnn], 1) 
        out = torch.cat([title_cnn, doc_cnn], 1) 
        out = self.fc1(out)
        a_vec = self.author_emb(authors)
        y = torch.bmm(a_vec, out.unsqueeze(2)) #32*11*512, 32*512*1
        y=y.squeeze(2)
        y = F.log_softmax(y, dim=-1)
        return y

    def make_gold(self, authors, t):
        a_vec = self.author_emb(authors).detach()
        g = torch.bmm(a_vec, a_vec.transpose(1,2)) #32*11*512, 32*512*11
        g = torch.cat([g[i][t[i]].view(-1,1) for i in range(len(t))], 1).transpose(0,1)
        return g 

class Author(nn.Module):
    def __init__(self, emb_dim, vocab_size, device):
        super(Author, self).__init__()
        self.fc = nn.Linear(vocab_size, emb_dim)        
        self.vocab_size = vocab_size        
        self.device = device        
    
    def forward(self, bow_vec):
        bow_vec = torch.zeros(bow_vec.size(0), bow_vec.size(1), self.vocab_size,device=self.device).scatter(2, bow_vec.unsqueeze(2).long(), 1)
        emb=bow_vec.view(-1,self.vocab_size)
        emb = self.fc(emb)
        emb = emb.view(bow_vec.size(0),bow_vec.size(1),-1)
        return emb


class BOW(nn.Module):
    def __init__(self, m1_dim, m2_dim, emb_dim, vocab_size, device):
        super(BOW, self).__init__()
        self.vocab_size = vocab_size        
        self.device = device        
        self.fc1 = nn.Linear(vocab_size, m1_dim)        
        self.fc2 = nn.Linear(m1_dim, m2_dim)        
        self.fc3 = nn.Linear(m2_dim, emb_dim)        
    
    def forward(self, idx):
        bow_vec = self.make_bow_vector(idx)
        emb = F.relu(self.fc1(bow_vec))
        emb = F.relu(self.fc2(emb))
        emb = F.relu(self.fc3(emb))
        return emb

    def make_bow_vector(self, sents):
        vec = torch.zeros(len(sents), self.vocab_size)
        for i, sent in enumerate(sents):
            for word in sent: 
                vec[i,word] +=1
        return vec.to(self.device)

class CNN(nn.Module):
    def __init__(self, emb_dim, vocab_size, device):
        super(CNN, self).__init__()
        weights = torch.load("w_matrix.pt")
        print("Vocab_Size:",  vocab_size)
        weights = torch.Tensor(weights).to(device)
        output_size = emb_dim
        in_channels = 1
        out_channels = 300
        kernel_heights = [3,4,5]
        stride = 1
        padding = (2,0)
        embedding_length = 300
	
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length, 1)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(len(kernel_heights)*out_channels, output_size)
	
    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
		
        return max_out
	
    def forward(self, input_sentences, batch_size=None):
	
        input = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)
		
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
	# all_out.size() = (batch_size, num_kernels*out_channels)
        out = self.dropout(all_out)
	# fc_in.size()) = (batch_size, num_kernels*out_channels)
        out = F.relu(self.out(out))	
        return out

class BiLSTM_MAX(nn.Module):
    def __init__(self, emb_size, vocab_size, device):
        super(BiLSTM_MAX, self).__init__()
        
        weights = torch.load("w_matrix.pt")
        weights = torch.Tensor(weights).to(device)
        self.output_size = emb_size
        self.hidden_size = emb_size
        self.vocab_size = vocab_size
        self.embedding_length = 300
        
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2*self.hidden_size, self.output_size)
    
    def forward(self, input_sentence):
    
        self.batch_size = len(input_sentence)
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
       	#h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).to(device))
       	#c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).to(device))
        out, (final_hidden_state, final_cell_state) = self.lstm(input)
        out = self.dropout(out)
        out = torch.max(out.transpose(0,1), 1)[0]
        out = F.relu(self.fc(out))
        
        return out
