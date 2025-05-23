import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init(self,d_model:int,vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return(self.embedding(x)*math.sqrt(self.d_model))

    class PositionalEncoding(nn.Module):
        # positional encoding gives positional info of text to the model
        def __init(self,d_model:int,seq_len:int,dropout:float)->None:
            super().__init__()
            self.d_model = d_model
            self.seq_len = seq_len
            self.dropout = nn.Dropout(dropout)

            #create a matrix of shape(seq_len, d_model)
            pe= torch.zeros(seq_len,d_model)
            position=torch.arrange(0,seq_len,dtype=torch.float).unsqueeze(1) # we create a tensor of shape = sequence and length 1
            # denom term for calculating positional encoding for each element in position vector
            div_term = torch.exp(torch.arrange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) # build in log space of numerical stability
            # apply sin to even positions
            pe[:,0::2] = torch.sin(position*div_term)
            # apply cosine to odd terms
            pe[:,1::2] = torch.cos(position*div_term)

            pe = pe.unsqueeze(0)

            self.register_buffer('pe',pe)

        def forward(self,x):
            #we need to add positional encoding to every word in sentence
            # we dont need to teach the model positional encoding, this is to to be trained
            # hence requires_grad_ =False
            x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
            return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) #multiplied
        self.bias=nn.Parameter(torch.zeros(1)) #added

    def forward(self,x):
        mean=x.mean(dim =-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std*self.eps) +self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout: float)-> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout =nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        #(Batch,Seq_Len,d_model)->(Batch,Seq_Len,d_ff)->(Batch,Seq_Len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model: int, h:int,dropout: float)->None:
        super().__init__()
        self.d_model=d_model
        self.h=h # d_model should be divisible by h same vector needs to go to each head
        self.dropout=nn.Dropout(dropout)
        assert d_model %h==0, 'd_model is not divisible by h'

        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model) #Wq
        self.w_k=nn.Linear(d_model,d_model) #Wk
        self.w_v=nn.Linear(d_model,d_model) #Wv

        self.w_o=nn.Linear(d_model,d_model)#Wo
        self.dropout = nn.Dropout(dropout)









