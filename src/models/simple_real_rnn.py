import torch
import torch.nn as nn

class SimpleRealRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.E = nn.Embedding(vocab_size, self.hidden_dim)
        self.V = nn.Linear(self.hidden_dim, vocab_size)
        self.W = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.U = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Q = nn.Linear(num_of_class, self.hidden_dim)
 
    def forward(self, x, z, h):
        if h is None:
            h_0 = torch.zeros(1, self.hidden_dim)
            return self.V(h_0), h_0
        e = self.E(x)
        h_t = torch.tanh(self.W(e) 
                         + self.U(h)
                         + self.Q(z))
        return self.V(h_t), h_t        
 