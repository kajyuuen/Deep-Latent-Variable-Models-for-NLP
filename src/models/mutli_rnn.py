import torch
import torch.nn as nn

class MultiRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.E = nn.Embedding(vocab_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, vocab_size)
        self.W = nn.Parameter(torch.zeros(num_of_class, hidden_dim, hidden_dim))
        self.U = nn.Parameter(torch.zeros(num_of_class, hidden_dim, hidden_dim))
 
        self.hidden_dim = hidden_dim

    def forward(self, x, z, h):
        if h is None:
            batch_size = z.shape[1]
            h_0 = torch.zeros(batch_size, 1, self.hidden_dim)
            return self.V(h_0), h_0
        e = self.E(x)
        h_t = torch.tanh(torch.einsum("nckh,ch->nck", [self.W[z], e]) 
                         + torch.einsum("nckh,cnh->nck",[self.U[z], h])).transpose(0,1)
        return self.V(h_t), h_t