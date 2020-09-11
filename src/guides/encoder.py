import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)
        self.mu = nn.Linear(hidden_dim, num_of_class)
        self.sigma = nn.Linear(hidden_dim, num_of_class)

    def forward(self, x):
        lstm_out = self.lstm(self.embedding_layer(x))[1][0]
        return self.mu(lstm_out), self.sigma(lstm_out).exp()
