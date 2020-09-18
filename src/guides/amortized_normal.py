import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from src.guides import Guide
from src.guides.encoder import Encoder

class AmortizedNormal(Guide):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 batch_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_of_class = num_of_class
        self.batch_size = batch_size
    
        self.encoder = Encoder(self.vocab_size,
                               self.num_of_class,
                               hidden_dim = 50)

    def guide(self, data):
        num_of_sentence, _ = data.shape
        encoder = pyro.module("encoder", self.encoder)        

        with pyro.plate("n", num_of_sentence, subsample_size = self.batch_size) as ind:
            mu, sigma = encoder(data[ind])
            z = pyro.sample("z", dist.Normal(mu, sigma).independent(1))