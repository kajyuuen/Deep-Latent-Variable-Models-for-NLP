import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from src.models.model import Model

class MixtureOfMultinomials(Model):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 batch_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.num_of_class = num_of_class
        self.batch_size = batch_size
    
    def model(self, data):
        num_of_sentence, sentence_length = data.shape
        # NOTE: constraints.simplexってなに？
        mu = pyro.param("mu", torch.rand(self.num_of_class),
                         constraint = constraints.simplex)
        pi = pyro.param("pi", torch.rand(self.num_of_class, self.vocab_size),
                         constraint = constraints.simplex)
        
        with pyro.plate("n", num_of_sentence, subsample_size = self.batch_size) as ind:
            z = pyro.sample("z",
                            dist.Categorical(mu.expand(self.batch_size, self.num_of_class)),
                            infer = dict(enumerate="parallel"))

            for t in pyro.plate('T', sentence_length):
                x = pyro.sample(f"x_{t}",
                                dist.Categorical(pi[z]),
                                obs = data[ind, t])
           
