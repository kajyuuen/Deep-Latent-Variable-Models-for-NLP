import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from src.models import Model
from src.models.simple_real_rnn import SimpleRealRNN

class NeuralTopicVector(Model):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 hidden_dim: int,
                 batch_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_of_class = num_of_class
        self.batch_size = batch_size

        self.rnn = SimpleRealRNN(self.vocab_size,
                                 self.num_of_class,
                                 hidden_dim)

    def model(self, data):
        num_of_sentence, sentence_length = data.shape

        rnn = pyro.module("rnn", self.rnn)        
        mu = pyro.param("mu", torch.rand(num_of_sentence, self.num_of_class))
        sigma = pyro.param("sigma", torch.rand(num_of_sentence, self.num_of_class),
                           constraint = constraints.positive)
        
        with pyro.plate("n", num_of_sentence, subsample_size = self.batch_size) as ind:
            z = pyro.sample("z",
                            dist.Normal(mu[ind], sigma[ind]).independent(1))
            h = None
            for t in pyro.plate('T', sentence_length):
                out, h = rnn(data[ind, t-1], z, h)
                x = pyro.sample(f"x_{t}",
                                dist.Categorical(logits=out),
                                obs = data[ind, t])
           
