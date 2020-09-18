import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from src.models import Model
from src.models.mutli_rnn import MultiRNN

class MixtureOfRNNs(Model):
    def __init__(self,
                 vocab_size: int,
                 num_of_class: int,
                 batch_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_of_class = num_of_class
        self.batch_size = batch_size

        self.rnn = MultiRNN(self.vocab_size,
                            self.num_of_class,
                            20)

    def model(self, data):
        num_of_sentence, sentence_length = data.shape
        rnn = pyro.module("rnn", self.rnn)
        mu = pyro.param("mu", torch.rand(self.num_of_class),
                         constraint = constraints.simplex)
        
        with pyro.plate("n", num_of_sentence, subsample_size = self.batch_size) as ind:
            z = pyro.sample("z",
                            dist.Categorical(mu.expand(self.batch_size, self.num_of_class)),
                            infer = dict(enumerate="sequential"))
            h = None
            for t in range(sentence_length):
                out, h = rnn(data[ind, t-1], z, h)
                pyro.sample(f"x_{t}",
                                dist.Categorical(logits=out),
                                obs = data[ind, t])
           
