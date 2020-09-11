import torch
import torch.nn as nn

import pyro

from src.models import Model
from src.guides import Guide

class Inference(nn.Module):
    def __init__(self,
                 p: Model,
                 q: Guide,
                 use_cuda: bool = False) -> None:
        super().__init__()
        self.p = p
        self.q = q

        if use_cuda:
            self.cuda()
    
    def guide(self, data):
        pyro.module("q", self.q)
        self.q.guide(data)

    def model(self, data):
        pyro.module("p", self.p)
        self.p.model(data)