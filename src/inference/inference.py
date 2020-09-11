import pyro
from pyro.optim import PyroOptim
from pyro.infer import SVI
from pyro.infer.elbo import ELBO

from src.models import Model
from src.guides import Guide

class Inference:
    def __init__(self,
                 p: Model,
                 q: Guide,
                 optimizer: PyroOptim,
                 elbo: ELBO) -> None:
        self.p = p
        self.q = q
        self.optimizer = optimizer
        self.elbo = elbo

    def optimize(self, data, epoch = 1000):
        pyro.clear_param_store()
        svi = pyro.infer.SVI(self.p.model,
                             self.q.guide,
                             self.optimizer,
                             loss = self.elbo)

        losses = []
        for _ in range(epoch):
            losses.append(svi.step(data.text.transpose(0, 1)) / 
                          (data.text.shape[1] * data.text.shape[0]))

        return losses

