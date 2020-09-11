import torch
import pyro

from src.data_modules.yahoo_data_module import YahooDataModule

from src.models.mixture_of_multinomials import MixtureOfMultinomials
from src.guides.amortized_discrete import AmortizedDiscrete

from src.inference.inference import Inference

def optimize(model,
             guide,
             optimizer,
             data,
             elbo,
             epoch = 1000,
             use_cuda = False):
    pyro.clear_param_store()
    svi = pyro.infer.SVI(model,
                         guide,
                         optimizer,
                         loss = elbo)
    text = data.text

    if use_cuda:
        text = text.cuda()

    losses = []
    for _ in range(epoch):
        losses.append(svi.step(text.transpose(0, 1)) / 
                        (text.shape[1] * text.shape[0]))

    return losses


def main():
    use_cuda = True
    num_of_class = 10

    data_module = YahooDataModule(path = "./datasets/val.txt")
    data_module.setup()
    train_iter = data_module.train_dataloader()

    mixture_of_multinomials = MixtureOfMultinomials(vocab_size = data_module.vocab_size,
                                                    num_of_class = num_of_class).cuda()
    amortized_discrete = AmortizedDiscrete(vocab_size = data_module.vocab_size,
                                           num_of_class = num_of_class).cuda()

    optimizer = pyro.optim.Adam({"lr": 1.0e-3})
    elbo = pyro.infer.Trace_ELBO()
    
    inference = Inference(p = mixture_of_multinomials,
                          q = amortized_discrete,
                          use_cuda = use_cuda)

    optimize(inference.model,
             inference.guide,
             optimizer,
             train_iter,
             elbo,
             epoch = 1000,
             use_cuda = use_cuda)

if __name__ == "__main__":
    main()
