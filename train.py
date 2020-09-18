import torch
import pyro

from omegaconf import DictConfig
import hydra

import matplotlib.pylab as plt

from src.data_modules.yahoo_data_module import YahooDataModule

from src.models.mixture_of_multinomials import MixtureOfMultinomials
from src.models.mixture_of_rnns import MixtureOfRNNs
from src.models.neural_topic_vector import NeuralTopicVector

from src.guides.amortized_discrete import AmortizedDiscrete
from src.guides.amortized_normal import AmortizedNormal
from src.guides.guide_blank import GuideBlank

from src.optimize.optimize import optimize
from src.inference.inference import Inference

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    use_cuda = cfg.use_cuda
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    num_of_class = 10

    data_module = YahooDataModule(path = hydra.utils.to_absolute_path(cfg.path.dataset))
    data_module.setup()
    train_iter = data_module.train_dataloader()

    # Select model
    if cfg.inference.model == "mixture_of_multinomials":
        p = MixtureOfMultinomials(vocab_size = data_module.vocab_size,
                                  num_of_class = num_of_class)
    elif cfg.inference.model == "mixture_of_rnns":
        p = MixtureOfRNNs(vocab_size = data_module.vocab_size,
                          num_of_class = num_of_class)
    elif cfg.inference.model == "neural_topic_vector":
        p = NeuralTopicVector(vocab_size = data_module.vocab_size,
                              num_of_class = num_of_class,
                              hidden_dim = 10)
    else:
        raise NotImplementedError
    
    # Select Guide
    if cfg.inference.guide == "amortized_discrete":
        q = AmortizedDiscrete(vocab_size = data_module.vocab_size,
                              num_of_class = num_of_class)
    elif cfg.inference.guide == "amortized_normal":
        q = AmortizedNormal(vocab_size = data_module.vocab_size,
                            num_of_class = num_of_class)
    elif cfg.inference.guide == "blank":
        q = GuideBlank()
    else:
        raise NotImplementedError
    
    optimizer = pyro.optim.Adam({"lr": 1.0e-3})
    elbo = pyro.infer.Trace_ELBO()
    # elbo = pyro.infer.TraceEnum_ELBO(max_iarange_nesting=1, 
    #                                    strict_enumeration_warning=True)
    inference = Inference(p = p,
                          q = q,
                          use_cuda = use_cuda)

    losses = optimize(inference.model,
                      inference.guide,
                      optimizer,
                      train_iter,
                      elbo,
                      epoch = 1000,
                      use_cuda = use_cuda)
    
    # Print result
    plt.scatter(x=list(range(len(losses))), y=losses)
    plt.ylim(0, 1000)
    plt.savefig("{}.png".format(cfg.inference.model))

    if cfg.inference.model == "mixture_of_multinomials":
        val, args = pyro.param("pi").cpu().sort(dim=1)
    elif cfg.inference.model == "mixture_of_rnns":
        val, args = pyro.param("mu").cpu().sort(dim=1)
    elif cfg.inference.model == "neural_topic_vector":
        val, args = pyro.param("mu").cpu().sort(dim=1)
    for j in range(10):
        print("State", j)
        for i in args[j, -20:]:
            print(data_module.text.vocab.itos[i], end=" ")
        print("")

if __name__ == "__main__":
    main()
