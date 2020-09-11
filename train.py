import torch
import pyro

from src.data_modules.yahoo_data_module import YahooDataModule

from src.models.mixture_of_multinomials import MixtureOfMultinomials
from src.guides.amortized_discrete import AmortizedDiscrete

from src.optimize.optimize import optimize
from src.inference.inference import Inference

def main():
    use_cuda = True
    if True:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    num_of_class = 10

    data_module = YahooDataModule(path = "./datasets/val.txt")
    data_module.setup()
    train_iter = data_module.train_dataloader()

    mixture_of_multinomials = MixtureOfMultinomials(vocab_size = data_module.vocab_size,
                                                    num_of_class = num_of_class)
    amortized_discrete = AmortizedDiscrete(vocab_size = data_module.vocab_size,
                                           num_of_class = num_of_class)

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

    # Print result
    val, args = pyro.param("pi").cpu().sort(dim=1)
    for j in range(10):
        print("State", j)
        for i in args[j, -20:]:
            print(data_module.text.vocab.itos[i], end=" ")
        print("")

if __name__ == "__main__":
    main()
