import pyro

from src.data_modules.yahoo_data_module import YahooDataModule

from src.models.mixture_of_multinomials import MixtureOfMultinomials
from src.guides.amortized_discrete import AmortizedDiscrete

from src.inference.inference import Inference

def main():
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
                          optimizer = optimizer,
                          elbo = elbo)
    inference.optimize(train_iter)

if __name__ == "__main__":
    main()
