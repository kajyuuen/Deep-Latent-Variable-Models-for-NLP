import pyro

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
