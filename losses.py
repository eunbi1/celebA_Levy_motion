import torch
import copy
import time
import numpy as np
import tqdm
from scipy.special import gamma
import torchlevy
from torchlevy import LevyStable
import torch.nn.functional as F

levy = LevyStable()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_func(x):
    return torch.tensor(gamma(x))

from torchlevy.approx_score import get_approx_score

from torchlevy.approx_score import get_approx_score




def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_L: torch,
            num_steps=1000, type="cft", training_clamp=4, mode='approximation'):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)

    if sde.alpha == 2:

      score = -1 / 2 * (e_L)*torch.pow(sigma+1e-4,-1)

      #score = -1 / 2 * (e_L)

    else:
        if mode =='approximation':
         score =get_approx_score(e_L, sde.alpha).to(device)
        elif mode =='brownian':
         score = -1 / 2 * (e_L)
        elif mode =='resampling':
         score = levy.score(e_L, sde.alpha, type=type).to(device)
        elif mode == 'normal':
         score = levy.score(e_L, sde.alpha, type=type).to(device) 

    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]
    output = model(x_t, t)
    # loss = torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)
    #
    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #loss = F.smooth_l1_loss(output, score, size_average=False,reduce=True, beta=4.0)
    weight = output-score
    loss = (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)
    #loss = F.smooth_l1_loss(output, score, size_average=False, reduce=True, beta=4.0)

    return  loss
