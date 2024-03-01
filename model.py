import numpy as np
import torch
from torch import nn
T, F = True, False

def BlockLinear(indim, outdim):
    return nn.Sequential(nn.Linear(indim, outdim),  nn.ReLU())

def linear_regressor(indim, outdim, dim_step=-1):
    dims = list(range(indim, outdim - dim_step, dim_step))
    blocks = (BlockLinear(ind, outd) for ind, outd in zip(dims, dims[1:]))
    model = nn.Sequential(*blocks)
    return nn.Sequential(model, nn.Linear(dims[-1], outdim))


def BlockConv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channel),

        nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(out_channel))

def conv_classifier(in_shape, n_class):     # in_shape=(1, 28, 28)
    # in_shape = tuple(in_shape)
    if len(in_shape) == 2:      in_shape = (1,) + tuple(in_shape)
    channels = (in_shape[0],) + (32, 64, 128, 256, 512)
    blocks = (BlockConv(ch1, ch2) for ch1, ch2 in zip(channels, channels[1:]))
    model = nn.Sequential(*blocks)
    outdim = get_conv_outdim(model, in_shape)
    return nn.Sequential(model,
                         nn.Flatten(),
                         nn.Linear(outdim, 50),     nn.ReLU(),  nn.BatchNorm1d(50),
                         nn.Linear(50, n_class))

def get_conv_outdim(model, in_shape):
    if len(in_shape) == 2:      in_shape = (1,) + tuple(in_shape)
    with torch.no_grad():
        out = model(torch.zeros(2, *in_shape))
    outdim = np.prod(out.detach().shape[1:])
    return int(outdim)