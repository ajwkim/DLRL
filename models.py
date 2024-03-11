import numpy as np
import torch
from torch import nn
T, F = True, False

def linear_block(indim, outdim, leaky=F, batchnorm=F, dropoutp=.5):
    return (nn.Linear(indim, outdim),   nn.LeakyReLU() if leaky else nn.ReLU(),
            nn.BatchNorm1d(outdim) if batchnorm else nn.Dropout(dropoutp))

def linear_model(indim, outdim, hiddens, leaky=F, batchnorm=F, dropoutp=.5,
                 logsoftmax=F):
    dims = (indim,) + tuple(hiddens)
    blocks = tuple()
    for ind, outd in zip(dims, dims[1:]):
        blocks += linear_block(ind, outd, leaky, batchnorm, dropoutp)
    final = (nn.Linear(dims[-1], outdim),)
    if logsoftmax: final += (nn.LogSoftmax(dim=-1),)
    model = nn.Sequential(*blocks, *final)
    print('\nmodel:', model)
    return model


def conv_block(in_channel, out_channel):
    return (nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel))

def conv_classifier(in_shape, n_class, channels, logsoftmax=F):     # in_shape=(1, 28, 28)
    if len(in_shape) == 2:      in_shape = (1,) + tuple(in_shape)
    # channels = (in_shape[0],) + (32, 64, 128, 256, 512)
    blocks = tuple()
    for ch1, ch2 in zip(channels, channels[1:]):
        blocks += conv_block(ch1, ch2)
    outdim = get_conv_outdim(nn.Sequential(*blocks), in_shape)
    final = (nn.Flatten(),
             nn.Linear(outdim, 50), nn.ReLU(), nn.BatchNorm1d(50),
             nn.Linear(50, n_class))
    if logsoftmax: final += (nn.LogSoftmax(dim=-1),)
    model = nn.Sequential(*blocks, *final)
    print('\nmodel:', model)
    return model

def get_conv_outdim(model, in_shape):
    if len(in_shape) == 2:      in_shape = (1,) + tuple(in_shape)
    with torch.no_grad():
        out = model(torch.zeros(2, *in_shape))
    outdim = np.prod(out.detach().shape[1:])
    return int(outdim)