import numpy as np
import torch
from torch import nn
T, F = True, False

def LinearBlock(indim, outdim, leaky=F, batchnorm=F, dropoutp=.5):
    return (nn.Linear(indim, outdim),   nn.LeakyReLU() if leaky else nn.ReLU(),
            nn.BatchNorm1d(outdim) if batchnorm else nn.Dropout(dropoutp))

def linear_model(indim, outdim, hiddens, leaky=F, batchnorm=F, dropoutp=.5,
                 logsoftmax=F):
    dims = (indim,) + tuple(hiddens)
    blocks = tuple()
    for ind, outd in zip(dims, dims[1:]):
        blocks += LinearBlock(ind, outd, leaky, batchnorm, dropoutp)
    model =\
        nn.Sequential(*blocks, nn.Linear(dims[-1], outdim), nn.LogSoftmax(dim=-1)) if logsoftmax else\
        nn.Sequential(*blocks, nn.Linear(dims[-1], outdim))
    print('\nmodel:', model)
    return model


def ConvBlock(in_channel, out_channel):
    return (nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),

            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel))

def conv_classifier(in_shape, n_class):     # in_shape=(1, 28, 28)
    # in_shape = tuple(in_shape)
    if len(in_shape) == 2:      in_shape = (1,) + tuple(in_shape)
    channels = (in_shape[0],) + (32, 64, 128, 256, 512)
    blocks = tuple()
    for ch1, ch2 in zip(channels, channels[1:]):
        blocks += ConvBlock(ch1, ch2)
    outdim = get_conv_outdim(nn.Sequential(*blocks), in_shape)
    model = nn.Sequential(*blocks,
                          nn.Flatten(),
                          nn.Linear(outdim, 50),    nn.ReLU(),  nn.BatchNorm1d(50),
                          nn.Linear(50, n_class))
    print('\nmodel:', model)
    return model

def get_conv_outdim(model, in_shape):
    if len(in_shape) == 2:      in_shape = (1,) + tuple(in_shape)
    with torch.no_grad():
        out = model(torch.zeros(2, *in_shape))
    outdim = np.prod(out.detach().shape[1:])
    return int(outdim)