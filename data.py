from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
T, F = True, False
path = '/Users/jwkim/Documents/data'

def plot(x):
    img = x.numpy()
    plt.imshow(img, cmap='gray')
    plt.show();

def get_dataloader(X, y, batch_size=256, shuffle=F):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
def get_dataloaders(Xtrn, Xval, Xtst, ytrn, yval, ytst, batch_size=256):
    return (DataLoader(TensorDataset(Xtrn, ytrn), batch_size=batch_size, shuffle=T),
            DataLoader(TensorDataset(Xval, yval), batch_size=batch_size, shuffle=F),
            DataLoader(TensorDataset(Xtst, ytst), batch_size=batch_size, shuffle=F))

def get_loaders(Xtrn, ytrn, Xtst, ytst, val_ratio=None, batch_size=256):
    if val_ratio:
        Xtrn, Xval, ytrn, yval = train_test_split(
            Xtrn, ytrn, test_size=val_ratio, stratify=ytrn)
    trnloader = get_dataloader(Xtrn, ytrn, batch_size, shuffle=T)
    if val_ratio:
        valloader = get_dataloader(Xval, yval, batch_size, shuffle=F)
    tstloader = get_dataloader(Xtst, ytst, batch_size, shuffle=F)
    return (trnloader, valloader, tstloader) if val_ratio else (
        trnloader, tstloader)

def getCaliforniaHousing(ratios=(6, 2, 2)):
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing()
    return get_trn_val_tst(ds, ratios)

def get_trn_val_tst(ds, ratios, stratify=F):
    X, y = ds.data, ds.target
    print(f'Original data shape: {X.shape}, {y.shape}')
    test_size = ratios[-1] / sum(ratios)
    Xtrn, Xtst, ytrn, ytst =\
        train_test_split(X, y, test_size=test_size, stratify=y) if stratify else\
        train_test_split(X, y, test_size=test_size)
    test_size = ratios[1] / sum(ratios[:-1])
    Xtrn, Xval, ytrn, yval =\
        train_test_split(Xtrn, ytrn, test_size=test_size, stratify=ytrn) if stratify else\
        train_test_split(Xtrn, ytrn, test_size=test_size)
    scaler = StandardScaler()
    Xtrn = torch.tensor(scaler.fit_transform(Xtrn)).float()
    Xval = torch.tensor(scaler.transform(Xval)).float()
    Xtst = torch.tensor(scaler.transform(Xtst)).float()
    ytrn = torch.tensor(ytrn).float().unsqueeze(1)
    yval = torch.tensor(yval).float().unsqueeze(1)
    ytst = torch.tensor(ytst).float().unsqueeze(1)
    print(f'Trn: {Xtrn.shape}, {ytrn.shape}')
    print(f'Val: {Xval.shape}, {yval.shape}')
    print(f'Tst: {Xtst.shape}, {ytst.shape}\n')
    return Xtrn, Xval, Xtst, ytrn, yval, ytst

def getWisconsinCancer(ratios=(6, 2, 2)):
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer()
    return get_trn_val_tst(ds, ratios, stratify=T)


def getMNIST():     # val_ratio=.2, batch_size=256
    from torchvision import datasets, transforms
    trn = datasets.MNIST(
        path, train=T, download=T, 
        transform=transforms.Compose([transforms.ToTensor()]))
    tst = datasets.MNIST(
        path, train=F, download=T,
        transform=transforms.Compose([transforms.ToTensor()]))
    # trn.data : (60000, 28, 28) 0~255 / trn.targets : (60000,), 0~9
    # tst.data : (10000, 28, 28)       / tst.targets : (10000,)
    Xtrn, ytrn = trn.data / 255, trn.targets.unsqueeze(1)           # reshape(len(trn.data), -1)
    Xtst, ytst = tst.data / 255, tst.targets.unsqueeze(1)           # reshape(len(tst.data), -1)
    print(f'{"X.shape":^30}{"X[0].max()":^15}{"y.shape":^20}{"y.max()":^12}')
    print('trn', Xtrn.shape, Xtrn[0].max(), ytrn.shape, ytrn.max())
    print('tst', Xtst.shape, Xtst[0].max(), ytst.shape, ytst.max())
    return Xtrn, ytrn, Xtst, ytst