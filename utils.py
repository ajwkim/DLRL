import time, copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from sklearn.metrics import confusion_matrix
import torch
from torch import nn, optim
T, F = True, False

def elapsed(t0):
    duration = int(time.time() - t0)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return (f'{h}h' if h else '') + (f'{m}m' if m else '') + f'{s}s'

def plot_loss_and_acc(trnlosses, vallosses, trnaccs, valaccs):
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].plot(trnlosses, label='TrnLoss')
    ax[0].plot(vallosses, label='ValLoss')
    ax[0].set_yscale('log')
    ax[0].grid(T)
    ax[0].legend()

    ax[1].plot(trnaccs, label='TrnAcc')
    ax[1].plot(valaccs, label='ValAcc')
    ax[1].grid(T)
    ax[1].legend()
    plt.show();

def plot_losses(trnlosses, vallosses, plot_from=10):
    plt.plot(range(plot_from, len(trnlosses)), trnlosses[plot_from:], label='Trn')
    plt.plot(range(plot_from, len(vallosses)), vallosses[plot_from:], label='Val')
    plt.title('Train & Valid Loss History')
    plt.grid(T)
    plt.yscale('log')
    plt.legend()
    plt.show();


class Regressor:
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()   # nn.functional.mse_loss
        self.device = device
        # self.model.to(device)

    def train(self, loader, train=T):
        self.model.train() if train else self.model.eval()
        total_loss = 0
        for Xi, yi in loader:
            yhati = self.model(Xi)
            loss = self.criterion(yhati, yi)
            if train:
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            total_loss += float(loss)
        return total_loss / len(loader)
    
    def eval(self, loader):
        with torch.no_grad():
            return self.train(loader, train=F)
        
    def run(self, trnloader, valloader, n_epoch, n_print=None, early_stop=None,
            plot_from=10):
        t0 = time.time()
        n = len(str(n_epoch))
        best_model, best_epoch, minloss = None, np.inf, np.inf
        trnhistory, valhistory = [], []
        for e in range(n_epoch):
            trnloss = self.train(trnloader)
            valloss = self.eval(valloader)
            trnhistory.append(trnloss)
            valhistory.append(valloss)
            if valloss < minloss:
                minloss = valloss
                best_epoch, best_model = e, copy.deepcopy(self.model.state_dict())
            if n_print and (e + 1) % n_print == 0:
                print(f'Epoch {e+1:{n}d}: TrnLoss {trnloss:.4e}, '
                      f'ValLoss {valloss:.4e}, MinLoss {minloss:.4e} ({elapsed(t0)})')
            if early_stop and e - best_epoch > early_stop:
                print(f'Epoch {e+1:{n}d}: No improvement during last {early_stop} epochs')
                break
        print(f'\nEpoch {best_epoch+1:{n}d}: TrnLoss {trnhistory[best_epoch]:.4e}, '
              f'ValLoss {valhistory[best_epoch]:.4e} ({elapsed(t0)})')
        self.model.load_state_dict(best_model)
        plot_losses(trnhistory, valhistory)
        return trnhistory, valhistory

    def test(self, loader):
        self.model.eval()
        total_loss, yhat, y = 0, [], []
        with torch.no_grad():
            for Xi, yi in loader:
                yhati = self.model(Xi)
                loss = self.criterion(yhati, yi)
                total_loss += loss
                yhat.append(yhati)
                y.append(yi)
        total_loss /= len(loader)
        print(f'TestLoss {total_loss:.4e}')
        y, yhat = torch.cat(y), torch.cat(yhat)
        df = pd.DataFrame(
            torch.cat([y, yhat], dim=-1).numpy(), columns='y yhat'.split())
        sns.pairplot(df);   # height=5


class Classifier:
    def __init__(self, model, binary=F, lr=1e-3, device='cpu'):
        self.model = model
        self.binary = binary
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss() if binary else nn.NLLLoss()
        self.device = device
        # self.model.to(device)

    def train(self, loader, train=T, use_tqdm=F):
        self.model.train() if train else self.model.eval()
        total_loss, correct, samplesize = 0, 0, 0
        for Xi, yi in tqdm(loader) if use_tqdm else loader:
            # Xi, yi = Xi.to(self.device), yi.to(self.device)
            prd = self.model(Xi)
            loss = self.criterion(prd, 
                                  yi if self.binary else yi.squeeze())
            if train:
                self.optimizer.zero_grad(); loss.backward();    self.optimizer.step()
            total_loss += float(loss.item())
            prd_ = prd.detach() > 0 if self.binary else\
                   prd.detach().argmax(dim=-1, keepdim=T)
            correct += (prd_ == yi).sum().item()
            samplesize += yi.size(0)
        return total_loss / len(loader), correct / samplesize

    def eval(self, loader):
        with torch.no_grad():
            return self.train(loader, train=F)
    
    def run(self, trnloader, valloader, n_epoch, n_print=None, early_stop=None):
        t0 = time.time()
        n = len(str(n_epoch))
        best_model, best_epoch, minloss = None, np.inf, np.inf
        trnlosses, vallosses, trnaccs, valaccs = [], [], [], []
        for e in range(n_epoch):
            trnloss, trnacc = self.train(trnloader)
            valloss, valacc = self.eval(valloader)
            trnlosses.append(trnloss)
            vallosses.append(valloss)
            trnaccs.append(trnacc)
            valaccs.append(valacc)
            if valloss < minloss:
                minloss = valloss
                best_epoch, best_model = e, copy.deepcopy(self.model.state_dict())
            if n_print and (e + 1) % n_print == 0:
                print(f'Epoch {e+1:{n}d}: TrnLoss {trnloss:.3e}, ValLoss {valloss:.3e}, MinLoss {minloss:.3e}, '
                      f'TrnAcc {trnacc:.3f}, ValAcc {valacc:.3f} ({elapsed(t0)})')
            if early_stop and e - best_epoch > early_stop:
                print(f'\nEpoch {e+1:{n}d}: No improvement during last {early_stop} epochs')
                break
        print(f'\nEpoch {best_epoch+1:{n}d}: TrnLoss {trnlosses[best_epoch]:.3e}, ValLoss {vallosses[best_epoch]:.3e}, '
              f'TrnAcc {trnaccs[best_epoch]:.3f}, ValAcc {valaccs[best_epoch]:.3f} ({elapsed(t0)})')
        self.model.load_state_dict(best_model)
        plot_loss_and_acc(trnlosses, vallosses, trnaccs, valaccs)

    def test(self, loader):
        self.model.eval()
        total_loss, yhat, y = 0, [], []
        with torch.no_grad():
            for Xi, yi in loader:
                yhati = self.model(Xi)
                loss = self.criterion(yhati, 
                                      yi if self.binary else yi.squeeze())
                total_loss += loss
                yhat.append(nn.functional.sigmoid(yhati) if self.binary else 
                            yhati.argmax(dim=-1, keepdim=T))
                y.append(yi)
        total_loss /= len(loader)
        y, yhat = torch.cat(y), torch.cat(yhat)
        accuracy = ((yhat > .5) == y).sum().item() if self.binary else\
                   (yhat == y).sum().item()
        accuracy /= y.size(0)
        print(f'TestLoss {total_loss:.4e}, TestAccuracy {accuracy:.3f}')
        if not self.binary:
            df_conf = pd.DataFrame(confusion_matrix(y, yhat))
            print('Confusion Matrix: index(true) Vs. columns(pred)')
            display(df_conf)
        df = pd.DataFrame(
            torch.cat([y, yhat], dim=-1).numpy(), columns='y yhat'.split())
        sns.histplot(df, x='yhat', hue='y', bins=50, stat='probability');