import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import os
import pickle


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=6, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=6, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 10, kernel_size=6, stride=1, padding=1)

    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(batch_size, 10)


def load_model():
    directory = os.path.abspath(os.path.dirname(__file__))

    model = Model()
    model.load_state_dict(torch.load(directory + '/weights.pth', map_location='cpu'))

    return model


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def get_validation_loss(model, loss_func, valid_dl):
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        )
    return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    best_loss = get_validation_loss(model, loss_func, valid_dl)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        val_loss = get_validation_loss(model, loss_func, valid_dl)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'weights.pth')

        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


if __name__ == "__main__":
    batch_size = 50

    with open('trn.pkl', 'rb') as f:
        trn = pickle.load(f)
    with open('val.pkl', 'rb') as f:
        val = pickle.load(f)

    x_train = trn['data'].transpose((0, 3, 1, 2)).astype('f4')
    y_train = trn['labels'].astype('i8')

    x_valid = val['data'].transpose((0, 3, 1, 2)).astype('f4')
    y_valid = val['labels'].astype('i8')

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)

    model = Model()
    model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
    loss_func = F.cross_entropy
    epochs = 10

    opt = optim.Adamax(model.parameters(), lr=0.0001)
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
