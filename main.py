import torch
import pandas as pd
import numpy as np

import random

from torch.utils.data import Dataset, DataLoader

from torch import nn

PATH = 'C:/Users/aaaaa/Desktop/kaggle-house-prices'

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu" 
)

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(287, 256),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        ans = self.layers(x)
        return ans

model = neural_network().to(device)

class house_prices_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None, training = True, add = 0):
        self.houses = pd.read_csv(csv_file)
        self.houses = pd.get_dummies(self.houses)
        self.houses = pd.DataFrame.fillna(self.houses, 0)

        if training == True:
            global train_houses
            train_houses = self.houses[0:0]
        else:
            for columnName in train_houses:
                for idx in range(len(self.houses)):
                    train_houses[columnName] = 0.0

            for columnName in self.houses:
                train_houses[columnName] = self.houses[columnName]
            
            # train_houses.drop('SalePrice', axis = 1)

            train_houses = pd.DataFrame.fillna(train_houses, 0)

            self.houses = train_houses

        self.normalized_data = self.houses[0:]

        for columnName in self.houses:
            ok = True

            for idx in range(len(self.houses)):
                if self.houses[columnName][idx] != True and self.houses[columnName][idx] != False:
                    ok = False
                    break

            # print(columnName, ok)

            if ok == False:
                self.normalized_data[columnName] = (self.houses[columnName] - self.houses[columnName].mean()) / self.houses[columnName].std()
        
        # print(len(self.normalized_data.columns))

        self.normalized_data = self.normalized_data.drop('Id', axis = 1)
        self.normalized_data = self.normalized_data.drop('SalePrice', axis = 1)

        self.houses = self.houses.astype(float)
        self.normalized_data = self.normalized_data.astype(float)

        self.root_dir = root_dir
        self.transform = transform

        if training == True:
            for _ in range(add): # Expanding the training data to reduce overfitting
                print(_)

                r = random.randrange(0, len(self.houses))
                nw_ind = len(self.houses)

                self.houses.loc[nw_ind] = self.houses.loc[r].copy()
                self.normalized_data.loc[nw_ind] = self.normalized_data.loc[r].copy()

                for j in self.normalized_data.columns:
                    noise = np.random.normal(0, 0.2)

                    self.normalized_data.loc[nw_ind, j] += noise

                # print(self.normalized_data.loc[nw_ind])
    
    def __len__(self):
        return len(self.houses)
    
    def __getitem__(self, idx):
        x = np.array(self.normalized_data.loc[idx], dtype = "float32")
        y = np.array([self.houses.loc[idx, 'SalePrice']], dtype = "float32")

        return torch.from_numpy(x), torch.from_numpy(y)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(x)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def submit(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    idx = 0

    submission = pd.DataFrame()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            for i in range(len(pred)):
                submission.loc[idx, 'Id'] = idx + 1461
                submission.loc[idx, 'SalePrice'] = pred[i, 0].item()
                idx = idx + 1
    
    submission['Id'] = submission['Id'].astype(int)
    submission.to_csv(PATH + '/submission.csv', index = False)

batch_size = 128

train_data = house_prices_dataset(csv_file = PATH + '/train.csv', root_dir = PATH)
train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

lmb = 40.0

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1, betas = (0.9, 0.999), eps = 1e-8, weight_decay = lmb / (2 * len(train_data)))

epochs = 500

for ep in range(epochs):
    print(f"Epoch {ep + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(train_dataloader, model, loss_fn)
    torch.save(model, PATH + '/model.pt')

model = torch.load(PATH + '/model.pt')

test_data = house_prices_dataset(csv_file = PATH + '/test.csv', root_dir = PATH, training = False)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

submit(test_dataloader, model, loss_fn)

print("Done!")
