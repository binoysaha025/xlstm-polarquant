import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
SEQ_LEN = 60
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

class PriceDataset(Dataset):
    def __init__(self, ticker, start, end, seq_len=SEQ_LEN):
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df = df[FEATURES].dropna()

        # normalize each feature
        self.mean = df.mean()
        self.std = df.std()
        df = (df - self.mean) / self.std

        data = df.values.astype(np.float32)    # (T, 5)

        # build sequences
        self.X, self.y = [], []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])    #60 days input
            self.y.append(data[i+seq_len, 3])   #next day close (index 3)
        self.X = torch.tensor(np.array(self.X))  #(N, 60, 5)
        self.y = torch.tensor(np.array(self.y))    #(N,)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_loaders(ticker='AAPL', train_start='2010-01-01', 
                train_end='2022-01-01', test_start='2022-01-01', 
                test_end='2024-01-01', batch_size=32):
    train_ds = PriceDataset(ticker, train_start, train_end)
    test_ds = PriceDataset(ticker, test_start, test_end)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
