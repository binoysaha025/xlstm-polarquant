import torch
import torch.nn as nn 
import torch.nn.functional as F

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #query, key, value projects
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)

        # input and forget gates (scalar)
        self.wi = nn.Linear(input_size, hidden_size)
        self.wf = nn.Linear(input_size, hidden_size)

        # output gate
        self.wo = nn.Linear(input_size, hidden_size)

    def forward(self, x , state):
        C, n, m = state
        #C: matrix memory (batch, hidden, hidden)
        #b: normalizer vector (batch, hidden)
        #m: stabilizer (batch, hidden)

        #projections
        q = self.Wq(x)    # (batch, hidden)
        k = self.Wk(x) / (self.hidden_size ** 0.5)     #scaled like attention
        v = self.Wv(x)    # (batch, hidden)

        #gates
        i = self.wi(x)      #(batch, hidden)
        f = self.wf(x)      #(batch, hidden)
        o = torch.sigmoid(self.wo(x))

        # exponential stabilization
        m_new = torch.max(f + m, i)
        i_stable = torch.exp(i - m_new)
        f_stable = torch.exp(f + m - m_new)

        # outer product: v @ k -> (batch, hidden, hidden)
        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))

        #matrix memory update
        C_new = f_stable.unsqueeze(2) * C + i_stable.unsqueeze(2) * vk

        #normalizer update
        n_new = f_stable * n + i_stable * k

        #read from memory
        Cq = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)    #(batch, hidden)

        #normalize output 
        denom = torch.clamp(
            torch.abs(torch.sum(n_new * q, dim=-1, keepdim=True)),
            min=1.0
        )
        h_new = o * (Cq / denom)

        return h_new, (C_new, n_new, m_new)
    
    def init_state(self, batch_size, device):
        C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
        n = torch.zeros(batch_size, self.hidden_size, device=device)
        m = torch.zeros(batch_size, self.hidden_size, device=device)
        return (C, n, m)
