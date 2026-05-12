import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # single linear projecting input+hidden -> all 4 gates at once
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x , state):
        h, c, m = state # hidden, cell, stabilizer

        #concat input and hidden
        xh = torch.cat([x,h], dim=-1)

        #project to gates
        gates = self.W(xh)
        z, i, f, o = gates.chunk(4, dim=-1)

        #cell update (tanh squash on input)
        z = torch.tanh(z)

        #EXPONENTIAL gates
        i = torch.exp(i)        #exponential input gate
        f = torch.exp(f)        #exponential forget gate

        #stabilizer: tracks log of max gate value to prevent overflow
        m_new = torch.max(f+m, i)

        #stablized gates
        i_stable = torch.exp(i - m_new)
        f_stable = torch.exp(f + m - m_new)

        # cell update
        c_new = f_stable * c + i_stable * z

        #normalizer
        n = f_stable + i_stable

        #hidden state - normalized cell
        h_new = torch.sigmoid(o) * (c_new / torch.clamp(n, min=1.0))

        return h_new, (h_new, c_new, m_new)
    
    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        m = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h, c, m)

