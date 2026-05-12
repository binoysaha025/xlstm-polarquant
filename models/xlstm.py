import torch
import torch.nn as nn
from models.slstm import sLSTMCell
from models.mlstm import mLSTMCell
from models.mlstm_polarquant import mLSTMCellPolarQuant

class XLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, use_polarquant=False, n_bits=8):
        super().__init__()
        self.hidden_size = hidden_size

        self.slstm = sLSTMCell(input_size, hidden_size)

        if use_polarquant:
            self.mlstm = mLSTMCellPolarQuant(hidden_size, hidden_size, n_bits=n_bits)
        else:
            self.mlstm = mLSTMCell(hidden_size, hidden_size)

        # layer norms for syable training
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        #project input to hidden size for residual
        self.input_proj = nn.Linear(input_size, hidden_size)

    def forward(self, x, slstm_state, mlstm_state):
        #project input for residual connection
        x_proj = self.input_proj(x)     #(batch, hidden)

        # sltsm: process input, get hidden
        h_s, slstm_state = self.slstm(x, slstm_state)

        # residual + nirm after sLSTM
        h_s = self.norm1(h_s + x_proj)

        # mLSTM takes sLSTM ouput as input
        h_m, mlstm_state = self.mlstm(h_s, mlstm_state)

        # residual + morm after mLSTM
        h_out = self.norm2(h_m + h_s)

        return h_out, slstm_state, mlstm_state
    
    def init_states(self, batch_size, device):
        return (
            self.slstm.init_state(batch_size, device),
            self.mlstm.init_state(batch_size, device)
        )
    
class xLSTM(nn.Module):
    "Full xLSTM model with xLSTM blocks + output headers"
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1, use_polarquant=False, n_bits=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # stack of xLSTM blocks
        self.blocks = nn.ModuleList([
            XLSTMBlock(
                input_size if i == 0 else hidden_size, 
                hidden_size,
                use_polarquant=use_polarquant,
                n_bits=n_bits
            )
            for i in range(num_layers)
        ])

        # final output head: hidden -> predicted prices
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, states=None):
        # x shape: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        device = x.device

        # initialize states if not provided
        if states is None:
            states = [block.init_states(batch, device) for block in self.blocks]

        # processs sequence timestep by tomestep
        for t in range(seq_len):
            x_t = x[:, t, :]   #(batch, input_size)
            new_states = []

            for i, block in enumerate(self.blocks):
                slstm_state, mlstm_state = states[i]
                x_t, slstm_state, mlstm_state = block(x_t, slstm_state, mlstm_state)
                new_states.append((slstm_state, mlstm_state))
            states = new_states
        
        out = self.output_head(x_t)   #(batch, output_size)
        return out, states

