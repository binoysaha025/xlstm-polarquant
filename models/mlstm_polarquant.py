import torch
import torch.nn as nn

class PolarQuant:
    def __init__(self, n_bits=8, hidden_size=16):
        self.n_bits = n_bits
        self.n_bins = 2 ** n_bits   #256 bins for 8 bit
        self.hidden_size = hidden_size
        self.R = None

    def quantize(self, C):
        norm = torch.norm(C, dim=-1, keepdim=True)
        C_normalized = C / torch.clamp(norm, min=1e-6)

        # quantize to int8 range but stay in float32 for MPS compatibility
        C_quantized = torch.round(
            torch.clamp(C_normalized, -1, 1) * ((self.n_bins // 2) - 1)
        ) / ((self.n_bins // 2) - 1) 
            
        return C_quantized, norm, C_normalized
    
    def _get_R(self, d, device):
        # lazy init - create once, reuse
        if self.R is None or self.R.device != torch.device(device):
            # JL random project: (d, d//2)
            # entries are +-1/sqrt(d//s2) for distance preservaion
            k = max(d // 2,1)
            R = torch.randint(0,2, (d, k), device=device).float() * 2 - 1
            self.R = R / (k ** 0.5)
        return self.R
    
    def qjl_correct(self, C, C_approx):
        # QJL residual correction
        residual = C - C_approx   #lost in quantization
        d = residual.shape[-1]
        R = self._get_R(d, residual.device) # (d,k)
        # project residual into lower dim, then back project
        projected = torch.matmul(residual, R)
        reconstructed = torch.matmul(projected, R.T)
        return C_approx + reconstructed.detach()
    
    def dequantize(self, C_quantized, norm, original_shape):
        # C_quantized is already normalized, just scale by magnitude
        C_approx = C_quantized * norm
        return C_approx
        
class mLSTMCellPolarQuant(nn.Module):
    def __init__(self, input_size, hidden_size, n_bits=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)
        self.wi = nn.Linear(input_size, hidden_size)
        self.wf = nn.Linear(input_size, hidden_size)
        self.wo = nn.Linear(input_size, hidden_size)

        self.pq = PolarQuant(n_bits=n_bits, hidden_size=hidden_size)

    def forward(self, x , state):
        C, n, m = state

        q = self.Wq(x)
        k = self.Wk(x) / (self.hidden_size ** 0.5)
        v = self.Wv(x)

        i = self.wi(x)
        f = self.wf(x)
        o = torch.sigmoid(self.wo(x))

        #stabilize
        m_new = torch.max (f + m, i)
        i_stable = torch.exp(i - m_new)
        f_stable = torch.exp(f + m - m_new)

        #outer product
        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))

        #update matrix memory
        C_new = f_stable.unsqueeze(2) * C + i_stable.unsqueeze(2) * vk

        #polarquantize the matrix memory
        C_q, norm, _ = self.pq.quantize(C_new)
        C_approx = self.pq.dequantize(C_q, norm, C_new.shape)
        C_new = self.pq.qjl_correct(C_new, C_approx)
        C_new = torch.nan_to_num(C_new, nan=0.0)

        # normalizer 
        n_new = f_stable * n + i_stable * k

        #read from memory
        Cq = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)
        denom = torch.clamp(
            torch.abs(torch.sum(n_new * q, dim=-1, keepdim=True)), min=1.0
        )
        h_new = o * (Cq / denom)

        return h_new, (C_new, n_new, m_new)
    
    def init_state(self, batch_size, device):
        C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
        n = torch.zeros(batch_size, self.hidden_size, device=device)
        m = torch.zeros(batch_size, self.hidden_size, device=device)
        return (C, n, m)