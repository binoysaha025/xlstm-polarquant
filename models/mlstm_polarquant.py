import torch
import torch.nn as nn
import math
class PolarQuant:
    def __init__(self, n_bits=8, hidden_size=64, k=8):
        self.n_bits = n_bits
        self.n_bins = 2 ** n_bits   #256 bins for 8 bit
        self.hidden_size = hidden_size
        self.k = k          # projection dims
        self.n_angles = k // 2

        torch.manual_seed(42)
        P = torch.randn(hidden_size, k)
        # orthonormalize for better distance preservation
        P, _ = torch.linalg.qr(P)
        self.P = P
        self._P_device = None

        #JL correction matrix (hidden, hidden // 2)
        torch.manual_seed(43)
        k_jl = max(hidden_size // 2, 1)
        R = torch.randint(0, 2, (hidden_size, k_jl)).float() * 2 - 1
        self.R = R / (k_jl ** 0.5)
        self._R_device = None

    def _get_P(self, device):
        if self._P_device != str(device):
            self.P = self.P.to(device)
            self._P_device = str(device)
        return self.P
    
    def _get_R(self, device):
        if self._R_device != str(device):
            self.R = self.R.to(device)
            self._R_device = str(device)
        return self.R

    def quantize(self, C):
        device = C.device

        #1 : magnitude per row
        norm = torch.norm(C, dim=-1, keepdim=True)
        C_norm = C / torch.clamp(norm, min=1e-6)

        #2: project onto k basis vectors
        P = self._get_P(device)
        projected = torch.matmul(C_norm, P)

        #3: compute n_angles angles via arctan2 on consecutive pairs
        angles = []
        for i in range(self.n_angles):
            theta = torch.atan2(
                projected[..., 2*i+1],
                projected[..., 2*i]
            )
            angles.append(theta)
        angles = torch.stack(angles, dim=-1)   #(batch, hidden, n_angles)

        #4: quantize angles to int8 range
        # theta in [-pi, pi] -> [-127, 127]
        angles_q = torch.round(
            angles / math.pi * 127
        ).clamp(-127, 127)                      # stay float32 for MPS compatibility

        return angles_q, norm, C_norm
    
    def dequantize(self, angles_q, norm, original_shape):
        device = norm.device
        P = self._get_P(device)

        #1: dequantize angles back to floats
        angles = angles_q / 127.0 * math.pi  #(batch, hidden, n_angles)

        #2: reconstruct k-dim projected vector from angles
        parts = []
        for i in range(self.n_angles):
            parts.append(torch.cos(angles[..., i]))     # even dims
            parts.append(torch.sin(angles[..., i]))     # odd dims
        projected_approx = torch.stack(parts, dim=-1)   #(batch, hidden, k)

        #3: unproject back to hidden dimns using P.T
        # P is orthonormal so P.T is its psuedo-inverse
        C_approx = torch.matmul(projected_approx, P.T)

        #4: scale by magnitude
        C_aprox = C_approx * norm           #(batch, hidden, hidden) - this is the quantized approximation of C

        return C_approx

    def qjl_correct(self, C, C_approx):
        # QJL residual correction
        residual = C - C_approx   #lost in quantization
        d = residual.shape[-1]
        R = self._get_R(residual.device) # (d,k)
        # project residual into lower dim, then back project
        projected = torch.matmul(residual, R)
        reconstructed = torch.matmul(projected, R.T)
        return C_approx + reconstructed.detach()
        
class mLSTMCellPolarQuant(nn.Module):
    def __init__(self, input_size, hidden_size, n_bits=8, k=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)
        self.wi = nn.Linear(input_size, hidden_size)
        self.wf = nn.Linear(input_size, hidden_size)
        self.wo = nn.Linear(input_size, hidden_size)

        self.pq = PolarQuant(n_bits=n_bits, hidden_size=hidden_size, k=k)

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
        with torch.no_grad():
            angles_q, norm, _ = self.pq.quantize(C_new)
            C_approx          = self.pq.dequantize(angles_q, norm, C_new.shape)

        # apply correction but keep C_new in the graph
        correction = (C_approx - C_new).detach()
        C_new = C_new + correction  # shifts C_new toward quantized version without breaking gradients
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