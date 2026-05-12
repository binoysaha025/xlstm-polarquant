import torch
from models.slstm import sLSTMCell
from models.mlstm import mLSTMCell
from models.mlstm_polarquant import mLSTMCellPolarQuant
from models.xlstm import xLSTM

batch, input_size, hidden_size, seq_len = 2, 5, 16, 60

# test sLSTM
slstm = sLSTMCell(input_size, hidden_size)
state = slstm.init_state(batch, 'cpu')
for t in range(seq_len):
    x = torch.randn(batch, input_size)
    h, state = slstm(x, state)
print(f"sLSTM output shape: {h.shape}")

# test mLSTM
mlstm = mLSTMCell(input_size, hidden_size)
state = mlstm.init_state(batch, 'cpu')
for t in range(seq_len):
    x = torch.randn(batch, input_size)
    h, state = mlstm(x, state)
print(f"mLSTM output shape: {h.shape}")
print("both cells good")

# test PolarQuant
mlstm_pq = mLSTMCellPolarQuant(input_size, hidden_size, n_bits=8)
state = mlstm_pq.init_state(batch, 'cpu')
for t in range(seq_len):
    x = torch.randn(batch, input_size)
    h, state = mlstm_pq(x, state)
print(f"mLSTM+PolarQuant output shape: {h.shape}")

# test full xLSTM
model = xLSTM(input_size=5, hidden_size=16, num_layers=2, use_polarquant=False)
x = torch.randn(2, 60, 5)
out, _ = model(x)
print(f"xLSTM output shape: {out.shape}")

model_pq = xLSTM(input_size=5, hidden_size=16, num_layers=2, use_polarquant=True)
out_pq, _ = model_pq(x)
print(f"xLSTM+PolarQuant output shape: {out_pq.shape}")

print("all tests passed")