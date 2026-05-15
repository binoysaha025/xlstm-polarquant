import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from scipy import stats

from models.xlstm import xLSTM
from models.transformer_baseline import TransformerBaseline

import yfinance as yf


app = FastAPI(title = "xLSTM PolarQuant API")

# LOAD TRAINING DATA STATS FOR DRIFT DETECTION
def compute_train_stats():
    df = yf.download('AAPL', start='2010-01-01', end='2022-01-01', auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    mean = df.mean().values.astype(np.float32)
    std  = df.std().values.astype(np.float32)
    # reference data: last 60 days of training data as numpy array
    ref  = df.values.astype(np.float32)[-60:]
    return mean, std, ref

TRAIN_MEAN, TRAIN_STD, reference_data = compute_train_stats()
print(f"train stats loaded — close mean: {TRAIN_MEAN[3]:.2f}, std: {TRAIN_STD[3]:.2f}")

INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SEQ_LEN = 60 

# LOAD MODELS
device = torch.device('cpu')

def load_model(model_type, use_pq=False):
    if model_type == 'transformer':
        model = TransformerBaseline(
            INPUT_SIZE,
            HIDDEN_SIZE, 
            NUM_LAYERS
        )
        path = 'weights/Tranformer.pt'
    elif use_pq:
        model = xLSTM(
            INPUT_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            use_polarquant=True
        )
        path = 'weights/xLSTM_polarquant.pt'
    else:
        model = xLSTM(
            INPUT_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            use_polarquant=False
        )
        path = 'weights/xLSTM_vanilla.pt'
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
vanilla_model = load_model('xlstm', use_pq=False)
pq_model = load_model('xlstm', use_pq=True)
transformer_model = load_model('transformer')

print('all models loaded')

# REQUEST SCHEMA 
class PredictRequest(BaseModel):
    # 60 days of OHCLV data; each day is [open, high, low, close, volume]
    sequence: List[List[float]]

class PredictResponse(BaseModel):
    vanilla_xlstm: float
    polarquant_xlstm: float
    transformer: float
    drift_detected: bool

# DRIFT DETECTION
# reference data - last 60 days of training data distribution
def check_drift(input_data: np.ndarray) -> bool:
    # normalize before comparing
    input_norm = (input_data - TRAIN_MEAN) / (TRAIN_STD + 1e-8)
    ref_norm   = (reference_data - TRAIN_MEAN) / (TRAIN_STD + 1e-8)
    for i in range(input_data.shape[1]):
        stat, p_value = stats.ks_2samp(ref_norm[:, i], input_norm[:, i])
        if p_value < 0.01:
            return True
    return False

# PREDICTION ENDPOINT
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if len(req.sequence) != SEQ_LEN:
        raise HTTPException(status_code=400, 
                            detail=f"Input sequence must be of length {SEQ_LEN}")
    if len(req.sequence[0]) != INPUT_SIZE:
        raise HTTPException(status_code=400, 
                            detail=f"Each timestep must have {INPUT_SIZE} features")
    
    arr = np.array(req.sequence, dtype=np.float32)  # shape (seq_len, input_size)

    # NORMALIZE INPUT on training stats
    arr_norm = np.array(arr - TRAIN_MEAN) / (TRAIN_STD + 1e-8)  # shape (seq_len, input_size)

    # CHECK DRIFT on user input vs training distribution
    drift = check_drift(arr)

    # TO TENSOR
    x = torch.tensor(arr_norm).unsqueeze(0)  # shape (1, seq_len, input_size)

    with torch.no_grad():
        v_out, _ = vanilla_model(x)
        pq_out, _ = pq_model(x)
        t_out, _ = transformer_model(x)

    # DENORMALIZE PREDICTIONS BACK TO DOLLAR VALUES
    close_mean = float(TRAIN_MEAN[3])
    close_std  = float(TRAIN_STD[3])
    
    def denorm(val):
        return float(val.squeeze()) * close_std + close_mean
    
    return PredictResponse(
        vanilla_xlstm=denorm(v_out),
        polarquant_xlstm=denorm(pq_out),
        transformer=denorm(t_out),
        drift_detected=drift
    )

@app.get("/health")
async def health():
    return {"status": "ok"}