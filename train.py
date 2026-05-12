import torch
import torch.nn as nn
from models.xlstm import xLSTM
from models.transformer_baseline import TransformerBaseline
from data.fetch import get_loaders
import mlflow
import mlflow.pytorch

# config
TICKER      = 'AAPL'
INPUT_SIZE  = 5
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
EPOCHS      = 50
LR          = 1e-4
BATCH_SIZE  = 32

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"training on {device}")

    train_loader, test_loader = get_loaders(TICKER, batch_size=BATCH_SIZE)

    # train both vanilla and polarquant, compare
    configs = [
        ('xLSTM_vanilla',     False, 'xlstm'),
        ('xLSTM_polarquant',  True, 'xlstm'),
        ('Tranformer', False, 'transformer'),
    ]

    for run_name, use_pq, model_type in configs:
        print(f"\n--- {run_name} ---")
        if model_type == 'transformer':
            model = TransformerBaseline(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS
            ).to(device)
        else:
            model = xLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                use_polarquant=use_pq
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        loss_fn   = nn.MSELoss()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                'ticker':      TICKER,
                'hidden_size': HIDDEN_SIZE,
                'num_layers':  NUM_LAYERS,
                'epochs':      EPOCHS,
                'lr':          LR,
                'polarquant':  use_pq
            })

            for epoch in range(EPOCHS):
                # --- train ---
                model.train()
                train_loss = 0
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    out, _ = model(X)
                    loss = loss_fn(out.squeeze(), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # --- eval ---
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X, y in test_loader:
                        X, y = X.to(device), y.to(device)
                        out, _ = model(X)
                        val_loss += loss_fn(out.squeeze(), y).item()
                val_loss /= len(test_loader)
                scheduler.step(val_loss)

                print(f"epoch {epoch+1}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f}")
                mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

            mlflow.pytorch.log_model(model, run_name)
            print(f"{run_name} done")

if __name__ == '__main__':
    train()