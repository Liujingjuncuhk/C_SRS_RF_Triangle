import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import joblib



# ── data ─────────────────────────────────────────────────────────────────────
data_file = "./data/training_data_all.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

ee_pos_all = np.array(data['ee_pos'],       dtype=np.float32)  # (N, 3)
cl_all     = np.array(data['cable_length'], dtype=np.float32)  # (N, 6)

print(f"Dataset: {len(ee_pos_all)} samples")
print(f"EE pos   range: {ee_pos_all.min(axis=0).round(4)} – {ee_pos_all.max(axis=0).round(4)}")
print(f"Cable L  range: {cl_all.min(axis=0).round(4)} – {cl_all.max(axis=0).round(4)}")

# ── normalisation ─────────────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = scaler_X.fit_transform(ee_pos_all)
Y = scaler_Y.fit_transform(cl_all)

X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)

dataset  = TensorDataset(X_t, Y_t)
n_val    = int(0.2 * len(dataset))
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)

# ── model ─────────────────────────────────────────────────────────────────────
class IK_MLP(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,   128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,   6),
        )

    def forward(self, x):
        return self.net(x)

model = IK_MLP()
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── training ──────────────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=30, factor=0.5, min_lr=1e-5)

best_val   = float('inf')
num_epochs = 600

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= n_train

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item() * len(xb)
    val_loss /= n_val
    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "ik_model_best.pth")

    if epoch % 50 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.2e}")

print(f"\nBest val MSE (normalised): {best_val:.6f}")

# ── report error in physical units ───────────────────────────────────────────
model.load_state_dict(torch.load("ik_model_best.pth"))
model.eval()
with torch.no_grad():
    Y_pred_norm = model(X_t).numpy()
Y_pred = scaler_Y.inverse_transform(Y_pred_norm)  # back to metres
mae    = np.abs(Y_pred - cl_all).mean(axis=0)
print(f"Per-cable MAE (m): {np.round(mae, 5)}")
print(f"Mean MAE (m):      {mae.mean():.5f}")

# ── save scalers ──────────────────────────────────────────────────────────────
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_Y, "scaler_Y.pkl")
print("Saved: ik_model_best.pth  scaler_X.pkl  scaler_Y.pkl")
