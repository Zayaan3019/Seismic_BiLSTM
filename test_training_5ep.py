"""Run 5 epochs of the full BiLSTM to verify training loop works end to end."""
import sys, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sda_pipeline_utils import INPUT_COLUMNS, OUTPUT_COLUMNS, RANDOM_STATE, prepare_dataset, split_data

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

print("Loading data...", flush=True)
bundle = prepare_dataset("NGA_Subduction_filtered.csv")
X, y = bundle["X"], bundle["y_log"]
splits = split_data(X, y)
X_train, X_val = splits["X_train"], splits["X_val"]
y_train, y_val = splits["y_train"], splits["y_val"]

sX = StandardScaler().fit(X_train)
sy = StandardScaler().fit(y_train)
X_tr = sX.transform(X_train)
X_va = sX.transform(X_val)
y_tr = sy.transform(y_train)
y_va = sy.transform(y_val)
print(f"Data ready: {X_tr.shape}, {y_tr.shape}", flush=True)

N_INPUTS, N_OUTPUTS = 6, 29

class SeismicBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_dims = (128, 256, 256, 128)
        period_embed_dim = 32
        lstm_hidden = 128
        dropout = 0.15
        layers = []
        in_dim = N_INPUTS
        for i, out_dim in enumerate(encoder_dims):
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()]
            if i in (1, 2): layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        self.period_embed = nn.Embedding(N_OUTPUTS, period_embed_dim)
        lstm_input_dim = encoder_dims[-1] + period_embed_dim
        self.bilstm = nn.LSTM(lstm_input_dim, lstm_hidden, num_layers=2,
                              batch_first=True, bidirectional=True,
                              dropout=dropout)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        batch = x.size(0)
        ctx = self.encoder(x)
        p_idx = torch.arange(N_OUTPUTS, device=x.device)
        p_emb = self.period_embed(p_idx).unsqueeze(0).expand(batch, -1, -1)
        ctx_exp = ctx.unsqueeze(1).expand(-1, N_OUTPUTS, -1)
        seq = torch.cat([ctx_exp, p_emb], dim=-1)
        out, _ = self.bilstm(seq)
        return self.output_head(out).squeeze(-1)

net = SeismicBiLSTM()
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Model: {total_params:,} params", flush=True)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                          batch_size=128, shuffle=True, drop_last=False)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va)),
                        batch_size=512, shuffle=False)

opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
crit = nn.MSELoss()

for epoch in range(1, 6):
    net.train()
    tloss = 0.0
    for Xb, yb in train_loader:
        opt.zero_grad()
        pred = net(Xb)
        loss = crit(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        tloss += loss.item() * Xb.size(0)
    tloss /= len(X_tr)

    net.eval()
    vloss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            vloss += crit(net(Xb), yb).item() * Xb.size(0)
    vloss /= len(X_va)

    print(f"Epoch {epoch}/5 | train={tloss:.5f} | val={vloss:.5f}", flush=True)

print("5-epoch test PASSED", flush=True)
