"""
LaCGKN Forecast + DA + UQ.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils import parameters_to_vector

SEED_STAGE1 = 0
SEED_STAGE2 = 1
device = "cuda:1"

###############################################
################# Data Import #################
###############################################

I_total = 1024                                      # total number of tracers in dataset
I = 64                                              # number of tracers used for DA / training steps

QG_Data = np.load(r"../data/qg_data.npz")
pos = QG_Data["xy_obs"]                             # (Nt, I_total, 2)
psi = QG_Data["psi_noisy"]                          # (Nt, H, W, 2)
pos_unit = np.stack([np.cos(pos[:, :, 0]), np.sin(pos[:, :, 0]), np.cos(pos[:, :, 1]), np.sin(pos[:, :, 1])], axis=-1) # (Nt, I_total, 4)
u1 = torch.tensor(pos_unit, dtype=torch.float32)    # (Nt, I_total, 4)
u2 = torch.tensor(psi, dtype=torch.float32)         # (Nt, H, W, 2)

# Train / Val / Test split
Ntrain, Nval, Ntest = 80000, 10000, 10000
train_u1, train_u2 = u1[:Ntrain], u2[:Ntrain]
val_u1, val_u2     = u1[Ntrain:Ntrain+Nval, :I], u2[Ntrain:Ntrain+Nval]
test_u1, test_u2   = u1[Ntrain+Nval:Ntrain+Nval+Ntest, :I], u2[Ntrain+Nval:Ntrain+Nval+Ntest]


###############################################
#################  Utilities  #################
###############################################

def unit2xy(xy_unit):
    cos0, sin0, cos1, sin1 = xy_unit[..., 0], xy_unit[..., 1], xy_unit[..., 2], xy_unit[..., 3]
    x = torch.atan2(sin0, cos0)
    y = torch.atan2(sin1, cos1)
    return x, y

class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)

    @staticmethod
    def _compute_padding(size, k, s):
        if size % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - (size % s), 0)
        p0 = pad // 2
        p1 = pad - p0
        return (p0, p1)

    def forward(self, x):
        # x: (B, C, H, W)
        H, W = x.shape[-2], x.shape[-1]
        pad_h = self._compute_padding(H, self.kernel_size[0], self.stride[0])
        pad_w = self._compute_padding(W, self.kernel_size[1], self.stride[1])
        x = nnF.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode="circular")
        return self.conv(x)

class CircularConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
        self.pad_w = int(0.5 + (self.kernel_size[1] - 1.0) / (2.0 * self.stride[1]))
        self.pad_h = int(0.5 + (self.kernel_size[0] - 1.0) / (2.0 * self.stride[0]))

    def forward(self, x):
        x = nnF.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="circular")
        out = self.deconv(x)
        crop_h = (self.stride[0] + 1) * self.pad_h
        crop_w = (self.stride[1] + 1) * self.pad_w
        return out[:, :, crop_h:out.shape[2]-crop_h, crop_w:out.shape[3]-crop_w]


###################################################
################# CGKN: AE + CGN  #################
###################################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            CircularConv2d(2, 32, kernel_size=3, stride=1), nn.SiLU(),
            CircularConv2d(32, 64, kernel_size=4, stride=2), nn.SiLU(),
            CircularConv2d(64, 64, kernel_size=3, stride=1), nn.SiLU(),
            CircularConv2d(64, 128, kernel_size=4, stride=2), nn.SiLU(),
            CircularConv2d(128, 64, kernel_size=3, stride=1), nn.SiLU(),
            CircularConv2d(64, 64, kernel_size=3, stride=1), nn.SiLU(),
            CircularConv2d(64, 32, kernel_size=3, stride=1), nn.SiLU(),
            CircularConv2d(32, 2, kernel_size=3, stride=1),
        )

    def forward(self, u):                           # u:(B, H, W, 2)
        u = u.permute(0, 3, 1, 2)                   # → (B, 2, H, W)
        return self.enc(u)                          # → (B, 2, H_z, W_z)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            CircularConvTranspose2d(2, 32, kernel_size=3, stride=1), nn.SiLU(),
            CircularConvTranspose2d(32, 64, kernel_size=3, stride=1), nn.SiLU(),
            CircularConvTranspose2d(64, 64, kernel_size=3, stride=1), nn.SiLU(),
            CircularConvTranspose2d(64, 128, kernel_size=3, stride=1), nn.SiLU(),
            CircularConvTranspose2d(128, 64, kernel_size=4, stride=2), nn.SiLU(),
            CircularConvTranspose2d(64, 64, kernel_size=3, stride=1), nn.SiLU(),
            CircularConvTranspose2d(64, 32, kernel_size=4, stride=2), nn.SiLU(),
            CircularConvTranspose2d(32, 2, kernel_size=3, stride=1),
        )

    def forward(self, z):                           # z:(B, 2, H_z, W_z)
        u = self.dec(z)                             # → (B, 2, H, W)
        return u.permute(0, 2, 3, 1)                # → (B, H, W, 2)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z
        self.use_pos_encoding = use_pos_encoding
        self.num_frequencies = num_frequencies
        # Homogeneous Tracers
        in_dim = 2
        if use_pos_encoding:
            in_dim = 2 + 4 * num_frequencies        # x,y + sin/cos for x/y for each frequency
        out_dim_f1 = 4
        out_dim_g1 = 4 * dim_z

        self.f1_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, out_dim_f1),
        )
        self.g1_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, out_dim_g1),
        )

        self.f2_param = nn.Parameter((1.0 / dim_z**0.5) * torch.rand(dim_z, 1))  # (dim_z, 1)
        self.g2_param = nn.Parameter((1.0 / dim_z) * torch.rand(dim_z, dim_z))   # (dim_z, dim_z)

    def positional_encoding(self, x):               # x: (B, 2)
        freqs = (2.0 ** torch.arange(self.num_frequencies, device=x.device)) * np.pi  # (K,)
        x1 = x[:, 0:1] * freqs                      # (B, K)
        x2 = x[:, 1:2] * freqs                      # (B, K)
        return torch.cat([x, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)  # (B, 2+4K)

    def forward(self, x):                           # x: (B, I, 2)
        B, I = x.shape[:2]
        if self.use_pos_encoding:
            x = self.positional_encoding(x.view(-1,2)).view(B, I, -1) # (B, I, in_dim)

        # per-tracer outputs -> combine into (B, I*4, ...)
        f1 = self.f1_net(x).view(B, self.dim_u1, 1)             # (B, I*4, 1)
        g1 = self.g1_net(x).view(B, self.dim_u1, self.dim_z)    # (B, I*4, dim_z)
        g1 = nnF.softmax(g1 / 1, dim=-1) # sharp attention 
        f2 = self.f2_param.expand(B, -1, -1)                    # (B, dim_z, 1)
        g2 = self.g2_param.expand(B, -1, -1)                    # (B, dim_z, dim_z)
        return [f1, g1, f2, g2]

class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, u1, z):                       # u1: (B, I, 4), z: (B, dim_z)
        x, y = unit2xy(u1)
        pos = torch.stack([x, y], dim=-1)           # (B, I, 2)
        f1, g1, f2, g2 = self.cgn(pos)              # f1: (B, I*4, 1), g1: (B, I*4, dim_z), f2: (B, dim_z, 1), g2: (B, dim_z, dim_z)
        z = z.unsqueeze(-1)                         # (B, dim_z, 1)
        u1_pred = f1 + g1@z                         # (B, I*4, 1)
        z_pred = f2 + g2@z                          # (B, dim_z, 1)
        return u1_pred.view(*u1.shape), z_pred.squeeze(-1)


###############################################
################# Train Stage 1 ###############
###############################################

# Stage1: Train cgkn with loss_ae + loss_forecast_u + loss_forecast_z
torch.manual_seed(SEED_STAGE1)
np.random.seed(SEED_STAGE1)

dim_u1 = I * 4
dim_u2 = 64 * 64 * 2
H_z, W_z = 16, 16
dim_z = H_z * W_z * 2

epochs_stage1 = 500
train_batch_size = 200
val_batch_size = 1000
val_per_epochs = 10
train_tensor = torch.utils.data.TensorDataset(train_u1[:-1], train_u2[:-1], train_u1[1:], train_u2[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
val_tensor = torch.utils.data.TensorDataset(val_u1[:-1], val_u2[:-1], val_u1[1:], val_u2[1:])
val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=val_batch_size)
train_num_batches = len(train_loader)
Niters = epochs_stage1 * train_num_batches
loss_history_stage1 = {
    "train_forecast_u1": [],
    "train_forecast_u2": [],
    "train_ae": [],
    "train_forecast_z": [],
    "val_forecast_u1": [],
    "val_forecast_u2": [],
    "val_ae": [],
    "val_forecast_z": [],
}
best_val_loss = float("inf")

autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1=dim_u1, dim_z=dim_z, use_pos_encoding=True, num_frequencies=6).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)

# Parameter count
cgn_params = parameters_to_vector(cgkn.cgn.parameters()).numel()
enc_params = parameters_to_vector(cgkn.autoencoder.encoder.parameters()).numel()
dec_params = parameters_to_vector(cgkn.autoencoder.decoder.parameters()).numel()
print(f"cgn #parameters:      {cgn_params:,}")
print(f"encoder #parameters:  {enc_params:,}")
print(f"decoder #parameters:  {dec_params:,}")
print(f"TOTAL #parameters:    {cgn_params + enc_params + dec_params:,}")

STAGE1_CKPT_PATH = r"../model/LaCGKN_stage1.pt"

# """
for ep in range(1, epochs_stage1 + 1):
    cgkn.train()
    t0 = time.time()
    train_loss_forecast_u1 = train_loss_forecast_u2 = train_loss_ae = train_loss_forecast_z = 0.0

    for u1_initial, u2_initial, u1_next, u2_next in train_loader:
        u1_initial, u2_initial, u1_next, u2_next = u1_initial.to(device), u2_initial.to(device), u1_next.to(device), u2_next.to(device)

        # Randomly choose I tracers out of I_total
        tracer_idx = torch.randperm(I_total, device=device)[:I]
        u1_initial = torch.index_select(u1_initial, dim=1, index=tracer_idx)  # (B, I, 4)
        u1_next    = torch.index_select(u1_next,    dim=1, index=tracer_idx)  # (B, I, 4)

        # --- AutoEncoder ---
        z_initial = cgkn.autoencoder.encoder(u2_initial)                # (B, 2, H_z, W_z)
        u2_initial_ae = cgkn.autoencoder.decoder(z_initial)             # (B, H, W, 2)
        loss_ae = nnF.mse_loss(u2_initial, u2_initial_ae)

        # --- Forecast ---
        z_initial_flat = z_initial.reshape(-1, dim_z)                   # (B, dim_z)
        u1_pred, z_flat_pred = cgkn(u1_initial, z_initial_flat)         # u1_pred:(B,I,4), z:(B,dim_z)
        z_pred = z_flat_pred.view(-1, 2, H_z, W_z)
        u2_pred = cgkn.autoencoder.decoder(z_pred)
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_pred)

        loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast_u1 += loss_forecast_u1.item()
        train_loss_forecast_u2 += loss_forecast_u2.item()
        train_loss_ae          += loss_ae.item()
        train_loss_forecast_z  += loss_forecast_z.item()

    train_loss_forecast_u1 /= train_num_batches
    train_loss_forecast_u2 /= train_num_batches
    train_loss_ae          /= train_num_batches
    train_loss_forecast_z  /= train_num_batches
    loss_history_stage1["train_forecast_u1"].append(train_loss_forecast_u1)
    loss_history_stage1["train_forecast_u2"].append(train_loss_forecast_u2)
    loss_history_stage1["train_ae"].append(train_loss_ae)
    loss_history_stage1["train_forecast_z"].append(train_loss_forecast_z)

    t1 = time.time()

    # Validation
    if ep % val_per_epochs == 0:
        cgkn.eval()
        val_loss_forecast_u1 = val_loss_forecast_u2 = val_loss_forecast_z = val_loss_ae = 0.0
        with torch.no_grad():
            for u1_initial, u2_initial, u1_next, u2_next in val_loader:
                u1_initial, u2_initial, u1_next, u2_next = map(lambda x: x.to(device), [u1_initial, u2_initial, u1_next, u2_next])

                z_initial = cgkn.autoencoder.encoder(u2_initial)
                u2_initial_ae = cgkn.autoencoder.decoder(z_initial) 
                z_initial_flat = z_initial.reshape(-1, dim_z)
                u1_pred, z_flat_pred = cgkn(u1_initial, z_initial_flat)
                z_pred = z_flat_pred.view(-1, 2, H_z, W_z)
                u2_pred = cgkn.autoencoder.decoder(z_pred)
                z_next = cgkn.autoencoder.encoder(u2_next)

                val_loss_forecast_u1 += nnF.mse_loss(u1_next, u1_pred).item()
                val_loss_forecast_u2 += nnF.mse_loss(u2_next, u2_pred).item()
                val_loss_ae          += nnF.mse_loss(u2_initial, u2_initial_ae).item()
                val_loss_forecast_z  += nnF.mse_loss(z_next, z_pred).item()

        val_loss_forecast_u1 /= len(val_loader)
        val_loss_forecast_u2 /= len(val_loader)
        val_loss_ae          /= len(val_loader)
        val_loss_forecast_z  /= len(val_loader)
        val_loss_total = val_loss_forecast_u1 + val_loss_forecast_u2 + val_loss_ae + val_loss_forecast_z
        loss_history_stage1["val_forecast_u1"].append(val_loss_forecast_u1)
        loss_history_stage1["val_forecast_u2"].append(val_loss_forecast_u2)
        loss_history_stage1["val_ae"].append(val_loss_ae)
        loss_history_stage1["val_forecast_z"].append(val_loss_forecast_z)

        improved = val_loss_total < best_val_loss
        if improved:
            best_val_loss = val_loss_total
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": cgkn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_total,
                },
                STAGE1_CKPT_PATH,
            )
        print(
            f"ep {ep:d} time {t1 - t0:.4f} | "
            f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f}  train_z: {train_loss_forecast_z:.4f}  ae: {train_loss_ae:.4f} | "
            f"val_u1: {val_loss_forecast_u1:.4f}  val_u2: {val_loss_forecast_u2:.4f}  val_z: {val_loss_forecast_z:.4f}  val_ae: {val_loss_ae:.4f}  val_loss_total: {val_loss_total:.4f} "
            f"{'✅' if improved else ''}"
        )
    else:
        loss_history_stage1["val_forecast_u1"].append(np.nan)
        loss_history_stage1["val_forecast_u2"].append(np.nan)
        loss_history_stage1["val_ae"].append(np.nan)
        loss_history_stage1["val_forecast_z"].append(np.nan)
        print(
            f"ep {ep:d} time {t1 - t0:.4f} | "
            f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f}  train_z: {train_loss_forecast_z:.4f}  ae: {train_loss_ae:.4f}"
        )

np.savez(r"../model/LaCGKN_stage1_loss_history.npz", **loss_history_stage1)
# """
# Load best stage1 model
ckpt = torch.load(STAGE1_CKPT_PATH, map_location=device, weights_only=False)
cgkn.load_state_dict(ckpt["model_state_dict"])


###############################################
################ Test Stage 1 #################
###############################################

cgkn.eval()

# One-Step Prediction
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
with torch.no_grad():
    test_z_flat = cgkn.autoencoder.encoder(test_u2).view(Ntest, dim_z)
    test_u1_pred, test_z_flat_pred = cgkn(test_u1, test_z_flat)
    test_z_pred = test_z_flat_pred.view(Ntest, 2, H_z, W_z)
    test_u2_pred = cgkn.autoencoder.decoder(test_z_pred)
MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
print("MSE1:", MSE1.item())
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")
np.save(r"../data/LaCGKN_xy_unit_OneStepPrediction_stage1.npy", test_u1_pred.to("cpu"))
np.save(r"../data/LaCGKN_psi_OneStepPrediction_stage1.npy", test_u2_pred.to("cpu"))

# Multi-step Prediction
si = 9000
steps = 50
u1_initial = test_u1[si:si+1].to(device)
u2_initial = test_u2[si:si+1].to(device)
test_u1_pred = torch.zeros(steps+1, *test_u1.shape[1:], device=device)
test_u2_pred = torch.zeros(steps+1, *test_u2.shape[1:], device=device)
test_u1_pred[0] = u1_initial[0]
test_u2_pred[0] = u2_initial[0]
with torch.no_grad():
    for n in range(1, steps + 1):
        z_initial_flat = cgkn.autoencoder.encoder(u2_initial).view(-1, dim_z)
        u1_pred, z_pred_flat = cgkn(u1_initial, z_initial_flat)
        z_pred = z_pred_flat.view(-1, 2, H_z, W_z)
        u2_pred = cgkn.autoencoder.decoder(z_pred)
        test_u1_pred[n] = u1_pred[0]
        test_u2_pred[n] = u2_pred[0]
        u1_initial = u1_pred
        u2_initial = u2_pred
np.save(r"../data/LaCGKN_xy_unit_MultiStepPrediction_stage1.npy", test_u1_pred.to("cpu"))
np.save(r"../data/LaCGKN_psi_MultiStepPrediction_stage1.npy", test_u2_pred.to("cpu"))


###############################################
############### Noise Coefficient #############
###############################################

def compute_sigma_hat(train_u1, train_u2, cgkn, dim_u1, dim_z, batch_size=100, device="cuda"):
    Ntrain = train_u1.size(0)
    train_u1 = train_u1.to(device)
    train_u2 = train_u2.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, Ntrain, batch_size):
            batch_u1 = train_u1[i:i+batch_size]
            batch_u2 = train_u2[i:i+batch_size]
            batch_z_flat = cgkn.autoencoder.encoder(batch_u2).view(batch_u2.size(0), dim_z)
            batch_u1_pred, _ = cgkn(batch_u1, batch_z_flat)
            preds.append(batch_u1_pred)
        train_u1_pred = torch.cat(preds, dim=0)
    sigma_hat = torch.zeros(dim_u1 + dim_z, device="cpu")
    sigma_hat[:dim_u1] = torch.sqrt(torch.mean((train_u1[1:] - train_u1_pred[:-1])**2, dim=0)).view(-1).to("cpu")
    sigma_hat[dim_u1:] = 0.1  # sigma2 manually set
    return sigma_hat

sigma_hat = compute_sigma_hat(train_u1[:, :I], train_u2, cgkn, dim_u1, dim_z, batch_size=100, device=device)
torch.save(sigma_hat, "../data/LaCGKN_sigma_hat.pt")
# sigma_hat = torch.load("../data/LaCGKN_sigma_hat.pt", weights_only=True)


###############################################
##################  CGFilter  #################
###############################################

def CGFilter(cgkn, sigma, u1, mu0, R0): # u1: (Nt, I, 4, 1), mu0: (dim_z, 1), R0: (dim_z, dim_z)
    device = u1.device
    Nt = u1.shape[0]
    u1_flat = u1.view(Nt, -1, 1)
    dim_u1 = u1_flat.shape[1]
    dim_z = mu0.shape[0]
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    mu_pred = torch.zeros((Nt, dim_z, 1)).to(device)
    R_pred = torch.zeros((Nt, dim_z, dim_z)).to(device)
    mu_pred[0] = mu0
    R_pred[0] = R0
    for n in range(1, Nt):
        x, y = unit2xy(u1[n-1].permute(2,0,1))
        pos = torch.stack([x, y], dim=-1) # (1, I, 2)
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(pos)]

        K0 = torch.linalg.solve(s1 @ s1.T + g1 @ R0 @ g1.T, g1 @ R0 @ g2.T).T
        mu1 = f2 + g2@mu0 + K0@(u1_flat[n]-f1-g1@mu0)
        R1 = g2@R0@g2.T + s2@s2.T - K0@g1@R0@g2.T
        R1 = 0.5 * (R1 + R1.T)

        mu_pred[n] = mu1
        R_pred[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_pred, R_pred)

def CGFilter_batch(cgkn, sigma, u1, mu0, R0): # u1: (B, Nt, I, 4, 1), mu0: (B, dim_z, 1), R0: (B, dim_z, dim_z)
    device = u1.device
    B, Nt = u1.shape[:2]
    u1_flat = u1.view(B, Nt, -1, 1)                                     # (B, Nt, dim_u1, 1)
    dim_u1 = u1_flat.shape[2]
    dim_z = mu0.shape[1]
    s1 = torch.diag(sigma[:dim_u1]).to(device)                          # (dim_u1, dim_u1)
    s2 = torch.diag(sigma[dim_u1:]).to(device)                          # (dim_z, dim_z)
    s1_cov = s1 @ s1.T                                                  # (dim_u1, dim_u1)
    s2_cov = s2 @ s2.T                                                  # (dim_z, dim_z)
    mu_pred = torch.zeros((B, Nt, dim_z, 1), device=device)
    R_pred = torch.zeros((B, Nt, dim_z, dim_z), device=device)
    mu_pred[:, 0] = mu0
    R_pred[:, 0] = R0
    for n in range(1, Nt):
        x, y = unit2xy(u1[:, n - 1].squeeze(-1))                        # u1[:, n-1]: (B, I, 4, 1)
        pos = torch.stack([x, y], dim=-1)                               # (B, I, 2)
        f1, g1, f2, g2 = cgkn.cgn(pos)  # shapes: (B, dim_u1, 1), (B, dim_u1, dim_z), ...

        # Kalman gain: solve per batch
        S = s1_cov + torch.bmm(g1, torch.bmm(R0, g1.transpose(1, 2)))    # (B, dim_u1, dim_u1)
        Cross = torch.bmm(g1, torch.bmm(R0, g2.transpose(1, 2)))         # (B, dim_u1, dim_z)
        K = torch.linalg.solve(S, Cross).transpose(1, 2)                 # (B, dim_z, dim_u1)
        innov = u1_flat[:, n] - f1 - torch.bmm(g1, mu0)                  # (B, dim_u1, 1)
        mu1 = f2 + torch.bmm(g2, mu0) + torch.bmm(K, innov)              # (B, dim_z, 1)
        R1 = torch.bmm(g2, torch.bmm(R0, g2.transpose(1, 2))) + s2_cov \
             - torch.bmm(K, torch.bmm(g1, torch.bmm(R0, g2.transpose(1, 2))))  # (B, dim_z, dim_z)
        R1 = 0.5 * (R1 + R1.transpose(1, 2))  # Ensure symmetry

        mu_pred[:, n] = mu1
        R_pred[:, n] = R1
        mu0 = mu1
        R0 = R1
    return mu_pred, R_pred


###############################################
################ Train Stage 2 ################
###############################################

# Stage 2: Train cgkn with loss_forecast_u + loss_ae + loss_forecast_z + loss_da 
torch.manual_seed(SEED_STAGE2)
np.random.seed(SEED_STAGE2)

short_steps = 2
long_steps = 100
cut_point = 20

epochs_stage2 = 500
train_batch_size = 200
train_batch_size_da = 10
val_batch_size = 10
val_per_epochs = 10
train_num_batches = int(Ntrain / train_batch_size)
val_per_itrs = int(Ntrain / train_batch_size * val_per_epochs)
Niters = epochs_stage2 * train_num_batches
loss_history_stage2 = {
    "train_forecast_u1": [],
    "train_forecast_u2": [],
    "train_ae": [],
    "train_forecast_z": [],
    "train_da": [],
    "val_forecast_u1": [],
    "val_forecast_u2": [],
    "val_ae": [],
    "val_forecast_z": [],
    "val_da": [],
}
best_val_loss = float("inf")

optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)

STAGE2_CKPT_PATH = r"../model/LaCGKN_stage2.pt"
# """
for itr in range(1, Niters + 1):
    t0 = time.time()
    cgkn.train()

    # Multi-step forecast by short_steps
    head_idx_short = torch.from_numpy(np.random.choice(Ntrain - short_steps + 1, size=train_batch_size, replace=False))
    u1_short = torch.stack([train_u1[idx:idx + short_steps] for idx in head_idx_short], dim=1).to(device)  # (S, B, I_total, 4)
    u2_short = torch.stack([train_u2[idx:idx + short_steps] for idx in head_idx_short], dim=1).to(device)  # (S, B, H, W, 2)

    # choose tracer subset
    tracer_idx = torch.randperm(I_total, device=device)[:I]
    u1_short = torch.index_select(u1_short, dim=2, index=tracer_idx)            # (S, B, I, 4)

    # --- AutoEncoder ---
    z_short = cgkn.autoencoder.encoder(u2_short.view(-1, *u2_short.shape[2:]))  # (S*B, 2, H_z, W_z)
    u2_ae_short = cgkn.autoencoder.decoder(z_short).view_as(u2_short)           # (S, B, H, W, 2)
    loss_ae = nnF.mse_loss(u2_short, u2_ae_short)

    # --- Forecast ---
    z_short = z_short.view(short_steps, train_batch_size, 2, H_z, W_z)          # (S, B, 2, H_z, W_z)
    z_flat_short = z_short.reshape(short_steps, train_batch_size, dim_z)        # (S, B, dim_z)
    u1_preds = [u1_short[0]]
    z_preds = [z_flat_short[0]]
    for n in range(1, short_steps):
        u1_n, z_n = cgkn(u1_preds[-1], z_preds[-1])
        u1_preds.append(u1_n)
        z_preds.append(z_n)
    u1_short_pred = torch.stack(u1_preds, dim=0)
    z_flat_short_pred = torch.stack(z_preds, dim=0)
    loss_forecast_z = nnF.mse_loss(z_flat_short[1:], z_flat_short_pred[1:])

    z_short_pred = z_flat_short_pred.reshape(short_steps, train_batch_size, 2, H_z, W_z)
    u2_short_pred = cgkn.autoencoder.decoder(z_short_pred.view(-1, 2, H_z, W_z)).view_as(u2_short)
    loss_forecast_u1 = nnF.mse_loss(u1_short[1:], u1_short_pred[1:])
    loss_forecast_u2 = nnF.mse_loss(u2_short[1:], u2_short_pred[1:])

    # --- DA ---
    head_idx_long = torch.from_numpy(np.random.choice(Ntrain - long_steps + 1, size=train_batch_size_da, replace=False))
    u1_long = torch.stack([train_u1[i:i + long_steps] for i in head_idx_long]).to(device)  # (B, Nt, I_total, 4)
    u2_long = torch.stack([train_u2[i:i + long_steps] for i in head_idx_long]).to(device)  # (B, Nt, H, W, 2)
    # tracer subset
    u1_long = torch.index_select(u1_long, dim=2, index=tracer_idx)              # (B, Nt, I, 4)
    mu0 = torch.zeros(train_batch_size_da, dim_z, 1, device=device)
    R0 = (0.1 * torch.eye(dim_z, device=device)).expand(train_batch_size_da, dim_z, dim_z)
    mu_z_flat_pred_long, _ = CGFilter_batch(cgkn, sigma_hat.to(device), u1_long.unsqueeze(-1), mu0, R0)  # (B, Nt, dim_z, 1)
    mu_z_pred_long = mu_z_flat_pred_long[:, cut_point:].squeeze(-1).reshape(-1, 2, H_z, W_z)             # (B*(Nt-cut), 2, H_z, W_z)
    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long)                     # (B*(Nt-cut), H, W, 2)
    mu_pred_long = mu_pred_long.view(train_batch_size_da, long_steps - cut_point, *mu_pred_long.shape[1:]) # (B, (Nt-cut), H, W, 2)
    u2_long_target = u2_long[:, cut_point:]                                     # (B, Nt-cut, H, W, 2)
    loss_da = nnF.mse_loss(mu_pred_long, u2_long_target)

    loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z + loss_da
    if torch.isnan(loss_total):
        print(itr, "nan")
        continue

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    loss_history_stage2["train_forecast_u1"].append(loss_forecast_u1.item())
    loss_history_stage2["train_forecast_u2"].append(loss_forecast_u2.item())
    loss_history_stage2["train_forecast_z"].append(loss_forecast_z.item())
    loss_history_stage2["train_ae"].append(loss_ae.item())
    loss_history_stage2["train_da"].append(loss_da.item())

    t1 = time.time()

    # Validation
    if itr % val_per_itrs == 0:
        cgkn.eval()
        val_loss_forecast_u1 = 0.
        val_loss_forecast_u2 = 0.
        val_loss_forecast_z = 0.
        val_loss_ae = 0.
        val_loss_physics = 0.
        val_loss_da = 0.
        val_total_samples = 0
        with torch.no_grad():
            for i in range(0, Nval - short_steps + 1, val_batch_size):
                B = min(val_batch_size, Nval - short_steps + 1 - i)
                u1_batch = torch.stack([val_u1[i+j:i+j+short_steps] for j in range(B)], dim=1).to(device)
                u2_batch = torch.stack([val_u2[i+j:i+j+short_steps] for j in range(B)], dim=1).to(device)

                # AE loss
                z_val = cgkn.autoencoder.encoder(u2_batch.view(-1, *u2_batch.shape[2:]))
                u2_val_ae = cgkn.autoencoder.decoder(z_val).view_as(u2_batch)
                val_loss_ae += nnF.mse_loss(u2_batch, u2_val_ae, reduction='mean').item() * B

                # Forecast loss
                z_val = z_val.view(short_steps, B, *z_val.shape[1:])
                z_flat_val = z_val.view(short_steps, B, -1)
                u1_preds = [u1_batch[0]]
                z_preds = [z_flat_val[0]]
                for n in range(1, short_steps):
                    u1_pred, z_pred = cgkn(u1_preds[-1], z_preds[-1])
                    u1_preds.append(u1_pred)
                    z_preds.append(z_pred)
                u1_preds = torch.stack(u1_preds)
                z_preds = torch.stack(z_preds)
                val_loss_forecast_z += nnF.mse_loss(z_flat_val[1:], z_preds[1:], reduction='mean').item() * B

                z_pred_reshaped = z_preds.view(short_steps, B, 2, H_z, W_z)
                u2_preds = cgkn.autoencoder.decoder(z_pred_reshaped.view(-1, 2, H_z, W_z)).view_as(u2_batch)
                val_loss_forecast_u1 += nnF.mse_loss(u1_batch[1:], u1_preds[1:], reduction='mean').item() * B
                val_loss_forecast_u2 += nnF.mse_loss(u2_batch[1:], u2_preds[1:], reduction='mean').item() * B

                val_total_samples += B
            val_loss_forecast_u1 = val_loss_forecast_u1 / val_total_samples
            val_loss_forecast_u2 = val_loss_forecast_u2 / val_total_samples
            val_loss_forecast_z = val_loss_forecast_z / val_total_samples
            val_loss_ae = val_loss_ae / val_total_samples

            del u1_batch, u2_batch, z_val, z_flat_val, u2_val_ae, u1_preds, z_preds, z_pred_reshaped
            torch.cuda.empty_cache()

            # DA loss
            val_u1_gpu = val_u1.to(device)
            val_u2_gpu = val_u2.to(device)
            val_mu_z_pred = CGFilter(cgkn, sigma_hat.to(device), val_u1_gpu.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1).reshape(-1, 2, H_z, W_z)
            val_mu_pred = cgkn.autoencoder.decoder(val_mu_z_pred)
            val_loss_da = nnF.mse_loss(val_u2_gpu[cut_point:], val_mu_pred[cut_point:]).item()

            # Free DA intermediates
            del val_mu_z_pred, val_mu_pred, val_u1_gpu, val_u2_gpu
            torch.cuda.empty_cache()
            
            loss_history_stage2["val_forecast_u1"].append(val_loss_forecast_u1)
            loss_history_stage2["val_forecast_u2"].append(val_loss_forecast_u2)
            loss_history_stage2["val_forecast_z"].append(val_loss_forecast_z)
            loss_history_stage2["val_ae"].append(val_loss_ae)
            loss_history_stage2["val_da"].append(val_loss_da)

            val_loss_total = val_loss_da
            improved = val_loss_total < best_val_loss
            if improved:
                best_val_loss = val_loss_total
                torch.save(
                    {
                        "itr": itr,
                        "model_state_dict": cgkn.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss_total,
                    },
                    STAGE2_CKPT_PATH
                )
            print(
                f"ep {itr // train_num_batches:d} time {t1 - t0:.4f} | "
                f"train_u1: {loss_forecast_u1:.4f}  train_u2: {loss_forecast_u2:.4f}  train_z: {loss_forecast_z:.4f} "
                f"ae: {loss_ae:.4f}  da: {loss_da:.4f} | "
                f"val_u1: {val_loss_forecast_u1:.4f}  val_u2: {val_loss_forecast_u2:.4f}  val_z: {val_loss_forecast_z:.4f} "
                f"val_ae: {val_loss_ae:.4f}  val_da: {val_loss_da:.4f}  val_total: {val_loss_total:.4f} "
                f"{'✅' if improved else ''}"
            )
    else:
        loss_history_stage2["val_forecast_u1"].append(np.nan)
        loss_history_stage2["val_forecast_u2"].append(np.nan)
        loss_history_stage2["val_forecast_z"].append(np.nan)
        loss_history_stage2["val_ae"].append(np.nan)
        loss_history_stage2["val_da"].append(np.nan)
        if itr % train_num_batches == 0:
            print(
                f"ep {itr // train_num_batches:d} time {t1 - t0:.4f} | "
                f"train_u1: {loss_forecast_u1:.4f}  train_u2: {loss_forecast_u2:.4f}  train_z: {loss_forecast_z:.4f} "
                f"ae: {loss_ae:.4f}  da: {loss_da:.4f}"
            )

np.savez(r"../model/LaCGKN_stage2_loss_history.npz", **loss_history_stage2)
# """
# Load best stage2 model
ckpt2 = torch.load(STAGE2_CKPT_PATH, map_location=device, weights_only=False)
cgkn.load_state_dict(ckpt2["model_state_dict"])


###############################################
########## UQ via Residual Analysis ###########
###############################################

# CGKN for Data Assimilation (Training Data)
def run_da(cgkn, train_u1, train_u2, sigma_hat, I_total, I, dim_z, H_z, W_z, cut_point, batch_steps=1000, device="cuda"):
    cgkn.eval()
    train_u1 = train_u1.to(device)
    train_u2 = train_u2.to(device)
    train_mu_preds = []
    with torch.no_grad():
        for i in range(0, train_u1.size(0), batch_steps):
            tracer_idx = torch.randperm(I_total, device=device)[:I]  # choose I tracers
            train_u1_batch = train_u1[i:i+batch_steps, tracer_idx]   # (B, I, features)
            train_mu_z_flat_pred_batch = CGFilter(
                cgkn, sigma_hat.to(device),
                train_u1_batch.unsqueeze(-1),
                mu0=torch.zeros(dim_z, 1, device=device),
                R0=0.1*torch.eye(dim_z, device=device)
            )[0].squeeze(-1)
            train_mu_z_pred_batch = train_mu_z_flat_pred_batch.reshape(-1, 2, H_z, W_z)
            train_mu_pred_batch = cgkn.autoencoder.decoder(train_mu_z_pred_batch)
            train_mu_preds.append(train_mu_pred_batch)
        train_mu_pred = torch.cat(train_mu_preds, dim=0)
    mse = nnF.mse_loss(train_u2[cut_point:], train_mu_pred[cut_point:]).item()
    print("MSE2_DA (training):", mse)
    return train_mu_pred.to('cpu')
train_mu_pred = run_da(cgkn, train_u1, train_u2, sigma_hat, I_total, I, dim_z, H_z, W_z, cut_point, batch_steps=1000, device=device)
torch.save(train_mu_pred, "../data/LaCGKN_train_mu_pred.pt")
# # Load if train_mu_pred has been saved
# train_mu_pred = torch.load("../data/LaCGKN_train_mu_pred.pt")

# Target Variable: Residual (squared error of posterior mean)
train_mu_se = (train_u2[cut_point:] - train_mu_pred[cut_point:])**2

class UncertaintyNet(nn.Module):
    def __init__(self, dim_u1, dim_u2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_u1, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, dim_u2))

    def forward(self, x):
        out = self.net(x)
        return out

uncertainty_net = UncertaintyNet(dim_u1, dim_u2).to(device)
epochs = 500
train_batch_size = 2000
train_tensor = torch.utils.data.TensorDataset(train_u1[cut_point:, :I], train_mu_se)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_uncertainty_history = []
optimizer = torch.optim.Adam(uncertainty_net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for ep in range(1, epochs+1):
    uncertainty_net.train()
    start_time = time.time()
    train_loss_uncertainty = 0.
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device).reshape(x_batch.size(0), dim_u1)
        y_batch = y_batch.to(device).reshape(y_batch.size(0), dim_u2)
        optimizer.zero_grad()
        preds = uncertainty_net(x_batch)
        loss = nnF.mse_loss(preds, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_uncertainty += loss.item()
    train_loss_uncertainty /= train_num_batches
    train_loss_uncertainty_history.append(train_loss_uncertainty)
    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss uncertainty:", round(train_loss_uncertainty, 4))

torch.save({
    'model_state_dict': uncertainty_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epochs,
    'loss': train_loss_uncertainty_history,
}, "../model/UQNet.pt")

# # Load if UQNet has been saved
# checkpoint = torch.load("../model/UQNet.pt", map_location=device)
# uncertainty_net.load_state_dict(checkpoint['model_state_dict'])


###############################################
################## Test cgkn ##################
###############################################

# One-Step Prediction
batch_size = 100
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
test_u1_preds = []
test_u2_preds = []
cgkn.eval()
with torch.no_grad():
    for i in range(0, Ntest, batch_size):
        test_u1_batch = test_u1[i:i+batch_size]
        test_u2_batch = test_u2[i:i+batch_size]
        test_z_flat_batch = cgkn.autoencoder.encoder(test_u2_batch).view(batch_size, dim_z)
        test_u1_pred_batch, test_z_flat_pred_batch = cgkn(test_u1_batch, test_z_flat_batch)
        test_z_pred_batch = test_z_flat_pred_batch.view(batch_size, 2, H_z, W_z)
        test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_pred_batch)
        test_u1_preds.append(test_u1_pred_batch)
        test_u2_preds.append(test_u2_pred_batch)
    test_u1_pred = torch.cat(test_u1_preds, dim=0)
    test_u2_pred = torch.cat(test_u2_preds, dim=0)
MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
print("MSE1:", MSE1.item())
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")
np.save(r"../data/LaCGKN_xy_unit_OneStepPrediction_stage2.npy", test_u1_pred.to("cpu"))
np.save(r"../data/LaCGKN_psi_OneStepPrediction_stage2.npy", test_u2_pred.to("cpu"))

# Data Assimilation
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
cgkn.eval()
with torch.no_grad():
    test_mu_z_flat_pred = CGFilter(cgkn, sigma_hat.to(device), test_u1.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
    test_mu_z_pred = test_mu_z_flat_pred.reshape(-1, 2, H_z, W_z)
    test_mu_pred = cgkn.autoencoder.decoder(test_mu_z_pred)
MSE2_DA = nnF.mse_loss(test_u2[cut_point:], test_mu_pred[cut_point:])
print("MSE2_DA:", MSE2_DA.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")
np.save(r"../data/LaCGKN_psi_DA.npy", test_mu_pred.to("cpu"))

# Uncertainty Quantification
test_u1 = test_u1.to(device)
uncertainty_net.eval()
with torch.no_grad():
    test_mu_se_pred = uncertainty_net(test_u1.reshape(-1, dim_u1)).reshape(-1,*test_u2.shape[1:]).cpu()
np.save(r"../data/LaCGKN_psi_DA_se.npy", test_mu_se_pred)
