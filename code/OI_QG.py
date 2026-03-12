import numpy as np
from os.path import dirname, join as pjoin
from QG_tracer import QG_tracer
from time import time

class LagrangianOI:
    """
    Molcard et al. (2003) single-step Lagrangian OI for drifter/tracer positions.
    - Forms observed/model Lagrangian velocities from positions over Δt (finite difference)
    - Applies Gaussian-localized velocity increment to the grid (surface layer)
    - Converts velocity increment → vorticity increment → streamfunction increment
    - Projects surface-layer streamfunction increment to the 2nd layer via linear regression
    References: Eq. (5) for the update; Fig. 2 implementation (modify at t0). 
    """
    def __init__(
        self, 
        Nx: int,
        dt: float,
        dt_obs: float,
        kd: float,
        psi2q_mat: np.ndarray,
        plus_hk: np.ndarray,
        model,                       # QG_tracer model
        sigma_pos: float,            # position std (same units as positions)
        sigma_b_vel: float,          # background vel error std (same units as u,v)
        gauss_radius: float = 1.0,   # Gaussian half-width in grid units (~ grid spacing)
        beta_u = 0.,                 # regression coefficient, scalar or (Nx,Nx) array
        beta_v = 0.,                 # regression coefficient, scalar or (Nx,Nx) array
        two_pi: float = 2*np.pi
    ):
        self.Nx = Nx
        self.dt = dt
        self.dt_obs = dt_obs
        self.obs_freq_timestep = int(round(self.dt_obs / self.dt))
        self.kd = kd
        self.psi2q_mat = psi2q_mat.astype(np.complex128)    # (Nx,Nx,2,2)
        self.plus_hk = plus_hk.astype(np.complex128)        # (Nx,Nx,2,1)
        self.model = model
        self.two_pi = two_pi
        sigma_o_vel = sigma_pos / dt_obs # sigma_o^2 = sigma_pos^2 / Δt^2 (position→velocity)
        self.gain = 1.0 / (1.0 + (sigma_o_vel**2) / (sigma_b_vel**2))
        # Gaussian localization parameters
        self.sig = gauss_radius
        self.sig2 = self.sig**2
        # Fourier wavenumbers for derivatives/inversion
        k = np.fft.fftfreq(Nx) * Nx
        self.KX, self.KY = np.meshgrid(k, k, indexing='xy')
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0,0] = 1.0 # avoid divide by zero; zero-mean adjust later
        # Linear projection from layer 1 to layer 2
        self.beta_u = beta_u 
        self.beta_v = beta_v

    # ---------------- utilities ----------------
    def _torus_delta(self, a, b):
        """Periodic minimal difference on [0, 2π)."""
        d = a - b
        d = (d + np.pi) % self.two_pi - np.pi
        return d

    def _positions_to_vel(self, pos_now, pos_prev):
        """Finite-difference Lagrangian velocity over Δt, with periodic wrap."""
        dv = self._torus_delta(pos_now, pos_prev) / self.dt_obs
        return dv  # shape (..., 2)

    def _grid_gaussian(self, xg, yg, x0, y0):
        """Gaussian weight on the torus in grid coordinates."""
        # distances on torus in grid units
        dx = (xg - x0 + self.Nx/2) % self.Nx - self.Nx/2
        dy = (yg - y0 + self.Nx/2) % self.Nx - self.Nx/2
        return np.exp(-(dx*dx + dy*dy) / (2*self.sig2))

    def _vel_from_psi(self, psi):
        """u=ψ_y, v=-ψ_x via spectral derivatives, per layer."""
        psik = np.fft.fft2(psi, axes=(0,1))
        ikx = 1j * self.KX[..., None]
        iky = 1j * self.KY[..., None]
        uk =  iky * psik
        vk = -ikx * psik
        u = np.real(np.fft.ifft2(uk, axes=(0,1)))
        v = np.real(np.fft.ifft2(vk, axes=(0,1)))
        return u, v

    def _curl_from_uv(self, u, v):
        """vorticity ζ = v_x - u_y (spectral)."""
        uk = np.fft.fft2(u, axes=(0,1))
        vk = np.fft.fft2(v, axes=(0,1))
        ikx = 1j * self.KX[..., None]
        iky = 1j * self.KY[..., None]
        zeta_k = ikx * vk - iky * uk
        zeta = np.real(np.fft.ifft2(zeta_k, axes=(0,1)))
        return zeta

    def _psi_increment_from_du_dv(self, du, dv):
        """Find δψ from δu,δv by solving ∇²δψ = -δζ"""
        delta_zeta = self._curl_from_uv(du, dv) # (Nx,Nx,2)
        delta_zeta_k = np.fft.fft2(delta_zeta, axes=(0,1))
        delta_psi_k = + delta_zeta_k / self.K2[..., None]
        delta_psi_k[0,0,:] = 0.0  # remove mean
        delta_psi = np.real(np.fft.ifft2(delta_psi_k, axes=(0,1)))
        return delta_psi

    def _q_from_psi(self, psi):
        """q from ψ."""
        psik = np.fft.fft2(psi, axes=(0,1))                  # (Nx,Nx,2)
        psik = psik[..., None]                               # (Nx,Nx,2,1)
        qk = (self.psi2q_mat @ psik + self.plus_hk)[..., 0]  # (Nx,Nx,2)
        q = np.real(np.fft.ifft2(qk, axes=(0,1)))
        return q

    # ------------- one OI analysis at time window [t0, t0+Δt_obs] -------------
    def analyze(
        self,
        psi_b: np.ndarray,            # background ψ at t0, shape (Nx,Nx,2)
        xy_obs_prev: np.ndarray,      # observed positions at t0, shape (I,2) in [0,2π)
        xy_obs_now: np.ndarray        # observed positions at t0+Δt_obs, shape (I,2)
    ):
        """
        Returns: psi_a at t0 (analysis), and a forecast to t0+Δt_obs using psi_a.
        Steps: forward model drifters from obs(t0) under ψ_b → v^b; form v^o from obs;
               apply u/v increments on layer 1; curl→δψ1; project to layer 2; ψ_a = ψ_b + δψ;
               advance with model from t0 to t0+Δt_obs using ψ_a.
        """
        Nx = self.Nx
        I = xy_obs_now.shape[0]

        # ------ model forecast of Lagrangian vel from r^o(t0) under ψ_b ------
        q_b = self._q_from_psi(psi_b)
        # model.forward_ens expects qp_ens shape (Nens,Nx,Nx,2)
        qp0_ens = q_b[None, ...]
        x0 = xy_obs_prev[None, :, 0]
        y0 = xy_obs_prev[None, :, 1]
        psi1_k_ens, x_traj, y_traj, _ = self.model.forward_ens(
            ens=1, Nt=self.obs_freq_timestep, dt=self.dt,
            qp_ens=qp0_ens, I=I, x0=x0, y0=y0
        )
        xy_b_now = np.stack([x_traj[0, :, -1], y_traj[0, :, -1]], axis=1)
        vb = self._positions_to_vel(xy_b_now, xy_obs_prev)     # (I,2)

        # ------------- observed Lagrangian vel from two positions --------------
        vo = self._positions_to_vel(xy_obs_now, xy_obs_prev)   # (I,2)

        # ------ innovation δv and Gaussian-weighted increment on grid (layer 1 only) ------
        delta_v = vo - vb                                      # (I,2)
        # convert obs pos (0..2π) to grid indices; take the observed positions as ICs to model
        x0g = (xy_obs_prev[:,0] / self.two_pi) * Nx
        y0g = (xy_obs_prev[:,1] / self.two_pi) * Nx
        # accumulate δu, δv on grid for layer 1; layer 2 zeros (filled later via projection)
        du = np.zeros((Nx, Nx, 2), dtype=float)
        dv = np.zeros((Nx, Nx, 2), dtype=float)
        # precomputed gain 
        gain = self.gain
        # stencil limit (~4σ for efficiency) MAY NEED TUNE THIS
        rcut = int(np.ceil(4*self.sig))
        for m in range(I):
            # local box around (x0g[m], y0g[m])
            ix0 = int(np.floor(x0g[m])) % Nx
            iy0 = int(np.floor(y0g[m])) % Nx
            xs = (np.arange(ix0-rcut, ix0+rcut+1) % Nx)
            ys = (np.arange(iy0-rcut, iy0+rcut+1) % Nx)
            XX, YY = np.meshgrid(xs, ys, indexing='xy')
            # Gaussian localization centered at model drifter location at t0 (observed location here)
            G = self._grid_gaussian(XX, YY, x0g[m], y0g[m])
            # increment (u,v) on layer 1
            du[YY, XX, 0] += gain * G * delta_v[m, 0]
            dv[YY, XX, 0] += gain * G * delta_v[m, 1]

        # --------- column-wise linear projection to layer 2 on (δu,δv) ---------
        du[:, :, 1] = self.beta_u * du[:, :, 0]
        dv[:, :, 1] = self.beta_v * dv[:, :, 0]

        # ------------------------- δψ from (δu,δv) -----------------------------
        delta_psi = self._psi_increment_from_du_dv(du, dv)  # (Nx,Nx,2)

        # -------------------------- analysis at t0 -----------------------------
        psi_a = psi_b + delta_psi

        # ---------------- forecast from t0 → t0+Δt_obs with ψ_a ----------------
        q_a = self._q_from_psi(psi_a)
        qp0_ens = q_a[None, ...]
        psi1_k_ens, x_traj_a, y_traj_a, _ = self.model.forward_ens(
            ens=1, Nt=self.obs_freq_timestep, dt=self.dt,
            qp_ens=qp0_ens, I=I,
            x0=x0, y0=y0
        )
        psi_pred = np.real(np.fft.ifft2(psi1_k_ens[0, -1], axes=(0,1)))  # (Nx, NX, 2)

        return psi_a, psi_pred


np.random.seed(0)

# --------------------- load data --------------------------
data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_data.npz')
data = np.load(datafname)
psi_truth_full = data['psi_truth']
xy_truth_full = data['xy_truth']
sigma_xy = data['sigma_xy'].item()
xy_obs_full = data['xy_obs']
sigma_obs = data['sigma_obs'].item()
dt_obs = data['dt_obs'].item()

# split training and test data set (training means tuning inflation and localzaiton)
train_size = 80000
val_size = 10000
test_size = 10000
data_size = test_size
psi_truth = psi_truth_full[train_size+val_size:train_size+val_size+test_size]
xy_truth = xy_truth_full[train_size+val_size:train_size+val_size+test_size]
xy_obs = xy_obs_full[train_size+val_size:train_size+val_size+test_size]

# ---------------------- model parameters ---------------------
kd = 10 # Nondimensional deformation wavenumber
kb = np.sqrt(22) # Nondimensional beta wavenumber, beta = kb^2 
U = 1 # Zonal shear flow
r = 9 # Nondimensional Ekman friction coefficient
nu = 1e-12 # Coefficient of biharmonic vorticity diffusion
H = 40 # Topography parameter
dt = 2e-3 # Time step size
Nx = 64 # Number of grid points in each direction
dx = 2 * np.pi / Nx # Domain: [0, 2pi)^2
X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
topo = H * (np.cos(X) + 2 * np.cos(2 * Y)) # topography
topo -= np.mean(topo)
hk = np.fft.fft2(topo)
model = QG_tracer(K=Nx, kd=kd, kb=kb, U=U, r=r, nu=nu, H=H, sigma_xy=sigma_xy)

# ------------------- observation parameters ------------------
I = 64 # number of tracers
xy_truth = xy_truth[:, :I, :]
xy_obs = xy_obs[:, :I, :]
obs_error_var = sigma_obs**2
nobstime = xy_obs.shape[0]

# --------------------- DA parameters -----------------------
sigma_b_vels = [1.0]                # (tune);
nsig = len(sigma_b_vels)
localization_radiuses = [1.0]       # ~1 grid; tune 0.8–2.0
nloc = len(localization_radiuses)
# analysis period
iobsbeg = 50
iobsend = None

# ---------------------- initialization -----------------------
# prepare matrix
Kx = np.fft.fftfreq(Nx) * Nx
Ky = np.fft.fftfreq(Nx) * Nx
KX, KY = np.meshgrid(Kx, Ky)
K_square = KX**2 + KY**2
psi2q_mat = (-np.eye(2) * K_square[:, :, None, None]) + (kd**2/2 * np.ones((2,2)) - np.eye(2) * kd**2)
psi2q_mat = psi2q_mat.astype(np.complex128)
plus_hk = np.zeros(((Nx,Nx,2,1)), dtype=complex)
plus_hk[:,:,1,0] = hk
# initial flow field
psi0 = np.random.randn(Nx, Nx, 2) * 0.1

# ---------- calculate layer regression coefficients -----------
# psi_k = np.fft.fft2(psi_truth_full[:train_size], axes=(1,2)) # shape (Nt,Nx,Nx,2)
# u = np.real(np.fft.ifft2(psi_k * 1j * KY[..., None], axes=(1,2)))
# v = np.real(np.fft.ifft2(psi_k * (-1j) * KX[..., None], axes=(1,2)))

# def _regression_slope_map(x, y, axis=0, eps=1e-12):
#     x_anom = x - np.mean(x, axis=axis, keepdims=True)
#     y_anom = y - np.mean(y, axis=axis, keepdims=True)
#     num = np.sum(x_anom * y_anom, axis=axis)         # shape (..., Nx, Nx)
#     den = np.sum(x_anom * x_anom, axis=axis) + eps   # add small ridge for stability
#     beta = num / den
#     return beta

# # Compute per-grid regression coefficients (layer-2 on layer-1)
# beta_u = _regression_slope_map(u[..., 0], u[..., 1], axis=0)   # shape (Nx, Nx)
# beta_v = _regression_slope_map(v[..., 0], v[..., 1], axis=0)   # shape (Nx, Nx)
# betas = {
#     'beta_u': beta_u,
#     'beta_v': beta_v,
# }
# np.savez('../data/oi_regression_coef.npz', **betas)

betas = np.load('../data/oi_regression_coef.npz')
beta_u = betas['beta_u']
beta_v = betas['beta_v']

prior_mse_flow = np.zeros((nobstime,nsig,nloc))
analy_mse_flow = np.zeros((nobstime,nsig,nloc))
prior_mse_flow1 = np.zeros((nobstime,nsig,nloc))
analy_mse_flow1 = np.zeros((nobstime,nsig,nloc))
prior_mse_flow2 = np.zeros((nobstime,nsig,nloc))
analy_mse_flow2 = np.zeros((nobstime,nsig,nloc))
prior_err_flow = np.zeros((nsig,nloc))
analy_err_flow = np.zeros((nsig,nloc))
prior_err_flow1 = np.zeros((nsig,nloc))
analy_err_flow1 = np.zeros((nsig,nloc))
prior_err_flow2 = np.zeros((nsig,nloc))
analy_err_flow2 = np.zeros((nsig,nloc))

t0 = time()
for isig, sigma_b_vel in enumerate(sigma_b_vels):
    print('-------------------------------------')
    print('sigma_b_vel=', sigma_b_vel)
    for iloc, localization_radius in enumerate(localization_radiuses):
        print('localization_radius=', localization_radius)
        print('-------------------------------------')
        oi = LagrangianOI(
            Nx=Nx, dt=dt, dt_obs=dt_obs, kd=kd,
            psi2q_mat=psi2q_mat, plus_hk=plus_hk, model=model,
            sigma_pos=sigma_obs,          
            sigma_b_vel=sigma_b_vel,      
            gauss_radius=localization_radius,
            beta_u = beta_u,
            beta_v = beta_v,
        )
        psi_prior = np.zeros_like(psi_truth)
        psi_analy = np.zeros_like(psi_truth)
        psi_b = psi0.copy()
        psi_prior[0] = psi_b
        
        for iassim in range(1, nobstime):   
            xy_prev = xy_obs[iassim-1, :I, :]   # r^o(t0)
            xy_now  = xy_obs[iassim,   :I, :]   # r^o(t0+Δt)
            psi_a, psi_b = oi.analyze(psi_b, xy_prev, xy_now)  # psi_b becomes forecast to next time
            psi_prior[iassim] = psi_b
            psi_analy[iassim-1] = psi_a
            
        psi_analy[-1] = psi_prior[-1] # the last analysis is not performed because it requires the observation of next step
        
        prior_mse_flow[:, isig, iloc] = np.mean((psi_truth - psi_prior) ** 2, axis=(1,2,3))
        analy_mse_flow[:, isig, iloc] = np.mean((psi_truth - psi_analy) ** 2, axis=(1,2,3))
        prior_mse_flow1[:, isig, iloc] = np.mean((psi_truth[..., 0] - psi_prior[..., 0]) ** 2, axis=(1,2))
        analy_mse_flow1[:, isig, iloc] = np.mean((psi_truth[..., 0] - psi_analy[..., 0]) ** 2, axis=(1,2))
        prior_mse_flow2[:, isig, iloc] = np.mean((psi_truth[..., 1] - psi_prior[..., 1]) ** 2, axis=(1,2))
        analy_mse_flow2[:, isig, iloc] = np.mean((psi_truth[..., 1] - psi_analy[..., 1]) ** 2, axis=(1,2))
        prior_err_flow[isig, iloc] = np.mean(prior_mse_flow[iobsbeg - 1: iobsend, isig, iloc])
        analy_err_flow[isig, iloc] = np.mean(analy_mse_flow[iobsbeg - 1: iobsend, isig, iloc])
        prior_err_flow1[isig, iloc] = np.mean(prior_mse_flow1[iobsbeg - 1: iobsend, isig, iloc])
        analy_err_flow1[isig, iloc] = np.mean(analy_mse_flow1[iobsbeg - 1: iobsend, isig, iloc])
        prior_err_flow2[isig, iloc] = np.mean(prior_mse_flow2[iobsbeg - 1: iobsend, isig, iloc])
        analy_err_flow2[isig, iloc] = np.mean(analy_mse_flow2[iobsbeg - 1: iobsend, isig, iloc])

t1 = time()
print('time used: {:.2f} hours'.format((t1-t0)/3600))

# -------------------------- Uncertainty ---------------------------
'''
u=ψ_y, v=-ψ_x, so u~ψ/L, u~ψ/L, L is the length scale over which ψ varies
Pull an effective length scale from data and then use β to push it down
'''
psi_k = np.fft.fft2(psi_truth_full[:train_size], axes=(1,2)) # shape (Nt,Nx,Nx,2)
u = np.real(np.fft.ifft2(psi_k * 1j * KY[..., None], axes=(1,2)))
v = np.real(np.fft.ifft2(psi_k * (-1j) * KX[..., None], axes=(1,2)))
# effective length scale per grid, per layer
l_scales = np.sqrt(np.mean(psi_truth_full[:train_size]**2, axis=0)) / np.sqrt(np.mean(u**2 + v**2, axis=0))
# layer 1 ψ std: σ_ψ1 = σ_v1 * L1
sigma_psi_1 = sigma_b_vel * l_scales[..., 0] 
# combine regression coefficients of two layers to get a magnitude
beta_mag = 0.5 * (np.abs(beta_u) + np.abs(beta_v))  
# layer 2 ψ std: σ_ψ2 = β * σ_ψ1 * (L2/L1) (consistent with linear projection)
sigma_psi_2 = beta_mag * sigma_psi_1 * (l_scales[..., 1] / l_scales[..., 0])  
sigma_psi = np.stack([sigma_psi_1, sigma_psi_2], axis=2)

np.savez('../data/OI_QG.npz',
         psi_prior=psi_prior,
         psi_analy=psi_analy,
         sigma_psi=sigma_psi,
         prior_mse_flow=prior_mse_flow,
         analy_mse_flow=analy_mse_flow,
         prior_mse_flow1=prior_mse_flow1,
         analy_mse_flow1=analy_mse_flow1,
         prior_mse_flow2=prior_mse_flow2,
         analy_mse_flow2=analy_mse_flow2)

prior_err = np.nan_to_num(prior_err_flow, nan=999999)
analy_err = np.nan_to_num(analy_err_flow, nan=999999)

# # uncomment these if for tuning inflation and localization
# minerr = np.min(prior_err)
# inds = np.where(prior_err == minerr)
# print('min prior mse = {0:.6e}, sigma_b_vel = {1:.3f}, localizaiton = {2:.3f}'.format(minerr, sigma_b_vels[inds[0][0]], localization_radiuses[inds[1][0]]))
# minerr = np.min(analy_err)
# inds = np.where(analy_err == minerr)
# print('min analy mse = {0:.6e}, sigma_b_vel = {1:.3f}, localizaiton = {2:.3f}'.format(minerr, sigma_b_vels[inds[0][0]], localization_radiuses[inds[1][0]]))

# uncomment these if for test
print('prior mse (two layers) = {0:.6e}, sigma_b_vel = {1:.3f}, localizaiton = {2:.3f}'.format(np.mean(((psi_truth - psi_prior)[iobsbeg:, :]) ** 2), sigma_b_vels[0], localization_radiuses[0]))
print('analy mse (two layers) = {0:.6e}, sigma_b_vel = {1:.3f}, localizaiton = {2:.3f}'.format(np.mean(((psi_truth - psi_analy)[iobsbeg:, :]) ** 2), sigma_b_vels[0], localization_radiuses[0]))
print('prior mean mse (layer 1) = {0:.6e}, sigma_b_vel = {1:.3e}, localizaiton = {2:.3f}'.format(prior_err_flow1[0,0], sigma_b_vels[0], localization_radiuses[0]))
print('analy mean mse (layer 1) = {0:.6e}, sigma_b_vel = {1:.3e}, localizaiton = {2:.3f}'.format(analy_err_flow1[0,0], sigma_b_vels[0], localization_radiuses[0]))
print('prior mean mse (layer 2) = {0:.6e}, sigma_b_vel = {1:.3e}, localizaiton = {2:.3f}'.format(prior_err_flow2[0,0], sigma_b_vels[0], localization_radiuses[0]))
print('analy mean mse (layer 2) = {0:.6e}, sigma_b_vel = {1:.3e}, localizaiton = {2:.3f}'.format(analy_err_flow2[0,0], sigma_b_vels[0], localization_radiuses[0]))