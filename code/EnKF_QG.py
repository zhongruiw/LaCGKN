import numpy as np
from os.path import dirname, join as pjoin
from QG_tracer import QG_tracer
from time import time
from numba import jit

def construct_GC_2d_general(cut, mlocs, ylocs, Nx=None):
    """
    Construct the Gaspari and Cohn localization matrix for a 2D field.

    Parameters:
        cut (float): Localization cutoff distance.
        mlocs (array of shape (nstates,2)): 2D coordinates of the model states [(x1, y1), (x2, y2), ...].
        ylocs (array of shape (nobs,2)): 2D coordinates of the observations [[x1, y1], [x2, y2], ...].
        Nx (int, optional): Number of grid points in each direction.

    Returns:
        np.ndarray: Localization matrix of shape (len(ylocs), len(mlocs)).
    """
    ylocs = ylocs[:, np.newaxis, :]  # Shape (nobs, 1, 2)
    mlocs = mlocs[np.newaxis, :, :]  # Shape (1, nstates, 2)

    # Compute distances
    dist = np.linalg.norm((mlocs - ylocs + Nx // 2) % Nx - Nx // 2, axis=2)

    # Normalize distances
    r = dist / (0.5 * cut)

    # Compute localization function
    V = np.zeros_like(dist)

    mask2 = (0.5 * cut <= dist) & (dist < cut)
    mask3 = (dist < 0.5 * cut)

    V[mask2] = (
        r[mask2]**5 / 12.0 - r[mask2]**4 / 2.0 + r[mask2]**3 * 5.0 / 8.0
        + r[mask2]**2 * 5.0 / 3.0 - 5.0 * r[mask2] + 4.0 - 2.0 / (3.0 * r[mask2])
    )
    
    V[mask3] = (
        -r[mask3]**5 * 0.25 + r[mask3]**4 / 2.0 + r[mask3]**3 * 5.0 / 8.0 
        - r[mask3]**2 * 5.0 / 3.0 + 1.0
    )

    return V

@jit(nopython=True)
def periodic_mean(x, size):
    '''
    compute the periodic mean for a single x of shape (size,) on domain [0, 2pi)
    '''
    # if np.max(x) - np.min(x) <= np.pi:
    if np.ptp(x) <= np.pi:
        x_mean = np.mean(x)
    else:
        # calculate the periodic mean of two samples
        x_mean = np.mod(np.arctan2((np.sin(x[0]) + np.sin(x[1])) / 2, (np.cos(x[0]) + np.cos(x[1])) / 2), 2*np.pi)

        # successively calculate the periodic mean for the rest samples
        for k in range(3, size+1):
            inc = np.mod(x[k-1] - x_mean + np.pi, 2*np.pi) - np.pi # increment with periodicity considered
            x_mean = x_mean + inc / k
        x_mean = np.mod(x_mean, 2*np.pi)

    return x_mean

@jit(nopython=True, parallel=True, fastmath=True)
def periodic_means(xs, size, L):
    '''
    compute the periodic means of multiple x of shape (size, L)
    '''
    means = np.zeros(L)
    for l in range(L):
        means[l] = periodic_mean(xs[:, l], size)

    return means
        

def eakf(ensemble_size, nobs, xens, Hk, obs_error_var, localize, CMat, obs):
    """
    Ensemble Adjustment Kalman Filter (EAKF) for Lagrangian data assimilation.

    Parameters:
        ensemble_size (int): Number of ensemble members.
        nobs (int): Number of observations.
        xens (np.ndarray): Ensemble matrix of shape (ensemble_size, nmod).
        Hk (np.ndarray): Observation operator matrix of shape (nobs, nmod).
        obs_error_var (float): Observation error variance.
        localize (int): Flag for localization (1 for applying localization, 0 otherwise).
        CMat (np.ndarray): Localization matrix of shape (nobs, nmod).
        obs (np.ndarray): Observations of shape (nobs,).
    
    Returns:
        np.ndarray: Updated ensemble matrix.
    """
    rn = 1.0 / (ensemble_size - 1)
    xmean = np.zeros(xens.shape[1])
    xprime = np.zeros((ensemble_size, xens.shape[1]))
    
    for iobs in range(nobs):
        xmean[:nobs] = periodic_means(xens[:, :nobs], ensemble_size, nobs) # tracer mean
        xmean[nobs:] = np.mean(xens[:, nobs:], axis=0) # flow mean
        xprime = xens - xmean
        xprime[:, :nobs] = np.mod(xprime[:, :nobs] + np.pi, 2*np.pi) - np.pi # tracer perturbation
        hxens = xens[:, iobs] # particularly for the tracer-flow case
        hxmean = xmean[iobs] # particularly for the tracer-flow case
        hxprime = xprime[:, iobs] # particularly for the tracer-flow case
        hpbht = hxprime @ hxprime.T * rn
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (hxprime @ xprime) * rn        
        obs_inc = np.mod(obs[iobs] - hxmean + np.pi, 2*np.pi) - np.pi

        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = Cvect * (pbht / (hpbht + obs_error_var))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        prime_inc = - (gainfact * kfgain[:, None] @ hxprime[None, :]).T
        mean_inc = kfgain * obs_inc
        xens = xens + mean_inc + prime_inc
        xens[:, :nobs] = np.mod(xens[:, :nobs], 2*np.pi) # periodic condition for tracer
    
    return xens

np.random.seed(0)

# --------------------- load data --------------------------
train_size = 80000
val_size = 10000
test_size = 10000
data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_data.npz')
data = np.load(datafname)
psi_truth_full = data['psi_truth']
xy_truth_full = data['xy_truth']
sigma_xy = data['sigma_xy'].item()
xy_obs_full = data['xy_obs']
sigma_obs = data['sigma_obs'].item()
dt_obs = data['dt_obs'].item()

# split training and test data set
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
mlocs = np.array([(ix, iy) for iy in range(Nx) for ix in range(Nx)])
mlocs = np.repeat(mlocs, 2, axis=0)
model = QG_tracer(K=Nx, kd=kd, kb=kb, U=U, r=r, nu=nu, H=H, sigma_xy=sigma_xy)

# ------------------- observation parameters ------------------
L = 64 # number of tracers
xy_truth = xy_truth[:, :L, :]
xy_obs = xy_obs[:, :L, :]
obs_error_var = sigma_obs**2
obs_freq_timestep = int(dt_obs / dt)
ylocs = xy_obs[0, :, :] / (2 * np.pi) * Nx
ylocs = np.repeat(ylocs, 2, axis=0)
nobs = ylocs.shape[0]
nobstime = xy_obs.shape[0]

# contatenate tracer and flow variables
mlocs = np.concatenate((ylocs, mlocs), axis=0)
nmod = mlocs.shape[0]

# --------------------- DA parameters -----------------------
# analysis period
iobsbeg = 50
iobsend = -1

# eakf parameters
ensemble_size = 40
inflation_values = [1.025] # provide multiple values if for tuning
localization_values = [16] # provide multiple values if for tuning
ninf = len(inflation_values)
nloc = len(localization_values)
localize = 1

# ---------------------- initialization -----------------------
Kx = np.fft.fftfreq(Nx) * Nx
Ky = np.fft.fftfreq(Nx) * Nx
KX, KY = np.meshgrid(Kx, Ky)
K_square = KX**2 + KY**2
psi2q_mat = (-np.eye(2) * K_square[:, :, None, None]) + (kd**2/2 * np.ones((2,2)) - np.eye(2) * kd**2)
psi2q_mat = psi2q_mat.astype(np.complex128)
plus_hk = np.zeros(((Nx,Nx,2,1)), dtype=complex)
plus_hk[:,:,1,0] = hk
Hk = np.zeros((nobs, nmod)) # observation forward operator H
Hk[:, :nobs] = np.eye(nobs, nobs)
ics_psi = psi_truth_full[:train_size, :, :, :]
n_ics = ics_psi.shape[0]

# initial flow field
psi0_ens = np.random.randn(ensemble_size, Nx, Nx, 2) * 0.1

# initial tracer displacements
xy0_ens = np.tile(xy_obs[0, :, :][None, :, :], (ensemble_size,1,1)) + sigma_obs * np.random.randn(ensemble_size, L, 2) # shape (Nens, L, 2)

# initial augmented variable
psi0_ens_flat = np.reshape(psi0_ens, (ensemble_size, -1)) # shape (Nens, Nx*Nx*2)
xy0_ens_flat = np.reshape(xy0_ens, (ensemble_size, -1)) # shape (Nens, L*2)
z0_ens = np.concatenate((xy0_ens_flat, psi0_ens_flat), axis=1) # shape (Nens, L*2+Nx*Nx*2)
zobs_total = np.reshape(xy_obs, (nobstime, -1))
ztruth = np.concatenate((np.reshape(xy_truth, (nobstime, -1)), np.reshape(psi_truth, (nobstime, -1))), axis=1)

prior_mse_flow = np.zeros((nobstime,ninf,nloc))
analy_mse_flow = np.zeros((nobstime,ninf,nloc))
prior_mse_flow1 = np.zeros((nobstime,ninf,nloc))
analy_mse_flow1 = np.zeros((nobstime,ninf,nloc))
prior_mse_flow2 = np.zeros((nobstime,ninf,nloc))
analy_mse_flow2 = np.zeros((nobstime,ninf,nloc))
prior_err_flow = np.zeros((ninf,nloc))
analy_err_flow = np.zeros((ninf,nloc))
prior_err_flow1 = np.zeros((ninf,nloc))
analy_err_flow1 = np.zeros((ninf,nloc))
prior_err_flow2 = np.zeros((ninf,nloc))
analy_err_flow2 = np.zeros((ninf,nloc))
prior_mse_tracer = np.zeros((nobstime,ninf,nloc))
analy_mse_tracer = np.zeros((nobstime,ninf,nloc))
prior_err_tracer = np.zeros((ninf,nloc))
analy_err_tracer = np.zeros((ninf,nloc))

t0 = time()
# ---------------------- assimilation -----------------------
for iinf in range(ninf):
    inflation_value = inflation_values[iinf]
    print('inflation:',inflation_value)
    
    for iloc in range(nloc):
        localization_value = localization_values[iloc]
        print('localization:',localization_value)

        zens = z0_ens
        zeakf_prior = np.zeros((nobstime, nmod))
        zeakf_analy = np.empty((nobstime, nmod))
        prior_spread = np.empty((nobstime, nmod))
        analy_spread = np.empty((nobstime, nmod))

        # t0 = time()
        for iassim in range(0, nobstime):
            print(iassim)

            # EnKF step
            obsstep = iassim * obs_freq_timestep + 1
            zeakf_prior[iassim, :nobs] = periodic_means(zens[:, :nobs], ensemble_size, nobs)
            zeakf_prior[iassim, nobs:] = np.mean(zens[:, nobs:], axis=0)  # prior ensemble mean
            
            zobs = zobs_total[iassim, :]

            # inflation RTPP
            ensmean = zeakf_prior[iassim, :]
            ensp = zens - ensmean
            ensp[:, :nobs] = np.mod(ensp[:, :nobs] + np.pi, 2*np.pi) - np.pi
            zens = ensmean + ensp * inflation_value
            zens[:, :nobs] = np.mod(zens[:, :nobs], 2*np.pi)

            prior_spread[iassim, :] = np.std(zens, axis=0, ddof=1)

            # localization matrix        
            CMat = construct_GC_2d_general(localization_value, mlocs, ylocs, Nx)
            np.fill_diagonal(CMat[:nobs, :nobs], 1) # the tracer observation with the same ID has localization value of 1

            # serial update
            zens = eakf(ensemble_size, nobs, zens, Hk, obs_error_var, localize, CMat, zobs)
            
            # save analysis
            zeakf_analy[iassim, :nobs] = periodic_means(zens[:, :nobs], ensemble_size, nobs)
            zeakf_analy[iassim, nobs:] = np.mean(zens[:, nobs:], axis=0)
            analy_spread[iassim, :] = np.std(zens, axis=0, ddof=1)

            # ensemble model integration
            if iassim < nobstime - 1:
                xy0_ens = np.reshape(zens[:, :nobs], (ensemble_size, L, 2))
                x0_ens = xy0_ens[:, :, 0]
                y0_ens = xy0_ens[:, :, 1]
                psi0_ens = np.reshape(zens[:, nobs:], (ensemble_size, Nx, Nx, 2))
                psi0_k_ens = np.fft.fft2(psi0_ens, axes=(1,2)) # shape (Nens,Nx,Nx,2)
                q0_k_ens = (psi2q_mat @ psi0_k_ens[:, :, :, :, None] + plus_hk)[:, :, :, :, 0] # shape (Nens,Nx,Nx,2)
                qp0_ens = np.real(np.fft.ifft2(q0_k_ens, axes=(1,2)))

                psi1_k_ens, x1_ens, y1_ens, _ = model.forward_ens(ens=ensemble_size, Nt=obs_freq_timestep, dt=dt, qp_ens=qp0_ens, L=L, x0=x0_ens, y0=y0_ens)
                
                psi1_k_ens = psi1_k_ens[:, -1, :, :, :]
                x1_ens = x1_ens[:, :, -1]
                y1_ens = y1_ens[:, :, -1]
                xy1_ens = np.concatenate((x1_ens[:, :, None], y1_ens[:, :, None]), axis=2)
                psi1_ens = np.real(np.fft.ifft2(psi1_k_ens, axes=(1,2)))

                zens = np.concatenate((np.reshape(xy1_ens, (ensemble_size, -1)), np.reshape(psi1_ens, (ensemble_size, -1))), axis=1)

                if np.isnan(zens).any():
                    print('Error: NaN detected.')
                    break

                # updata tracer locations
                ylocs = xy_obs[iassim, :, :] / (2 * np.pi) * Nx
                ylocs = np.repeat(ylocs, 2, axis=0)
                mlocs_tracer = np.mean(xy1_ens, axis=0) / (2 * np.pi) * Nx
                mlocs_tracer = np.repeat(mlocs_tracer, 2, axis=0)
                mlocs[:2*L, :] = mlocs_tracer
        
        prior_mse_flow[:, iinf, iloc] = np.mean((ztruth[:, nobs:] - zeakf_prior[:, nobs:]) ** 2, axis=1)
        analy_mse_flow[:, iinf, iloc] = np.mean((ztruth[:, nobs:] - zeakf_analy[:, nobs:]) ** 2, axis=1)
        prior_mse_flow1[:, iinf, iloc] = np.mean((ztruth[:, nobs::2] - zeakf_prior[:, nobs::2]) ** 2, axis=1)
        analy_mse_flow1[:, iinf, iloc] = np.mean((ztruth[:, nobs::2] - zeakf_analy[:, nobs::2]) ** 2, axis=1)
        prior_mse_flow2[:, iinf, iloc] = np.mean((ztruth[:, nobs+1::2] - zeakf_prior[:, nobs+1::2]) ** 2, axis=1)
        analy_mse_flow2[:, iinf, iloc] = np.mean((ztruth[:, nobs+1::2] - zeakf_analy[:, nobs+1::2]) ** 2, axis=1)
        prior_err_flow[iinf, iloc] = np.mean(prior_mse_flow[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_flow[iinf, iloc] = np.mean(analy_mse_flow[iobsbeg - 1: iobsend, iinf, iloc])
        prior_err_flow1[iinf, iloc] = np.mean(prior_mse_flow1[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_flow1[iinf, iloc] = np.mean(analy_mse_flow1[iobsbeg - 1: iobsend, iinf, iloc])
        prior_err_flow2[iinf, iloc] = np.mean(prior_mse_flow2[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_flow2[iinf, iloc] = np.mean(analy_mse_flow2[iobsbeg - 1: iobsend, iinf, iloc])
        prior_mse_tracer[:, iinf, iloc] = np.mean((np.mod(ztruth[:, :nobs] - zeakf_prior[:, :nobs] + np.pi, 2*np.pi) - np.pi) ** 2, axis=1)
        analy_mse_tracer[:, iinf, iloc] = np.mean((np.mod(ztruth[:, :nobs] - zeakf_analy[:, :nobs] + np.pi, 2*np.pi) - np.pi) ** 2, axis=1)
        prior_err_tracer[iinf, iloc] = np.mean(prior_mse_tracer[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_tracer[iinf, iloc] = np.mean(analy_mse_tracer[iobsbeg - 1: iobsend, iinf, iloc])

t1 = time()
print('time used: {:.2f} hours'.format((t1-t0)/3600))

save = {
    'mean_analy': zeakf_analy,
    'mean_prior': zeakf_prior,
    'spread_analy': analy_spread,
    'spread_prior': prior_spread,
    'mse_prior_flow1': prior_mse_flow1,
    'mse_analy_flow1': analy_mse_flow1,
    'mse_prior_flow2': prior_mse_flow2,
    'mse_analy_flow2': analy_mse_flow2,
    'mse_prior_tracer': prior_mse_tracer,
    'mse_analy_tracer': analy_mse_tracer,
}
np.savez('../data/EnKF_QG.npz', **save)

# # uncomment these if for tuning inflation and localization
# prior_err = np.nan_to_num(prior_err_flow, nan=999999)
# analy_err = np.nan_to_num(analy_err_flow, nan=999999)
# minerr = np.min(prior_err)
# inds = np.where(prior_err == minerr)
# print('min prior mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))
# minerr = np.min(analy_err)
# inds = np.where(analy_err == minerr)
# print('min analy mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))

# uncomment these if for test
print('prior mse (two layers) = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((ztruth[:, nobs:] - zeakf_prior[:, nobs:])[iobsbeg:, :]) ** 2), inflation_values[0], localization_values[0]))
print('analy mse (two layers) = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((ztruth[:, nobs:] - zeakf_analy[:, nobs:])[iobsbeg:, :]) ** 2), inflation_values[0], localization_values[0]))
print('prior mean mse (layer 1) = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(prior_err_flow1[0,0], inflation_values[0], localization_values[0]))
print('analy mean mse (layer 1) = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(analy_err_flow1[0,0], inflation_values[0], localization_values[0]))
print('prior mean mse (layer 2) = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(prior_err_flow2[0,0], inflation_values[0], localization_values[0]))
print('analy mean mse (layer 2) = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(analy_err_flow2[0,0], inflation_values[0], localization_values[0]))