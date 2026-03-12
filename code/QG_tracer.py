import numpy as np
from QG import QG
from Lagrangian_tracer import Lagrangian_tracer_model
from time import time
from numba import jit, prange
from tqdm import tqdm
'''
require rocket-fft for numba to be aware of np.fft
!pip install rocket-fft 
'''


###############################################
################## QG model ###################
###############################################

@jit(nopython=True)
def rhs_spectral_topo(q_hat, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian):
    q_vec = q_hat - subtract_hk
    # psi_hat = (q2psi @ q_vec[:,:,:,None])[:,:,:,0]
    # if using numba, manually do the matrix multiplication (2x2 @ 2x1)
    psi_hat = np.zeros((K,K,2), dtype=np.complex128)
    psi_hat[:, :, 0] = q2psi[:, :, 0, 0] * q_vec[:, :, 0] + q2psi[:, :, 0, 1] * q_vec[:, :, 1]
    psi_hat[:, :, 1] = q2psi[:, :, 1, 0] * q_vec[:, :, 0] + q2psi[:, :, 1, 1] * q_vec[:, :, 1]

    # Calculate Ekman plus beta plus mean shear
    RHS = np.zeros_like(q_hat, dtype=np.complex64)
    RHS[:,:,0] = -Ut * dX * q_hat[:,:,0] - (kb**2 + Ut * kd**2) * dX * psi_hat[:,:,0]
    RHS[:,:,1] = Ut * dX * q_hat[:,:,1] - (kb**2 - Ut * kd**2) * dX * psi_hat[:,:,1] - r * Laplacian * psi_hat[:,:,1]

    # For using a 3/2-rule dealiased jacobian
    Psi_hat = np.zeros((int(1.5 * K), int(1.5 * K), 2), dtype=np.complex128)
    Q_hat = np.zeros((int(1.5 * K), int(1.5 * K), 2), dtype=np.complex128)
    Psi_hat[:K//2+1, :K//2+1, :] = (9/4) * psi_hat[:K//2+1, :K//2+1, :]
    Psi_hat[:K//2+1, K+1:int(1.5*K), :] = (9/4) * psi_hat[:K//2+1, K//2+1:K, :]
    Psi_hat[K+1:int(1.5*K), :K//2+1, :] = (9/4) * psi_hat[K//2+1:K, :K//2+1, :]
    Psi_hat[K+1:int(1.5*K), K+1:int(1.5*K), :] = (9/4) * psi_hat[K//2+1:K, K//2+1:K, :]
    Q_hat[:K//2+1, :K//2+1, :] = (9/4) * q_hat[:K//2+1, :K//2+1, :]
    Q_hat[:K//2+1, K+1:int(1.5*K), :] = (9/4) * q_hat[:K//2+1, K//2+1:K, :]
    Q_hat[K+1:int(1.5*K), :K//2+1, :] = (9/4) * q_hat[K//2+1:K, :K//2+1, :]
    Q_hat[K+1:int(1.5*K), K+1:int(1.5*K), :] = (9/4) * q_hat[K//2+1:K, K//2+1:K, :]

    # Calculate u.gradq on 3/2 grid
    u = np.real(np.fft.ifft2(-DY[:,:,None] * Psi_hat, axes=(0,1)))
    v = np.real(np.fft.ifft2(DX[:,:,None] * Psi_hat, axes=(0,1)))
    qx = np.real(np.fft.ifft2(DX[:,:,None] * Q_hat, axes=(0,1)))
    qy = np.real(np.fft.ifft2(DY[:,:,None] * Q_hat, axes=(0,1)))
    jaco_real = u * qx + v * qy

    # FFT, 3/2 grid; factor of (4/9) scales fft
    Jaco_hat = (4/9) * np.fft.fft2(jaco_real, axes=(0,1))

    # Reduce to normal grid
    jaco_hat = np.zeros((K, K, 2), dtype=np.complex128)
    jaco_hat[:K//2 + 1, :K//2 + 1, :] = Jaco_hat[:K//2 + 1, :K//2 + 1, :]
    jaco_hat[:K//2 + 1, K//2 + 1:, :] = Jaco_hat[:K//2 + 1, K+1:int(1.5 * K), :]
    jaco_hat[K//2 + 1:, :K//2 + 1, :] = Jaco_hat[K+1:int(1.5 * K), :K//2 + 1, :]
    jaco_hat[K//2 + 1:, K//2 + 1:, :] = Jaco_hat[K+1:int(1.5 * K), K+1:int(1.5 * K), :]

    # Put it all together
    RHS -= jaco_hat

    return RHS
    
@jit(nopython=True, parallel=True, fastmath=True)
def forward_loop(ens, qp_ens, Nt, dt, HV, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian):
    qp_ens_history = np.zeros((ens, Nt+1, K, K, 2))
    qp_ens_history[:, 0, :, :, :] = qp_ens
    q_ens = np.fft.fft2(qp_ens, axes=(1,2))
    # Timestepping
    for e in prange(ens):  # Parallelize over ensemble members
        q = q_ens[e, :, :, :]
        # Timestepping (Fourth-order Runge=Kutta pseudo-spectral scheme)
        for n in range(1, Nt+1):
            M = 1 / (1 - .25 * dt * HV)
            # First stage ARK4
            k0 = rhs_spectral_topo(q, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l0 = HV * q
            # Second stage
            q1 = M * (q + .5 * dt * k0 + .25 * dt * l0)
            k1 = rhs_spectral_topo(q1, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l1 = HV * q1
            # Third stage
            q2 = M * (q + dt * (13861 * k0 / 62500 + 6889 * k1 / 62500 + 8611 * l0 / 62500 - 1743 * l1 / 31250))
            k2 = rhs_spectral_topo(q2, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l2 = HV * q2
            # Fourth stage
            q3 = M * (q + dt * (-0.04884659515311858 * k0 - 0.1777206523264010 * k1 + 0.8465672474795196 * k2 +
                                0.1446368660269822 * l0 - 0.2239319076133447 * l1 + 0.4492950415863626 * l2))
            k3 = rhs_spectral_topo(q3, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l3 = HV * q3
            # Fifth stage
            q4 = M * (q + dt * (-0.1554168584249155 * k0 - 0.3567050098221991 * k1 + 1.058725879868443 * k2 +
                                0.3033959883786719 * k3 + 0.09825878328356477 * l0 - 0.5915442428196704 * l1 +
                                0.8101210538282996 * l2 + 0.2831644057078060 * l3))
            k4 = rhs_spectral_topo(q4, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l4 = HV * q4
            # Sixth stage
            q5 = M * (q + dt * (0.2014243506726763 * k0 + 0.008742057842904184 * k1 + 0.1599399570716811 * k2 +
                                0.4038290605220775 * k3 + 0.2260645738906608 * k4 + 0.1579162951616714 * l0 +
                                0.1867589405240008 * l2 + 0.6805652953093346 * l3 - 0.2752405309950067 * l4))
            k5 = rhs_spectral_topo(q5, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l5 = HV * q5

            # Successful step, proceed to evaluation
            qp = np.real(np.fft.ifft2(q + dt * (0.1579162951616714 * (k0 + l0) +
                                                0.1867589405240008 * (k2 + l2) +
                                                0.6805652953093346 * (k3 + l3) -
                                                0.2752405309950067 * (k4 + l4) +
                                                (k5 + l5) / 4), axes=(0,1)))
            q = np.fft.fft2(qp, axes=(0,1))

            qp_ens_history[e, n, :, :, :] = qp

    return qp_ens_history

class QG:
    def __init__(self, K=128, kd=10, kb=np.sqrt(22), U=1, r=9, nu=1e-12, H=40, topo=None):
        # Set up hyperviscous PV dissipation
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)
        HV = np.tile((-nu * (KX**2 + KY**2)**4)[:,:, None], (1,1,2))

        # Initialize topography
        if topo is None:
            dx = 2 * np.pi / K
            X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
            topo = H * (np.cos(X) + 2 * np.cos(2 * Y))
            topo -= np.mean(topo)  # subtracting the mean to center the topography
        hk = np.fft.fft2(topo)

        # Initialize additional variables for the simulation
        Ut = U  # zonal shear flow
        dX = 1j * KX
        dY = 1j * KY
        Laplacian = dX**2 + dY**2
        K_square = KX**2 + KY**2
        q2psi = -1 / (K_square * (K_square + kd**2))[:,:,None,None] * (np.eye(2) * K_square[:, :, None, None] + kd**2/2)
        q2psi[0,0,:,:] = 0
        q2psi = q2psi.astype(np.complex128)
        subtract_hk = np.zeros(((K,K,2)), dtype=complex)
        subtract_hk[:,:,1] = hk

        # for 3/2-rule dealiased Jacobian
        K_Jac = int(K * 3/2)
        Kx_Jac = np.fft.fftfreq(K_Jac) * K_Jac
        Ky_Jac = np.fft.fftfreq(K_Jac) * K_Jac
        KX_Jac, KY_Jac = np.meshgrid(Kx_Jac, Ky_Jac)
        DX = 1j * KX_Jac
        DY = 1j * KY_Jac

        self.K = K
        self.kd = kd
        self.kb = kb
        self.r = r
        self.hk = hk
        self.dX = dX
        self.dY = dY
        self.DX = DX
        self.DY = DY
        self.q2psi = q2psi
        self.HV = HV
        self.Ut = Ut
        self.Laplacian = Laplacian
        self.subtract_hk = subtract_hk

    def forward(self, Nt=10000, dt=1e-3, qp=None):
        K = self.K
        HV = self.HV
        kd = self.kd
        r = self.r
        subtract_hk = self.subtract_hk
        DX  = self.DX
        DY = self.DY
        q2psi = self.q2psi
        Ut = self.Ut
        dX = self.dX
        dY = self.dY

        # Initialize potential vorticity
        if qp is None:
            qp = np.zeros((K, K, 2))
            qp[:, :, 1] = 10 * np.random.randn(K, K)
            qp[:, :, 1] -= np.mean(qp[:, :, 1])  # Centering the PV
            qp[:, :, 0] = qp[:, :, 1]
        q = np.fft.fft2(qp, axes=(0,1))

        qp_history = np.zeros((Nt+1, K, K, 2))
        qp_history[0, :, :, :] = qp
        # Timestepping (Fourth-order Runge=Kutta pseudo-spectral scheme)
        for ii in tqdm(range(1, Nt+1), desc='Timestepping'):
            M = 1 / (1 - .25 * dt * HV)
            # First stage ARK4
            k0 = rhs_spectral_topo(q, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l0 = HV * q
            # Second stage
            q1 = M * (q + .5 * dt * k0 + .25 * dt * l0)
            k1 = rhs_spectral_topo(q1, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l1 = HV * q1
            # Third stage
            q2 = M * (q + dt * (13861 * k0 / 62500 + 6889 * k1 / 62500 + 8611 * l0 / 62500 - 1743 * l1 / 31250))
            k2 = rhs_spectral_topo(q2, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l2 = HV * q2
            # Fourth stage
            q3 = M * (q + dt * (-0.04884659515311858 * k0 - 0.1777206523264010 * k1 + 0.8465672474795196 * k2 +
                                0.1446368660269822 * l0 - 0.2239319076133447 * l1 + 0.4492950415863626 * l2))
            k3 = rhs_spectral_topo(q3, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l3 = HV * q3
            # Fifth stage
            q4 = M * (q + dt * (-0.1554168584249155 * k0 - 0.3567050098221991 * k1 + 1.058725879868443 * k2 +
                                0.3033959883786719 * k3 + 0.09825878328356477 * l0 - 0.5915442428196704 * l1 +
                                0.8101210538282996 * l2 + 0.2831644057078060 * l3))
            k4 = rhs_spectral_topo(q4, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l4 = HV * q4
            # Sixth stage
            q5 = M * (q + dt * (0.2014243506726763 * k0 + 0.008742057842904184 * k1 + 0.1599399570716811 * k2 +
                                0.4038290605220775 * k3 + 0.2260645738906608 * k4 + 0.1579162951616714 * l0 +
                                0.1867589405240008 * l2 + 0.6805652953093346 * l3 - 0.2752405309950067 * l4))
            k5 = rhs_spectral_topo(q5, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l5 = HV * q5

            # Successful step, proceed to evaluation
            qp = np.real(np.fft.ifft2(q + dt * (0.1579162951616714 * (k0 + l0) +
                                                0.1867589405240008 * (k2 + l2) +
                                                0.6805652953093346 * (k3 + l3) -
                                                0.2752405309950067 * (k4 + l4) +
                                                (k5 + l5) / 4), axes=(0,1)))
            q = np.fft.fft2(qp, axes=(0,1))

            qp_history[ii, :, :, :] = qp

        return qp_history
            
    def forward_ens(self, ens, Nt=1, dt=1e-3, qp_ens=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        '''
        K = self.K
        HV = self.HV
        kd = self.kd
        kb = self.kb
        r = self.r
        subtract_hk = self.subtract_hk
        DX  = self.DX
        DY = self.DY
        q2psi = self.q2psi
        Ut = self.Ut
        dX = self.dX
        dY = self.dY
        Laplacian = self.Laplacian

        return forward_loop(ens, qp_ens, Nt, dt, HV, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)


###############################################
################ tracer model #################
###############################################

def truncate(kk, r, style='circle'):
    '''
    1. require the input kk has shape (K,) or (K,K,...)
    2. r is the radius to be preserved.
    3. style = 'circle' or 'square', default is 'circle'
    4. return flattened modes with 'F' order with shape (k_left,...)
    '''
    K = kk.shape[0]

    if kk.ndim == 1:
        index_to_remove = np.zeros(K, dtype=bool)
        index_to_remove[r+1:-r] = True
        new_shape = np.array(kk.shape)
        new_shape[:2] = min((2*r+1), K)
        kk_cut = kk[~index_to_remove].reshape(new_shape)
        
    elif kk.ndim > 1:
        if style == 'square':
            index_to_remove = np.zeros((K, K), dtype=bool)
            index_to_remove[r+1:-r, :] = True
            index_to_remove[:, r+1:-r] = True
            k_left = min((2*r+1), K)**2
            
        elif style == 'circle':
            index_to_remove = np.ones((K, K), dtype=bool)
            k_left = 0
            for ix in range(K):
                for iy in range(K):
                    r2_xy = min((ix**2 + iy**2), ((ix-K)**2 + iy**2), ((iy-K)**2 + ix**2), ((ix-K)**2 + (iy-K)**2))
                    if r2_xy <= r**2:
                        index_to_remove[ix,iy] = False  
                        k_left += 1

        else:
            raise Exception("unknown style key input")
                                    
        # To retrieve elements in Fortran ('F') order:
        axes = np.arange(kk.ndim)
        axes[0], axes[1] = axes[1], axes[0]
        kk_T = np.transpose(kk, axes)
        kk_cut = kk_T[~index_to_remove.T]

        # Returned flattened truncatation modes with order 'F' 
        new_shape = np.array(kk.shape[1:])
        new_shape[0] = k_left
        kk_cut = kk_cut.reshape(new_shape, order='F')

    return kk_cut

def inv_truncate(kk_truncated, r, K, style='circle'):
    ''' Recovers the original array from a truncated version by filling zeros.
        Parameters:
        - kk_truncated: The truncated array of shape (k_left,...). *if the original array has dim>1, kk_truncated should have dim>1
        - r: radius to be preserved.
        - style: 'circle' or 'square'.
    '''
    if kk_truncated.ndim == 1:
        recovered = np.zeros(K, dtype=kk_truncated.dtype)
        recovered[:r+1] = kk_truncated[:r+1]
        recovered[-r:] = kk_truncated[-r:]

    elif kk_truncated.ndim > 1:
        k_left = kk_truncated.shape[0]
        new_shape = [K] + list(kk_truncated.shape)
        new_shape[1] = K
        recovered = np.zeros(new_shape, dtype=kk_truncated.dtype)
        if style == 'square':
            k_cut = int(np.sqrt(k_left))
            shape_cut = [k_cut, k_cut] + list(kk_truncated.shape[1:])
            kk_temp = np.reshape(kk_truncated, shape_cut, order='F')
            recovered[:r+1, :r+1] = kk_temp[:r+1, :r+1]
            recovered[-r:, :r+1] = kk_temp[-r:, :r+1]
            recovered[:r+1, -r:] = kk_temp[:r+1, -r:]
            recovered[-r:, -r:] = kk_temp[-r:, -r:]
            
        elif style == 'circle':
            kx = np.fft.fftfreq(K) * K
            ky = np.fft.fftfreq(K) * K
            KX, KY = np.meshgrid(kx, ky)
            k_index_map = {(KX[iy, ix], KY[iy, ix]): (ix, iy) for ix in range(K) for iy in range(K) if (KX[iy, ix]**2 + KY[iy, ix]**2) <=r**2}
            for ik_, (k, ik) in enumerate(k_index_map.items()):
                ikx, iky = ik
                recovered[iky, ikx] = kk_truncated[ik_]

        else:
            raise Exception("unknown style key input")
            
    return recovered

@jit(nopython=True, parallel=True, fastmath=True)
def for_loop_tracer(K, ens, L, N, dt, x, y, KX_flat, KY_flat, psi_hat_flat, sigma_xy):
    KX_flat_c = np.ascontiguousarray(KX_flat[None, :])
    KY_flat_c = np.ascontiguousarray(KY_flat[None, :])
    for e in prange(ens):  # Parallelize over ensemble members
        for i in range(1, N+1):
            exp_term = np.exp(1j * x[e, :, i-1][:, None] @ KX_flat_c + 1j * y[e, :, i-1][:, None] @ KY_flat_c)
            uk = (psi_hat_flat[e, i-1, :] * (1j) * KY_flat)
            vk = (psi_hat_flat[e, i-1, :] * (-1j) * KX_flat)

            # # ensure conjugate symmetric
            # if style == 'square' and psi_hat.shape[0] == K:
            #     uk[K//2::K] = 0; uk[K*K//2:K*(K//2 + 1)] = 0
            #     vk[K//2::K] = 0; vk[K*K//2:K*(K//2 + 1)] = 0 
            
            u = np.real(exp_term @ uk) / K**2
            v = np.real(exp_term @ vk) / K**2
            
            x[e, :, i] = x[e, :, i-1] + u * dt + np.random.randn(L) * sigma_xy * np.sqrt(dt)
            y[e, :, i] = y[e, :, i-1] + v * dt + np.random.randn(L) * sigma_xy * np.sqrt(dt)
            x[e, :, i] = np.mod(x[e, :, i], 2*np.pi)  # Periodic boundary conditions
            y[e, :, i] = np.mod(y[e, :, i], 2*np.pi)  # Periodic boundary conditions

    return x, y

class Lagrangian_tracer_model:
    """
    math of the model:
    v(x, t) =\sum_k{psi_hat_{1,k}(t) e^(ik·x) r_k}
    r_k=(ik_2,-ik_1). 

    """
    def __init__(self, K, sigma_xy, style='square'):
        """
        Parameters:
        - K: number of modes
        - sigma_xy: float, standard deviation of the noise
        - style: truncation style
        """
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)
        KX_flat = KX.flatten()
        KY_flat = KY.flatten()
        KX_flat = KX_flat.astype(np.complex128)
        KY_flat = KY_flat.astype(np.complex128)

        self.K = K
        self.sigma_xy = sigma_xy
        self.style = style
        self.KX = KX
        self.KY = KY
        self.KX_flat = KX_flat
        self.KY_flat = KY_flat

    def forward(self, L, N, dt, x0, y0, psi_hat, interv=1, t_interv=None):
        """
        Integrates tracer locations using forward Euler method.
        
        Parameters:
        - N: int, total number of steps
        - L: int, number of tracers
        - psi_hat: np.array of shape (K, K, N) or truncated, Fourier time series of the upper layer stream function.
        - dt: float, time step
        - x0:  Initial tracer locations in x of shape (L,)
        - y0:  Initial tracer locations in y of shape (L,)
        - interv:  int, wave number inverval for calculating u, v field 
        - t_interv: int, time interval for calculate and save u, v field
        """
        K = self.K
        KX = self.KX
        KY = self.KY
        style = self.style
        x = np.zeros((L, N+1))
        y = np.zeros((L, N+1))
        x[:,0] = x0
        y[:,0] = y0
        ut = np.zeros((K//interv, K//interv, N//t_interv))  
        vt = np.zeros((K//interv, K//interv, N//t_interv))

        if psi_hat.shape[0] < K and style == 'square':
            r_cut = (psi_hat.shape[0] - 1) // 2
            KX_flat = truncate(KX, r_cut, style=style)
            KY_flat = truncate(KY, r_cut, style=style)
            psi_hat_flat = np.reshape(psi_hat, (psi_hat.shape[0]**2, -1), order='F')
        elif psi_hat.shape[0] == K:
            KX_flat = KX.flatten(order='F')
            KY_flat = KY.flatten(order='F')
            psi_hat_flat = np.reshape(psi_hat, (K**2, -1), order='F')
        else:
            raise Exception("unknown truncation style")

        l = 0
        for i in range(1, N+1):
            exp_term = np.exp(1j * x[:, i-1][:,None] @ KX_flat[None,:] + 1j * y[:, i-1][:,None] @ KY_flat[None,:])
            uk = (psi_hat_flat[:, i-1] * (1j) * KY_flat)
            vk = (psi_hat_flat[:, i-1] * (-1j) * KX_flat)

            # ensure conjugate symmetric
            if style == 'square' and psi_hat.shape[0] == K:
                uk[K//2::K] = 0; uk[K*K//2:K*(K//2 + 1)] = 0
                vk[K//2::K] = 0; vk[K*K//2:K*(K//2 + 1)] = 0 

            u = np.squeeze(exp_term @ uk[:,None]) / K**2
            v = np.squeeze(exp_term @ vk[:,None]) / K**2
            u = np.real(u)
            v = np.real(v)

            # max_imag_abs = max(np.max(np.abs(np.imag(u))), np.max(np.abs(np.imag(v))))
            # if max_imag_abs > 1e-10:
            #     raise Exception("get significant imaginary parts, check the ifft2")
            # else:
            #     u = np.real(u)
            #     v = np.real(v)
            
            x[:, i] = x[:, i-1] + u * dt + np.random.randn(L) * self.sigma_xy * np.sqrt(dt)
            y[:, i] = y[:, i-1] + v * dt + np.random.randn(L) * self.sigma_xy * np.sqrt(dt)
            x[:, i] = np.mod(x[:, i], 2*np.pi)  # Periodic boundary conditions
            y[:, i] = np.mod(y[:, i], 2*np.pi)  # Periodic boundary conditions

            if np.mod(i,t_interv) == 0:
                if psi_hat.shape[0] == K:
                    psi_hat_KK = psi_hat[:, :, i-1]
                else:
                    psi_hat_KK = inv_truncate(psi_hat_flat[:, i-1][:,None], r_cut, K, style)[:,:,0]

                # using built-in ifft2
                u_ifft = np.fft.ifft2(psi_hat_KK * 1j * KY)
                ut[:,:,l] = u_ifft[::interv, ::interv] # only save the sparsely sampled grids
                v_ifft = np.fft.ifft2(psi_hat_KK * (-1j) * KX)
                vt[:,:,l] = v_ifft[::interv, ::interv] # only save the sparsely sampled grids
                        
                l += 1

        return x, y, ut, vt

    def forward_ens(self, ens, L, N, dt, x0, y0, psi_hat, eps_nufft=1e-9):
        '''
        Ensemble forecast.

        Parameters:
        - N: int, total number of time steps
        - L: int, number of tracers
        - psi_hat: np.array of shape (ens, N+1, K, K), Fourier time series of the upper layer stream function.
        - dt: float, time step
        - x0:  Initial tracer locations in x of shape (ens, L)
        - y0:  Initial tracer locations in y of shape (ens, L)
        '''
        K = self.K
        KX_flat = self.KX_flat
        KY_flat = self.KY_flat
        sigma_xy = self.sigma_xy
        x = np.zeros((ens, L, N+1))
        y = np.zeros((ens, L, N+1))
        x[:,:,0] = x0
        y[:,:,0] = y0

        psi_hat_flat = np.reshape(psi_hat, (ens, N+1, -1))
        x, y = for_loop_tracer(K, ens, L, N, dt, x, y, KX_flat, KY_flat, psi_hat_flat, sigma_xy)

        return x, y

###############################################
########### QG flow + tracer model ############
###############################################

def generate_topo(N=128, alpha=4.0):
    '''generate a gaussian random field as topography'''
    kx = np.fft.fftfreq(N).reshape(-1, 1)
    ky = np.fft.fftfreq(N).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-6  # avoid division by zero
    spectrum = 1.0 / k**alpha # Power-law spectrum: 1 / k^alpha
    noise = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))
    fft_field = noise * np.sqrt(spectrum)
    field = np.fft.ifft2(fft_field).real
    field -= np.mean(field)
    field /= np.std(field)
    return field

class QG_tracer:
    def __init__(self, K=128, kd=10, kb=np.sqrt(22), U=1, r=9, nu=1e-12, H=40, topo=None, sigma_xy=0.1, style='square'):
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)

        # Initialize topography
        if topo is None:
            dx = 2 * np.pi / K
            X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
            topo = H * (np.cos(X) + 2 * np.cos(2 * Y))
            topo -= np.mean(topo)  # subtracting the mean to center the topography
        hk = np.fft.fft2(topo)

        # Initialize additional variables for the simulation
        K_square = KX**2 + KY**2
        q2psi = -1 / (K_square * (K_square + kd**2))[:,:,None,None] * (np.eye(2) * K_square[:, :, None, None] + kd**2/2)
        q2psi[0,0,:,:] = 0
        q2psi = q2psi.astype(np.complex128)
        subtract_hk = np.zeros(((K,K,2)), dtype=complex)
        subtract_hk[:,:,1] = hk

        self.flow_model = QG(K, kd, kb, U, r, nu, H, topo)
        self.tracer_model = Lagrangian_tracer_model(K, sigma_xy, style)
        self.q2psi = q2psi
        self.subtract_hk = subtract_hk

    def forward_ens(self, ens, Nt=1, dt=1e-3, qp_ens=None, L=1, x0=None, y0=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        x0, y0: arrays of shape (ens, L)
        '''
        K = qp_ens.shape[1]
        flow_model = self.flow_model
        tracer_model = self.tracer_model
        q2psi = self.q2psi
        subtract_hk = self.subtract_hk

        # run flow model
        qp_history = flow_model.forward_ens(ens, Nt, dt, qp_ens)
        q_hat_history = np.fft.fft2(qp_history, axes=(2,3))

        # q to psi
        q_vec = q_hat_history - subtract_hk
        psi_hat_history = (q2psi @ q_vec[:,:,:,:,:,None])[:,:,:,:,:,0] # of shape (ens,Nt+1,K,K,2)

        # # psi to velocity
        # u = np.real(np.fft.ifft2(psi_hat_history[:, -1, :, :, :] * 1j * KY[:, :, None], axes=(1,2)))
        # v = np.real(np.fft.ifft2(psi_hat_history[:, -1, :, :, :] * (-1j) * KX[:, :, None], axes=(1,2)))

        # run tracer model
        x, y = tracer_model.forward_ens(ens, L, Nt, dt, x0, y0, psi_hat_history[:, :, :, :, 0]) # of shape (ens,L,Nt+1)

        return psi_hat_history, x, y, qp_history#, u, v

    def forward_flow(self, ens=1, Nt=1, dt=1e-3, qp_ens=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        '''
        flow_model = self.flow_model
        q2psi = self.q2psi
        subtract_hk = self.subtract_hk

        # run flow model
        qp_history = flow_model.forward_ens(ens, Nt, dt, qp_ens) # of shape (ens, Nt+1, K, K, 2)
        q_hat_history = np.fft.fft2(qp_history, axes=(2,3))

        # q to psi
        q_vec = q_hat_history - subtract_hk
        psi_hat_history = (q2psi @ q_vec[:,:,:,:,:,None])[:,:,:,:,:,0] # of shape (ens,Nt+1,K,K,2)

        return qp_history, psi_hat_history


if __name__ == "__main__": 
    np.random.seed(0)

    # ---------- QG model parameters ------------
    K = 128 # Number of points (also Fourier modes) in each direction
    kd = 10 # Nondimensional deformation wavenumber
    kb = np.sqrt(22) # Nondimensional beta wavenumber, beta = kb^2 
    U = 1 # Zonal shear flow
    r = 9 # Nondimensional Ekman friction coefficient
    nu = 1e-12 # Coefficient of biharmonic vorticity diffusion
    H = 40 # Topography parameter
    dt = 2e-3 # Time step size
    warm_up = 1000 # Warm-up time steps
    Nt = 2e6 + warm_up # Number of time steps

    # ------- Tracer observation parameters -------
    L = 1024 # Number of tracers
    sigma_xy = 0.1 # Tracer observation noise std (in sde)
    dt_obs = 4e-2 # Observation time interval
    obs_freq = int(dt_obs / dt) # Observation frequency
    Nt_obs = int((Nt - warm_up) / obs_freq + 1) # Number of observations saved

    # -------------- initialization ---------------
    # topo = generate_topo(K, 4) # generate topography
    # topo = H * topo
    dx = 2 * np.pi / K
    X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
    topo = H * (np.cos(X) + 2 * np.cos(2 * Y))
    topo -= np.mean(topo)  # subtracting the mean to center the topography
    qp = np.zeros((K, K, 2))
    qp[:, :, 1] = 10 * np.random.randn(K, K)
    qp[:, :, 1] -= np.mean(qp[:, :, 1])
    qp[:, :, 0] = qp[:, :, 1]
    psi_k_t = np.zeros((Nt_obs, K, K, 2), dtype=complex)
    model = QG_tracer(K=K, kd=kd, kb=kb, U=U, r=r, nu=nu, topo=topo, sigma_xy=sigma_xy)
    x_t = np.zeros((Nt_obs, L))
    y_t = np.zeros((Nt_obs, L))
    x0 = np.pi + 0.1 * np.random.randn(L) # np.random.uniform(0, 2*np.pi, L)
    y0 = np.pi + 0.1 * np.random.randn(L) # np.random.uniform(0, 2*np.pi, L)
    x_t[0, :] = x0
    y_t[0, :] = y0

    # warm up
    qp_t, psi_k_t1 = model.forward_flow(ens=1, Nt=warm_up, dt=dt, qp_ens=qp[None,:,:,:])
    qp_t0 = qp_t[:, -1, :, :, :]
    psi_k_t[0, :, :, :] = psi_k_t1[0, -1, :, :, :]

    t0 = time()
    # ------------ model integration --------------
    for i in range(1, Nt_obs):
        psi_k_t1, x1, y1, qp_t1 = model.forward_ens(ens=1, Nt=obs_freq, dt=dt, qp_ens=qp_t0, L=L, x0=x0[None, :], y0=y0[None, :])
        psi_k_t[i, :, :, :] = psi_k_t1[0, -1, :, :, :]
        x_t[i, :] = x1[0, :, -1]
        y_t[i, :] = y1[0, :, -1]
        x0 = x1[0, :, -1]
        y0 = y1[0, :, -1]
        qp_t0 = qp_t1[:, -1, :, :, :]

        if np.isnan(psi_k_t1).any():
            print('Error: NaN detected at steps {0:d}.'.format(i))
            break

    psi_truth = np.fft.ifft2(psi_k_t, axes=(1,2))
    # check imaginary part
    max_imag_abs = np.max(np.abs(np.imag(psi_truth)))
    if max_imag_abs > 1e-8:
        raise Exception("get significant imaginary parts, check ifft2")
    else:
        psi_truth = np.real(psi_truth)

    psi_truth = psi_truth[:, ::2, ::2] # subsample to 64*64
    xy_truth = np.concatenate((x_t[:,:,None], y_t[:,:,None]), axis=2)
    sigma_obs = 0.01
    xy_obs = xy_truth + sigma_obs * np.random.randn(xy_truth.shape[0], xy_truth.shape[1], xy_truth.shape[2])
    xy_obs = np.mod(xy_obs, 2*np.pi)  # Periodic boundary conditions
    sigma_psi = 0.01
    psi_noisy = psi_truth + sigma_psi * np.random.randn(*psi_truth.shape)

    t1 = time()
    print('time used: {:.2f} hours'.format((t1-t0)/3600))

    save = {
    'K': K,
    'kd': kd,
    'kb': kb,
    'U': U,
    'r': r,
    'H': H,
    'nu': nu,
    'topo': topo,
    'dt': dt,
    'L': L,
    'dt_obs': dt_obs,
    'xy_obs': xy_obs,
    'psi_noisy': psi_noisy,
    'sigma_obs': sigma_obs,
    'sigma_psi': sigma_psi,
    'psi_truth': psi_truth,
    'xy_truth': xy_truth,
    'sigma_xy': sigma_xy,
    }
    np.savez('../data/qg_data.npz', **save)

