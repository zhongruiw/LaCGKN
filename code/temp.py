import numpy as np
from os.path import dirname, join as pjoin
def unit2xy(xy_unit):
    if xy_unit.shape[-1] == 4:
        cos0 = xy_unit[..., 0]
        sin0 = xy_unit[..., 1]
        cos1 = xy_unit[..., 2]
        sin1 = xy_unit[..., 3]
    else:
        cos0 = xy_unit[..., ::4]
        sin0 = xy_unit[..., 1::4]
        cos1 = xy_unit[..., 2::4]
        sin1 = xy_unit[..., 3::4]

    # Recover the angles
    angle0 = np.arctan2(sin0, cos0)
    angle1 = np.arctan2(sin1, cos1)
    
    # Ensure angles are in [0, 2π)
    x = np.mod(angle0, 2 * np.pi)
    y = np.mod(angle1, 2 * np.pi)
    
    # Reconstruct the original pos
    xy = np.stack([x, y], axis=-1)

    return xy

# --------------------- load data --------------------------
train_size = 80000
val_size = 10000
test_size = 10000
data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_data_long.npz')
data = np.load(datafname)
psi_truth_full = data['psi_truth']
xy_truth_full = data['xy_truth']
sigma_xy = data['sigma_xy'].item()
xy_obs_full = data['xy_obs']
sigma_obs = data['sigma_obs'].item()
dt_obs = data['dt_obs'].item()

Nx = 64 # Number of grid points in each direction
dt = 2e-3 # Time step size
I = 64 # number of tracers

# split training and test data set
data_size = test_size
psi_truth = psi_truth_full[train_size+val_size:train_size+val_size+test_size]
xy_truth = xy_truth_full[train_size+val_size:train_size+val_size+test_size]
xy_obs = xy_obs_full[train_size+val_size:train_size+val_size+test_size]

data = np.load('../data/qg_enkf_long_L64_loc16.npz')
zeakf_analy = data['mean_analy']
psi_enkf = np.reshape(zeakf_analy[:, 2*I:], (test_size, Nx, Nx, 2))

data = np.load('../data/qg_oi_long.npz')
psi_oi = data['psi_analy']

psi_cgkn_da = np.load('../data/CGKN_L64_noretrain_psi_DA.npy')
psi_cgkn_pred_stage2 = np.load('../data/CGKN_L64_noretrain_psi_OneStepPrediction.npy')
xy_unit_cgkn_pred_stage2 = np.load('../data/CGKN_L64_noretrain_xy_unit_OneStepPrediction.npy')
xy_cgkn_pred_stage2 = unit2xy(xy_unit_cgkn_pred_stage2)

psi_cgkn_pred_stage1 = np.load('../data/CGKN_L1024_complex_long_sepf1g1_psi_OneStepPrediction_stage1.npy')
xy_unit_cgkn_pred_stage1 = np.load('../data/CGKN_L1024_complex_long_sepf1g1_xy_unit_OneStepPrediction_stage1.npy')
xy_cgkn_pred_stage1 = unit2xy(xy_unit_cgkn_pred_stage1)

psi_cgkn_dimz32_pred_stage1 = np.load('../data/CGKN_resblock_dimz32_psi_OneStepPrediction_stage1.npy')
xy_unit_cgkn_dimz32_pred_stage1 = np.load('../data/CGKN_resblock_dimz32_xy_unit_OneStepPrediction_stage1.npy')
xy_cgkn_dimz32_pred_stage1 = unit2xy(xy_unit_cgkn_dimz32_pred_stage1)

psi_cnn_pred = np.load('../data/CNN_L1024_long_psi_OneStepPrediction.npy')
xy_unit_cnn_pred = np.load('../data/CNN_L1024_long_xy_unit_OneStepPrediction.npy')
xy_cnn_pred = unit2xy(xy_unit_cnn_pred)

xy_obs = xy_obs[:, :I]
xy_truth = xy_truth[:, :I]
xy_cgkn_pred_stage1 = xy_cgkn_pred_stage1[:, :I]
xy_cgkn_pred_stage2 = xy_cgkn_pred_stage2[:, :I]
xy_cgkn_dimz32_pred_stage1 = xy_cgkn_dimz32_pred_stage1[:, :I]
xy_cnn_pred = xy_cnn_pred[:, :I]


def rmse_tracer(xy_truth, xy_pred, iobsbeg=0):
    """
    xy_truth, xy_pred: (T, I, 2)
    """
    d = ((xy_truth - xy_pred) + np.pi) % (2*np.pi) - np.pi
    rmse = np.sqrt(np.mean(d[iobsbeg:]**2))
    return rmse

def rmse_field(psi_truth, psi_pred, iobsbeg=0, layer=None):
    """
    psi_truth, psi_pred: (T, H, W, 2)
    """
    if layer is None:
        d = psi_truth - psi_pred
    else:
        d = psi_truth[:,:,:,layer] - psi_pred[:,:,:,layer]
    rmse = np.sqrt(np.mean(d[iobsbeg:]**2))
    return rmse

def report_pred(model_name, xy_truth, psi_truth, xy_pred, psi_pred, iobsbeg=0):
    rmse_x = rmse_tracer(xy_truth[1:], xy_pred[:-1], iobsbeg=iobsbeg)
    rmse_u = rmse_field(psi_truth[1:], psi_pred[:-1], iobsbeg=iobsbeg, layer=0)
    rmse_l = rmse_field(psi_truth[1:], psi_pred[:-1], iobsbeg=iobsbeg, layer=1)
    rmse_t = rmse_field(psi_truth[1:], psi_pred[:-1], iobsbeg=iobsbeg, layer=None)
    print(f"{model_name}:")
    print(f"  pred tracer mse : {rmse_x**2:.6e} | rmse: {rmse_x:.6e}")
    print(f"  pred upper mse  : {rmse_u**2:.6e} | rmse: {rmse_u:.6e}")
    print(f"  pred lower mse  : {rmse_l**2:.6e} | rmse: {rmse_l:.6e}")
    print(f"  pred total mse  : {rmse_t**2:.6e} | rmse: {rmse_t:.6e}")

def report_da(model_name, psi_truth, psi_da, iobsbeg=50):
    rmse_u = rmse_field(psi_truth, psi_da, iobsbeg=iobsbeg, layer=0)
    rmse_l = rmse_field(psi_truth, psi_da, iobsbeg=iobsbeg, layer=1)
    rmse_t = rmse_field(psi_truth, psi_da, iobsbeg=iobsbeg, layer=None)
    print(f"{model_name}:")
    print(f"  analy upper mse  : {rmse_u**2:.6e} | rmse: {rmse_u:.6e}")
    print(f"  analy lower mse  : {rmse_l**2:.6e} | rmse: {rmse_l:.6e}")
    print(f"  analy total mse  : {rmse_t**2:.6e} | rmse: {rmse_t:.6e}")

################################ One-step prediction ######################################
iobsbeg = 0

print('------------- One-step prediction -----------')
report_pred("CGKN (stage 1)", xy_truth, psi_truth, xy_cgkn_pred_stage1, psi_cgkn_pred_stage1, iobsbeg=iobsbeg)
report_pred("CGKN (dimz32, stage 1)", xy_truth, psi_truth, xy_cgkn_dimz32_pred_stage1, psi_cgkn_dimz32_pred_stage1, iobsbeg=iobsbeg)
report_pred("DNN+CNN", xy_truth, psi_truth, xy_cnn_pred, psi_cnn_pred, iobsbeg=iobsbeg)
report_pred("Persistence", xy_truth, psi_truth, xy_truth, psi_truth, iobsbeg=iobsbeg)

# (optional) evaluate CGKN stage2:
# report_pred("CGKN (stage 2)", xy_truth, psi_truth, xy_cgkn_pred_stage2, psi_cgkn_pred_stage2, iobsbeg=iobsbeg_pred)

################################ Data assimilation ######################################
iobsbeg = 50

print('------------- Data assimilation -----------')
report_da("OI", psi_truth, psi_oi, iobsbeg=iobsbeg)
report_da("EnKF", psi_truth, psi_enkf, iobsbeg=iobsbeg)
report_da("CGKN", psi_truth, psi_cgkn_da, iobsbeg=iobsbeg)
psi_clim = np.mean(psi_truth, axis=0)[None, ...]
report_da("Clim", psi_truth, psi_clim, iobsbeg=iobsbeg)
