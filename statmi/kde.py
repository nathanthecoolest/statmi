import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

def hlog_single(x):
    """
    Safe logarithm function for entropy integrals.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        log(x) if x > 0, otherwise 0.
    """
    if x <= 0:
        return 0.0
    else:
        return np.log(x)
    
hlog = np.vectorize(hlog_single)

def h_kd(x,bins=100):
    """
    Estimate the differential entropy H(X) using kernel density estimation (KDE).

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input variable.

    bins : int, default=100
        Number of evaluation points for numerical integration.

    Returns
    -------
    h : float
        Estimated entropy H(X).
    """
    x = np.asanyarray(x).reshape(-1,1)
    kd = KernelDensity(kernel='gaussian',bandwidth='scott')
    kd.fit(x)
    space = np.linspace(
        x.mean()-1.5*(x.mean()-x.min()), 
        x.mean()+1.5*(x.max()-x.mean()),
        bins).reshape(-1,1)
    dx = space[1,0] - space[0,0]
    P_x = np.exp(kd.score_samples(space)).flatten()
    h = -(P_x * hlog( P_x ) * dx).sum()
    return h




def MI_kd(x,y,bins=100):
    """
    Estimate mutual information I(X; Y) using kernel density estimation.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        First variable.

    y : ndarray of shape (n_samples,)
        Second variable.

    bins : int, default=100
        Number of grid points for integration.

    Returns
    -------
    mi : float
        Estimated mutual information I(X; Y).
    """
    assert len(x.shape)==1 and len(y.shape)==1,\
        'x and y must be 1d arrays'
    assert x.shape[0] == y.shape[0],\
        'x and y must be same length'
    assert np.abs(x-y).sum() != 0,\
        'x and y must not be point by point equal'
    xy = np.array([x,y])
    xmean = x.mean()
    xspace = np.linspace(
        xmean-1.5*(xmean-x.min()), 
        xmean+1.5*(x.max()-xmean),
        bins)
    ymean = y.mean()
    yspace = np.linspace(
        ymean-1.5*(ymean-y.min()), 
        ymean+1.5*(y.max()-ymean),
        bins)
    mgrid = np.meshgrid(xspace,yspace)
    xyspace = np.array([
        mgrid[0].flatten(),
        mgrid[1].flatten()])
    dx = xspace[1] - xspace[0]
    dy = yspace[1] - yspace[0]
    kde_xy = gaussian_kde(xy)
    kde_y = gaussian_kde(y)
    kde_x = gaussian_kde(x)
    P_xy = kde_xy(xyspace)
    P_x = np.tile(kde_x(xspace),yspace.shape[0])
    P_y = np.repeat(kde_y(yspace),xspace.shape[0])
    dydx = dx*dy
    ### I'm here
    I = (P_xy * hlog( P_xy / (P_x*P_y) ) ).sum() * dydx
    return I


def standardize(data):
    """
    Standardize a 1D array: (X - mean) / std.

    Parameters
    ----------
    data : ndarray
        Input array.

    Returns
    -------
    standardized : ndarray
        Standardized array.
    """
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + 1e-12)

def scott_bandwidth(data):
    """
    Compute bandwidth using Scott's rule for KDE.

    Parameters
    ----------
    data : ndarray of shape (d, n_samples)
        Input stacked data, each row a variable.

    Returns
    -------
    bandwidth : float
        Scott's rule bandwidth estimate.
    """
    n, d = data.shape
    return np.power(n, -1.0 / (d + 4)) * np.std(data, axis=1).mean()

def fit_kde(samples, bandwidth):
    """
    Fit a multivariate Gaussian KDE.

    Parameters
    ----------
    samples : ndarray of shape (d, n_samples)
        Input variables stacked by rows.

    bandwidth : float
        Bandwidth to use in the KDE.

    Returns
    -------
    kde : sklearn.neighbors.KernelDensity
        Fitted KDE object.
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(samples.T)
    return kde

def eval_kde(kde, points):
    """
    Evaluate a fitted KDE on a set of grid points.

    Parameters
    ----------
    kde : sklearn.neighbors.KernelDensity
        Fitted KDE model.

    points : ndarray of shape (d, n_points)
        Points at which to evaluate the density.

    Returns
    -------
    densities : ndarray of shape (n_points,)
        Evaluated density values at input points.
    """
    log_density = kde.score_samples(points.T)
    return np.exp(log_density)

def rescaled_space(v, bins):
    """
    Construct a rescaled grid around a 1D variable for integration.

    Parameters
    ----------
    v : ndarray of shape (n_samples,)
        Input variable.

    bins : int
        Number of grid points.

    Returns
    -------
    grid : ndarray of shape (bins,)
        Rescaled grid from slightly below min to above max.
    """
    mean = v.mean()
    lower = mean - 1.5 * (mean - v.min())
    upper = mean + 1.5 * (v.max() - mean)
    return np.linspace(lower, upper, bins)

def CMI_kd(x, y, z, bins=50, bw_alpha=0.2):
    """
    Estimate conditional mutual information I(X; Y | Z) using KDE.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Variable X.

    y : ndarray of shape (n_samples,)
        Variable Y.

    z : ndarray of shape (n_samples,)
        Conditioning variable Z.

    bins : int, default=50
        Grid resolution per dimension for numerical integration.

    bw_alpha : float, default=0.2
        Scaling factor for Scott's bandwidth rule.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information I(X; Y | Z).
    """
    assert x.ndim == y.ndim == z.ndim == 1, 'All inputs must be 1D arrays'
    assert x.shape[0] == y.shape[0] == z.shape[0], 'Inputs must be same length'

    # Standardize inputs
    x = standardize(x)
    y = standardize(y)
    z = standardize(z)

    # Stack data
    xyz = np.vstack([x, y, z])
    xz = np.vstack([x, z])
    yz = np.vstack([y, z])
    z_only = np.vstack([z])

    # Compute common bandwidth using Scott's Rule
    bandwidth = bw_alpha*scott_bandwidth(xyz)

    # Fit KDEs
    kde_xyz = fit_kde(xyz, bandwidth)
    kde_xz = fit_kde(xz, bandwidth)
    kde_yz = fit_kde(yz, bandwidth)
    kde_z = fit_kde(z_only, bandwidth)

    # Create custom rescaled grid
    xspace = rescaled_space(x, bins)
    yspace = rescaled_space(y, bins)
    zspace = rescaled_space(z, bins)
    dx = xspace[1] - xspace[0]
    dy = yspace[1] - yspace[0]
    dz = zspace[1] - zspace[0]

    X, Y, Z = np.meshgrid(xspace, yspace, zspace, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    # Evaluate densities
    P_xyz = eval_kde(kde_xyz, grid_points)
    P_xz = eval_kde(kde_xz, grid_points[[0, 2]])
    P_yz = eval_kde(kde_yz, grid_points[[1, 2]])
    P_z = eval_kde(kde_z, grid_points[[2]])

    # Compute integrand
    ratio = (P_xyz * P_z) / (P_xz * P_yz + 1e-12)
    integrand = P_xyz * hlog(ratio)

    # Integrate
    cmi = integrand.sum() * dx * dy * dz
    return cmi

