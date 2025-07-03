import numpy as np

class variable_binner:
    # def __init__(self):
    def fit(self, X, n_bins=3, method='uniform_quantile'):
        """
        Fit bin edges to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be binned. For a single variable
            1d array use reshape(-1,1).

        n_bins : int, default=3
            Number of bins to use for discretization.

        method : {'uniform_quantile', 'regular'}, default='uniform_quantile'
            Binning strategy:
            - 'uniform_quantile': bins are based on quantiles of the data.
            - 'regular': bins are evenly spaced between min and max.
        """
        self.n_bins = n_bins
        self.method = method
        X = np.array(X)

        self.train_count = X.shape[0]
                
        if method == 'uniform_quantile':            
            self.quantiles = np.linspace(0,1,n_bins+1) 
            self.bin_edges = np.quantile(X, self.quantiles[1:-1], axis=0)  
            
        elif method == 'regular':
            xmax = X.max(axis=0)
            xmin = X.min(axis=0)
            pre_bin_edges = np.linspace(xmin, xmax, n_bins+1)
            self.bin_edges = pre_bin_edges[1:-1]
    def transform(self, X):
        """
        Transform the input data into one-hot bin representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform using previously computed bin edges.
            For a single variable 1d array use reshape(-1,1).

        Returns
        -------
        x_bins : ndarray of shape (n_samples, n_features, n_bins)
            One-hot encoded representation of which bin each value falls into.
        """
        X = np.array(X)
        # X = X.reshape((X.shape[0],X.shape[1],1))
        if len(self.bin_edges.shape) == 1:
            x_bins = np.zeros((X.shape[0],X.shape[1],self.n_bins), dtype=int)
            x_bins[:,:,0] = (X <= self.bin_edges[0])
            x_bins[:,:,-1] = (X > self.bin_edges[-1])
            for i in range(1,self.n_bins-1):
                x_bins[:,:,i] = (X <= self.bin_edges[i])&(X > self.bin_edges[i-1])
        
        elif len(self.bin_edges.shape) == 2:
            x_bins = np.zeros((X.shape[0],X.shape[1],self.n_bins), dtype=int)
            x_bins[:,:,0] = (X <= self.bin_edges[0])
            x_bins[:,:,-1] = (X > self.bin_edges[-1])
            for i in range(1,self.n_bins-1):
                x_bins[:,:,i] = (X <= self.bin_edges[i])&(X > self.bin_edges[i-1])
        
        return x_bins
            
def log_(x):
    """
    Safe logarithm function that avoids log(0).

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    logx : ndarray
        Natural logarithm of `x`, with log(1) used for zero entries.
    """
    x = np.array(x)
    x[x==0] = 1
    logx = np.log(x)
    return logx

def xlogx(x):
    """
    Compute x * log(x) with safe handling of zero and negative values.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    xlogx : ndarray
        Element-wise computation of x * log(x), where 0 * log(0) is defined as 0.
        NaN is returned for negative values.
    """
    xlogx = np.zeros(x.shape)
    mask = x>0
    xlogx[mask] = x[mask]*np.log(x[mask])
    xlogx[x==0] = 0.0
    xlogx[x<0] = np.nan
    return xlogx
            
def MI_binned(binned_x, binned_y):
    """
    Compute the mutual information between two binned variables.

    Parameters
    ----------
    binned_x : ndarray of shape (n_samples, n_xbins)
        One-hot encoded bins for variable X.

    binned_y : ndarray of shape (n_samples, n_ybins)
        One-hot encoded bins for variable Y.

    Returns
    -------
    mi : float
        Estimated mutual information I(X; Y).
    """
    assert (binned_x.shape[0]==binned_y.shape[0]),\
        'x and y should have the same number of samples'
    n_xbins = binned_x.shape[2]
    n_ybins = binned_y.shape[2]
    n_samples = binned_x.shape[0]
    
    # xy_bins = np.zeros((n_samples,n_xbins,n_ybins),dtype=int)
    mi = 0
    p_1 = 1 / n_samples
    x_p = binned_x * p_1
    y_p = binned_y * p_1
    for i in range(n_xbins):
        xbins = x_p[:,0,i]
        px = xbins.sum() 
        for j in range(n_ybins):
            ybins = y_p[:,0,j]
            py = ybins.sum() 
            pxy = (xbins*ybins).sum() / p_1
            
            mi += xlogx(pxy) - pxy*log_(px*py)
    return mi
    

def MI_binned_matrix(X_binned):
    if parallel == False:
        m_variables = X_binned.shape[1]
        MI_matrix = np.zeros((m_variables,m_variables))
        for i in range(m_variables):
            for j in range(i):
                MI_matrix[i,j] = MI_matrix[j,i] = MI_binned(
                    X_binned[:,i], X_binned[:,j])
        return MI_matrix



def H_binned(binned_x):
    '''
    Compute entropy of a single binned variable.

    Parameters
    ----------
    binned_x : np.ndarray
        Binned representation of a single variable.
        Shape: (n_samples, 1, n_bins)

    Returns
    -------
    entropy : float
        Entropy H(X)
    '''
    n_samples, n_vars, n_bins = binned_x.shape
    p_1 = 1 / n_samples
    x_p = binned_x * p_1

    entropy = 0.0
    for i in range(n_bins):
        px = x_p[:, 0, i].sum()
        entropy -= xlogx(px)
    return entropy


def CMI_binned(binned_x, binned_y, binned_z):
    """
    Compute the conditional mutual information I(X; Y | Z) using binned variables.

    Parameters
    ----------
    binned_x : ndarray of shape (n_samples, 1, n_xbins)
        One-hot encoded bins for variable X.

    binned_y : ndarray of shape (n_samples, 1, n_ybins)
        One-hot encoded bins for variable Y.

    binned_z : ndarray of shape (n_samples, 1, n_zbins)
        One-hot encoded bins for conditioning variable Z.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information I(X; Y | Z).
    """
    assert binned_x.shape[0] == binned_y.shape[0] == binned_z.shape[0], \
        "All inputs must have the same number of samples"

    n_samples = binned_x.shape[0]
    n_xbins = binned_x.shape[2]
    n_ybins = binned_y.shape[2]
    n_zbins = binned_z.shape[2]

    p_1 = 1 / n_samples
    x_p = binned_x * p_1
    y_p = binned_y * p_1
    z_p = binned_z * p_1

    cmi = 0.0
    for k in range(n_zbins):
        z_mask = binned_z[:, 0, k].astype(bool)
        pz = z_p[:, 0, k].sum()
        if pz == 0:
            continue
        for i in range(n_xbins):
            for j in range(n_ybins):
                p_xyz = (binned_x[:, 0, i] * binned_y[:, 0, j] * binned_z[:, 0, k]).sum() * p_1
                p_xz = (binned_x[:, 0, i] * binned_z[:, 0, k]).sum() * p_1
                p_yz = (binned_y[:, 0, j] * binned_z[:, 0, k]).sum() * p_1

                if p_xyz > 0 and p_xz > 0 and p_yz > 0 and pz > 0:
                    cmi += p_xyz * (np.log(pz * p_xyz / (p_xz * p_yz)))

    return cmi















        