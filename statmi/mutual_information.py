import numpy as np
from .binning import MI_binned, MI_binned_matrix, variable_binner, H_binned, CMI_binned
from .kde import h_kd, MI_kd, CMI_kd

def MI(x,y,method='uniform_binned',bins=None):
    """
    Compute mutual information I(X; Y) between two variables using 
    binning or kernel density estimation (KDE).

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        First input variable (X), a 1D array.

    y : ndarray of shape (n_samples,)
        Second input variable (Y), a 1D array.

    method : {'uniform_binned', 'regular_binned', 'kde'}, default='uniform_binned'
        Method for estimating mutual information:
        - 'uniform_binned': uses quantile-based equal-frequency binning
        - 'regular_binned': uses equal-width binning
        - 'kde': uses kernel density estimation (continuous MI)

    bins : int or None, optional
        Number of bins for discretization (used for binned methods).
        If None, the number of bins is determined internally.
        For 'kde', this controls the grid resolution (default ~100).

    Returns
    -------
    mi : float
        Estimated mutual information I(X; Y).
    """
    if method == 'uniform_binned':
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        if bins is None:
            xbinner = variable_binner()
            ybinner = variable_binner()
            xbinner.fit(x, method='uniform_quantile')
            ybinner.fit(y, method='uniform_quantile')
            binned_x = xbinner.transform(x)
            binned_y = ybinner.transform(y)
            mi = MI_binned(binned_x, binned_y)
            return mi
        else:
            xbinner = variable_binner()
            ybinner = variable_binner()
            xbinner.fit(x, method='uniform_quantile', n_bins = bins)
            ybinner.fit(x, method='uniform_quantile', n_bins = bins)
            binned_x = xbinner.transform(x)
            binned_y = ybinner.transform(y)
            mi = MI_binned(binned_x, binned_y)
            return mi
    elif method == 'regular_binned':
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        if bins is None:
            xbinner = variable_binner()
            ybinner = variable_binner()
            xbinner.fit(x, method='regular')
            ybinner.fit(y, method='regular')
            binned_x = xbinner.transform(x)
            binned_y = ybinner.transform(y)
            mi = MI_binned(binned_x, binned_y)
            return mi
        else:
            xbinner = variable_binner()
            ybinner = variable_binner()
            xbinner.fit(x, method='regular', n_bins = bins)
            ybinner.fit(x, method='regular', n_bins = bins)
            binned_x = xbinner.transform(x)
            binned_y = ybinner.transform(y)
            mi = MI_binned(binned_x, binned_y)
            return mi
    elif method == 'kde':
        if bins is None:
            mi = MI_kd(x, y)
            return mi
        else:
            mi = MI_kd(x, y, bins=bins)
            return mi
    else:
        raise ValueError(f"Unsupported method: {method}")
            
        
def CMI(x, y, z, method='uniform_binned', bins=None):
    """
    Compute the conditional mutual information I(X; Y | Z)
    using either binning or KDE-based estimation.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Variable X.

    y : ndarray of shape (n_samples,)
        Variable Y.

    z : ndarray of shape (n_samples,)
        Conditioning variable Z.

    method : {'uniform_binned', 'regular_binned', 'kde'}, default='uniform_binned'
        Estimation method.

    bins : int or None, optional
        Number of bins (for binned methods) or grid resolution (for KDE).

    Returns
    -------
    cmi : float
        Estimated conditional mutual information I(X; Y | Z).
    """
    if method == 'uniform_binned':
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
        xbinner = variable_binner()
        ybinner = variable_binner()
        zbinner = variable_binner()
        if bins is None:
            xbinner.fit(x, method='uniform_quantile')
            ybinner.fit(y, method='uniform_quantile')
            zbinner.fit(z, method='uniform_quantile')
        else:
            xbinner.fit(x, method='uniform_quantile', n_bins=bins)
            ybinner.fit(y, method='uniform_quantile', n_bins=bins)
            zbinner.fit(z, method='uniform_quantile', n_bins=bins)
        binned_x = xbinner.transform(x)
        binned_y = ybinner.transform(y)
        binned_z = zbinner.transform(z)
        return CMI_binned(binned_x, binned_y, binned_z)

    elif method == 'regular_binned':
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
        xbinner = variable_binner()
        ybinner = variable_binner()
        zbinner = variable_binner()
        if bins is None:
            xbinner.fit(x, method='regular')
            ybinner.fit(y, method='regular')
            zbinner.fit(z, method='regular')
        else:
            xbinner.fit(x, method='regular', n_bins=bins)
            ybinner.fit(y, method='regular', n_bins=bins)
            zbinner.fit(z, method='regular', n_bins=bins)
        binned_x = xbinner.transform(x)
        binned_y = ybinner.transform(y)
        binned_z = zbinner.transform(z)
        return CMI_binned(binned_x, binned_y, binned_z)

    elif method == 'kde':
        return CMI_kd(x, y, z, bins=bins or 50)  # default bins=50 if None

    else:
        raise ValueError(f"Unsupported method: {method}")
    

def MI_matrix(X, method='uniform_binned', bins=None):
    """
    Compute the mutual information matrix for all variable pairs in X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_variables)
        Input data matrix where each column is a variable.

    method : {'uniform_binned', 'regular_binned', 'kde'}, default='uniform_binned'
        Method for estimating mutual information:
        - 'uniform_binned': quantile-based equal-frequency binning
        - 'regular_binned': equal-width binning
        - 'kde': kernel density estimation (continuous)

    bins : int or None, optional
        Number of bins (for binned methods) or grid resolution (for KDE).
        If None, a default is used.

    Returns
    -------
    MI_mat : ndarray of shape (n_variables, n_variables)
        Symmetric matrix of mutual information values.
    """
    n_samples, n_vars = X.shape
    MI_mat = np.zeros((n_vars, n_vars))

    if method in ['uniform_binned', 'regular_binned']:
        bin_method = 'uniform_quantile' if method == 'uniform_binned' else 'regular'
        binner = variable_binner()
        if bins is None:
            binner.fit(X, method=bin_method)
        else:
            binner.fit(X, method=bin_method, n_bins=bins)
        X_binned = binner.transform(X)

        for i in range(n_vars):
            for j in range(i):
                mi = MI_binned(X_binned[:, i:i+1], X_binned[:, j:j+1])
                MI_mat[i, j] = MI_mat[j, i] = mi

    elif method == 'kde':
        for i in range(n_vars):
            for j in range(i):
                mi = MI_kd(X[:, i], X[:, j], bins=bins or 100)
                MI_mat[i, j] = MI_mat[j, i] = mi

    else:
        raise ValueError(f"Unsupported method: {method}")

    return MI_mat

def H(x, method='regular_binned', bins=None):
    """
    Compute the entropy H(X) of a single variable using either
    regular binning or kernel density estimation (KDE).

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input variable (1D array).

    method : {'regular_binned', 'kde'}, default='regular_binned'
        Method for estimating entropy:
        - 'regular_binned': equal-width binning
        - 'kde': kernel density estimation (continuous entropy)

    bins : int or None, optional
        Number of bins (for 'regular_binned') or grid resolution (for 'kde').
        If None, a default is used.

    Returns
    -------
    entropy : float
        Estimated entropy H(X).
    """
    if method == 'regular_binned':
        x = x.reshape(-1,1)
        binner = variable_binner()
        if bins is None:
            binner.fit(x, method='regular')
        else:
            binner.fit(x, method='regular', n_bins=bins)
        binned_x = binner.transform(x)
        return H_binned(binned_x)

    elif method == 'kde':
        return h_kd(x, bins=bins or 100)

    else:
        raise ValueError(f"Unsupported method: {method}")