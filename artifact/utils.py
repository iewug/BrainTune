import numpy as np
from typing import Optional

def _mad(x:np.ndarray, axis:Optional[int]=None):
    """
    Median Absolute Deviation (MAD)
    
    MAD = median(|X - median(X)|)

    Input a 1D or 2D ndarray, calculate MAD along the specified axis.
    """
    # Ensure array is either 1D or 2D
    if x.ndim > 2:
        raise ValueError(f"Only 1D and 2D arrays are supported (input has {x.ndim} dimensions)")

    return np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def _mat_quantile(arr, q, axis=None):
    """Calculate the numeric value at quantile (`q`) for a given distribution.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing samples from the distribution to summarize. Must
        be either a 1-D or 2-D array.
    q : float
        The quantile to calculate for the input data. Must be between 0 and 1,
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis along which quantile values should be calculated. Defaults to
        calculating the value at the given quantile for the entire array.

    Returns
    -------
    quantile : scalar or np.ndarray
        If no axis is specified, returns the value at quantile (q) for the full
        input array as a single numeric value. Otherwise, returns an
        ``np.ndarray`` containing the values at quantile (q) for each row along
        the specified axis.

    Notes
    -----
    MATLAB calculates quantiles using different logic than Numpy: Numpy treats
    the provided values as a whole population, whereas MATLAB treats them as a
    sample from a population of unknown size and adjusts quantiles accordingly.
    This function mimics MATLAB's logic to produce identical results.

    """
    # Sort the array in ascending order along the given axis (any NaNs go to the end)
    # Return NaN if array is empty.
    if len(arr) == 0:
        return np.nan
    arr_sorted = np.sort(arr, axis=axis)

    # Ensure array is either 1D or 2D
    if arr_sorted.ndim > 2:
        e = "Only 1D and 2D arrays are supported (input has {0} dimensions)"
        raise ValueError(e.format(arr_sorted.ndim))

    # Reshape data into a 2D array with the shape (num_axes, data_per_axis)
    if axis is None:
        arr_sorted = arr_sorted.reshape(-1, 1)
    else:
        arr_sorted = np.moveaxis(arr_sorted, axis, 0)

    # Initialize quantile array with values for non-usable (n < 2) axes.
    # Sets quantile to only non-NaN value if n == 1, or NaN if n == 0
    quantiles = arr_sorted[0, :]

    # Get counts of non-NaN values for each axis and determine which have n > 1
    n = np.sum(np.isfinite(arr_sorted), axis=0)
    n_usable = n[n > 1]

    if np.any(n > 1):
        # Calculate MATLAB-style sample-adjusted quantile values
        q = np.asarray(q, dtype=np.float64)
        q_adj = ((q - 0.5) * n_usable / (n_usable - 1)) + 0.5

        # Get the exact (float) index position of the quantile for each usable axis, as
        # well as the indices of the values below and above it (if not a whole number)
        exact_idx = (n_usable - 1) * np.clip(q_adj, 0, 1)
        pre_idx = np.floor(exact_idx).astype(np.int32)
        post_idx = np.ceil(exact_idx).astype(np.int32)

        # Interpolate exact quantile values for each usable axis
        axis_idx = np.arange(len(n))[n > 1]
        pre = arr_sorted[pre_idx, axis_idx]
        post = arr_sorted[post_idx, axis_idx]
        quantiles[n > 1] = pre + (post - pre) * (exact_idx - pre_idx)

    return quantiles[0] if quantiles.size == 1 else quantiles


def _mat_iqr(arr, axis=None):
    """Calculate the inter-quartile range (IQR) for a given distribution.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing samples from the distribution to summarize.
    axis : {int, tuple of int, None}, optional
        Axis along which IQRs should be calculated. Defaults to calculating the
        IQR for the entire array.

    Returns
    -------
    iqr : scalar or np.ndarray
        If no axis is specified, returns the IQR for the full input array as a
        single numeric value. Otherwise, returns an ``np.ndarray`` containing
        the IQRs for each row along the specified axis.

    Notes
    -----
    See notes for :func:`utils._mat_quantile`.

    """
    return _mat_quantile(arr, 0.75, axis) - _mat_quantile(arr, 0.25, axis)



if __name__ == '__main__':
    IQR_TO_SD = 0.7413
    MAD_TO_SD = 1.4826
    x = np.array([[0],[1]])
    print(_mat_iqr(x, axis=1)*IQR_TO_SD)
    print(_mad(x,axis=1)*MAD_TO_SD)
    # print(np.quantile(x,0.75,axis=1)-np.quantile(x,0.25,axis=1))
    # print(_mat_iqr(x,axis=1))
    print(_mat_iqr(x, axis=1)*IQR_TO_SD)
    print(_mad(x,axis=1)*MAD_TO_SD)