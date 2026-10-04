import numpy as np
import numpy.typing as npt

def wiener_process_noise(x: npt.NDArray) -> npt.NDArray:
    """
    Based off an ndarray of time samples x, compute the wiener process based on
    the gaps of the last dimension of x.
    """
    z0 = np.zeros(shape=(*x.shape[:-1], 1))
    diffs = x[..., 1:] - x[..., :x.shape[-1] - 1]
    z = np.random.normal(size=(*x.shape[:-1], x.shape[-1] - 1))
    z = np.concatenate((z0, diffs * z), axis=-1)
    z = np.cumsum(z, axis=-1)
    return z
