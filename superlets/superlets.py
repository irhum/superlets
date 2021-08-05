from functools import partial

import jax
import jax.numpy as jnp

from .morlet import wavelet_transform


@partial(jax.jit, static_argnums=3)
@partial(jax.vmap, in_axes=(None, None, 0, None))
def superlet_transform_helper(signal, freqs, order, sampling_freq):
    return wavelet_transform(signal, freqs, order, sampling_freq) * jnp.sqrt(2)

def order_to_cycles(base_cycle, max_order, mode):
    if mode == "add":
        return jnp.arange(0, max_order) + base_cycle
    elif mode == "mul":
        return jnp.arange(1, max_order+1) * base_cycle
    else: raise ValueError("mode should be one of \"mul\" or \"add\"")

def get_order(f, f_min: int, f_max: int, o_min: int, o_max: int):
    return o_min + round((o_max - o_min) * (f - f_min) / (f_max - f_min))

@partial(jax.vmap, in_axes=(0, None))
def get_mask(order, max_order):
    return jnp.arange(1, max_order+1) > order

@jax.jit
def norm_geomean(X, root_pows, eps):
    X = jnp.log(X + eps).sum(axis=0)

    return jnp.exp(X / jnp.array(root_pows).reshape(-1, 1))

# @jax.jit 
def adaptive_superlet_transform(signal, freqs, sampling_freq: int, base_cycle: int, min_order: int, max_order: int, eps=1e-12, mode="mul"):
    """Computes the adaptive superlet transform of the provided signal

    Args:
        signal (jnp.ndarray): 1D array containing the signal data
        freqs (jnp.ndarray): 1D sorted array containing the frequencies to compute the wavelets at
        sampling_freq (int): Sampling frequency of the signal 

        base_cycle (int): The number of cycles corresponding to order=1
        min_order (int): The minimum upper limit of orders to be used for a frequency in the adaptive superlet.
        max_order (int): The maximum upper limit of orders to be used for a frequency in the adaptive superlet.
        
        eps (float, optional): Epsilon value to be used for numerical stability in the geometric mean. Defaults to 1e-12.
        mode (str, optional): "add" or "mul", corresponding to the use of additive or multiplicative adaptive superlets. Defaults to "mul".

    Returns:
        jnp.ndarray: 2D array (Frequency x Time) representing the computed scalogram
    """
    cycles = order_to_cycles(base_cycle, max_order, mode)
    orders = get_order(freqs, min(freqs), max(freqs), min_order, max_order)
    mask = get_mask(orders, max_order)

    out = superlet_transform_helper(signal, freqs, cycles, sampling_freq)
    out = jax.ops.index_update(out, mask.T, 1)

    return norm_geomean(out, orders, eps)