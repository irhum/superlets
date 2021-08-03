from functools import partial

import jax
import jax.numpy as jnp


def get_bc(cycles, freq, k_sd=5):
  return cycles/(k_sd * freq)

def cxmorelet(freq, cycles, sampling_freq):
    t = jnp.linspace(-1, 1, sampling_freq*2)

    bc = get_bc(cycles, freq)
    norm = 1/(bc * jnp.sqrt(2*jnp.pi))
    gauss = jnp.exp(-t**2/(2*bc**2))
    sine = jnp.exp(1j*2*jnp.pi*freq*t)

    wavelet = norm * gauss * sine
    return wavelet / jnp.sum(jnp.abs(wavelet))

@partial(jax.jit, static_argnums=3)
@partial(jax.vmap, in_axes=(None, 0, None, None))
def wavelet_transform(signal, freq, cycles, sampling_freq):    
    wavelet = cxmorelet(freq, cycles, sampling_freq)
    return jax.scipy.signal.convolve(signal, wavelet, mode="same")
