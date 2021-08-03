from functools import partial

import jax
import jax.numpy as jnp


def get_bc(cycles, freq, k_sd=5):
  return cycles/(k_sd * freq)

@partial(jax.jit, static_argnums=2)
def cxmorelet(f, c, fs):
  t = jnp.linspace(-1, 1, fs*2)

  bc = get_bc(c, f)
  norm = 1/(bc * jnp.sqrt(2*jnp.pi))
  gauss = jnp.exp(-t**2/(2*bc**2))
  sine = jnp.exp(1j*2*jnp.pi*f*t)

  wavelet = norm * gauss * sine
  return wavelet / jnp.sum(jnp.abs(wavelet))

@jax.jit
@partial(jax.vmap, in_axes=(None, 0, None))
def wavelet_transform(signal, f, c):
  wavelet = cxmorelet(f, c, 1024)
  return jax.scipy.signal.convolve(signal, wavelet, mode="same")
