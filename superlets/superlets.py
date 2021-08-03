from functools import partial

import jax
import jax.numpy as jnp

from .morlet import wavelet_transform


@jax.jit
@partial(jax.vmap, in_axes=(None, None, 0))
def superlet_transform_helper(signal, freqs, order):
  return wavelet_transform(signal, freqs, order)

def order_to_cycles(base_cycle, max_order):
    # return jnp.arange(1, max_order) * base_cycle
    return jnp.arange(0, max_order) + base_cycle

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
def superlet_transform(signal, freqs, base_cycle: int, min_order: int, max_order: int, eps=1e-12):
  cycles = order_to_cycles(base_cycle, max_order)
  orders = get_order(freqs, min(freqs), max(freqs), min_order, max_order)

  out = superlet_transform_helper(signal, freqs, cycles)

  mask = get_mask(orders, max_order)

  out = jax.ops.index_update(out, mask.T, 1)

  return norm_geomean(out, orders, eps)
#   out = jnp.log(out + eps).sum(axis=0)

#   return jnp.exp(out / jnp.array(orders).reshape(-1, 1))
