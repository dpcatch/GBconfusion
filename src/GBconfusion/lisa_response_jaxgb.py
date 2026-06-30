import numpy as np
import jax
from lisaorbits import KeplerianOrbits, EqualArmlengthOrbits
import jax.numpy as jnp
import pandas as pd
from jaxgb.jaxgb import JaxGBaccurate
from jaxgb.params import GBObject
jax.config.update("jax_enable_x64", True)



def jaxgb_response_batch(params,  t_obs, n, myjaxgb, tdi_fn):
    #tdi_fd = myjaxgb.get_tdi(jnp.asarray(params),  tdi_generation=2.0,  tdi_combination="AET")
    tdi_fd = tdi_fn(jnp.array(params))
    A = tdi_fd[0]
    E = tdi_fd[1]
    T = tdi_fd[2]

    f0s = np.asarray(params)[:, 0]
    df = 1 / t_obs
    kmins = myjaxgb.get_kmin(f0s)
    freqs = myjaxgb.get_frequency_grid(kmins)

    return A, E,  freqs