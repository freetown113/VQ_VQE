import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from models import Encoder, Decoder, Quantizer, VQ_VAE


def create_fwd_fn(model, *args, **kwargs):
    def fwd_pass(input):
        return model(*args, **kwargs)(input)
    return hk.without_apply_rng(hk.transform(fwd_pass))


def make_loss_fn(input, var, kwargs):
    args = {kwargs[i].name: kwargs[i].value for i in dir(kwargs)}
    enc = Encoder
    dec = Decoder
    quant = Quantizer
    model = create_fwd_fn(VQ_VAE, enc, dec, quant, var, **args)
    return model
    
