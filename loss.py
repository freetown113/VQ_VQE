import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from models import Encoder, Decoder, Quantizer, VQ_VAE


def create_fwd_fn(model, **kwargs):
    def fwd_pass(input):
        return model(**kwargs)(input)
    return hk.without_apply_rng(hk.transform(fwd_pass))


def make_loss(input):
    enc = Encoder()
    dec = Decoder()
    quant = Quantizer()
    model = VQ_VAE(enc, dec, quant, )
