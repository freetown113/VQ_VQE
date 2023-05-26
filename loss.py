import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from models import Encoder, Decoder, Quantizer, VQ_VAE


def create_fwd_fn(model, *args, **kwargs):
    def fwd_pass(input):
        #vq_vae_model = model(*args, **kwargs)
        #output = vq_vae_model(input)
        #return output
        return model(*args, **kwargs)(input)
    return hk.without_apply_rng(hk.transform(fwd_pass))

def build_model(input, var, kwargs):
    cfg = {kwargs[i].name: kwargs[i].value for i in dir(kwargs)}
    model = create_fwd_fn(VQ_VAE, Encoder, Decoder, Quantizer, var, **cfg)
    return model


def get_loss_fn(fwd, params, input):
    output = fwd.apply(params, input)
    loss = output['loss']
    return loss, output
