import functools
from itertools import count
from typing import Any, NamedTuple, Callable
import os
import cv2

import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from cifar import get_loader


class Encoder(hk.Module):
    def __init__(self, enc_hid=(8, 16, 32, 32, 64), name=None):
        super().__init__(name=name)
        self.hid = tuple([int(i) for i in enc_hid])

    def __call__(self, x):
        for id, out in enumerate(self.hid, start=1):
            x = hk.Conv2D(out, (4, 4), stride=1 if id%2 else 2,
                          padding='SAME', name=f'enc_{id}')(x)
            x = jax.nn.leaky_relu(x)

        x = hk.Conv2D(out, (1, 1), stride=1)(x)
        return x


class Decoder(hk.Module):
    def __init__(self, dec_hid=(32, 32, 16, 8, 3), name=None):
       super().__init__(name=name)
       self.hid = tuple([int(i) for i in dec_hid])

    def __call__(self, x):
       for i, out in enumerate(self.hid, start=1):
           x = hk.Conv2DTranspose(out, [4, 4], stride=1 if i%2 else 2,
                                  padding='SAME', name=f'dec_{i}')(x)
           x = jax.nn.leaky_relu(x)
       return x


class Quantizer(hk.Module):
    def __init__(self, num_emb, emb_dim, commitment_coeff=0.25, name=None):
        super().__init__(name=name)
        self.dim = emb_dim
        self.num = num_emb
        self.com_coef = commitment_coeff
        self.dict = hk.get_parameter(name='EmbeddingDictionary', 
                                     shape=(emb_dim, num_emb), 
                                     dtype=np.float32, 
                                     init=hk.initializers.VarianceScaling(distribution="uniform"))

        
    def __call__(self, x):
        input = x
        assert x.shape[-1] == self.dim
        original_shape = x.shape

        x = jnp.reshape(x, [-1, self.dim])
        dists = self.calc_distances_matrix(x)

        indexes = jnp.argmin(dists, axis=-1)
        encodings = jax.nn.one_hot(indexes, self.num)
        
        quantized = jnp.matmul(encodings, jnp.transpose(self.dict, axes=(1,0)))
        quantized = jnp.reshape(quantized, original_shape)

        latent_loss = jnp.square(jax.lax.stop_gradient(input) - quantized)
        commitment_loss = jnp.square(jax.lax.stop_gradient(quantized) - input)
  
        loss = jnp.mean(latent_loss + self.com_coef * commitment_loss)

        quantized = input + jax.lax.stop_gradient(quantized - input)
        avg_probs = jnp.mean(encodings, 0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        out = dict({'quantized': quantized, 'loss': loss, 'perplexity': perplexity})
        return out


    def calc_distances_matrix(self, x):
        similarity = jnp.matmul(x, self.dict)

        distances = jnp.sum(jnp.square(x), axis=-1, keepdims=True) + \
                    jnp.sum(jnp.square(self.dict), axis=0) - \
                    2 * similarity

        return distances


class VQ_VAE(hk.Module):
    def __init__(self, enc, dec, quant, variance, *args, **kwargs):
        super().__init__(name=kwargs['name'])
        self.enc = enc(kwargs['enc_hid'], name=kwargs['name']+'_ENC')
        self.dec = dec(kwargs['dec_hid'], name=kwargs['name']+'_DEC')
        self.quant = quant(kwargs['num_emb'], kwargs['emb_dim'], name=kwargs['name']+'_QNT')
        self.var = variance

    def __call__(self, x):
        encoded = self.enc(x)
        quantized = self.quant(encoded)
        decoded = self.dec(quantized['quantized'])

        reconstructed_loss = jnp.square(x - decoded).mean() / self.var

        loss = reconstructed_loss + quantized['loss']

        return loss, dict({'loss': loss, 
                     'reconstruct': reconstructed_loss, 
                     'decoded': decoded, 
                     'perplex': quantized['perplexity'],
                     'reconstructed': jax.lax.stop_gradient(decoded)
                })









