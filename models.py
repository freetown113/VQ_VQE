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
    def __init__(self, out_ch, hiddens=(8, 16, 32, 32, 64), name=None)
        super().__init(name=name)
        self.out = out_ch
        self.hid = hiddens

    def __call__(self, x):
        for id, out in enumerate(hidden, start=1):
            x = hk.Conv2D(out, [4, 4], stride=1 if id%2 else 2,
                          padding='SAME')(x)
            x = jax.nn.leaky_relu(x)
        return x


class Decoder(hk.Module):
    def __init__(self, in_ch, hiddens=(32, 32, 16, 8, 3), name=None):
       super().__init__(name=name)
       self.input = in_ch
       self.hid = hiddens

    def __call__(self, x):
       for i, out in enumerate(self.hid, start=1):
           x = hk.Conv2DTranspose(out, [4, 4], stride=1 if id%2 else 2,
                                  padding='SAME')(x)
           x = jax.nn.leaky_relu(x)
       return x
        

class VQ(hk.Module):
    def __init__(self, num_emb, emb_dim, commitment_coeff=0.25):
        super().__init__()
        self.dim = emb_dim
        self.num = num_emb
        self.com_coef = commitment_coeff
        self.dict = hk.get_parameter('EmbeddingDictionary', shape=(num_emb, emb_dim), 
                                     type=np.float32, 
                                     init=hk.initializers.VarianceScaling(distribution="uniform"))

        
    def __call__(self, x):
        assert x.shape[-1] == self.dim
        original_shape = x.shape

        x = jnp.reshape(x, [-1, self.dim])
        dists = self.calc_distances_matrix(x)

        indexes = jnp.argmin(dists, axis=-1)
        encodings = jax.nn.one_hot(indexes, self.num)

        quantized = jnp.matmul(encodings, self.dict, transpose_b=True)
        quantized = jnp.reshape(original_shape)

        commitment_loss = 


    def calc_distances_matrix(self, x):
        similarity = jnp.matmul(x, self.dict)

        distances = jnp.sum(jnp.square(x), axis=-1) + \
                    jnp.sum(jnp.square(self.dict), axis=0) - \
                    2 * similarity

        return distances








