import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cifar import get_loader

from absl import app
from absl import flags


FLAGS = flags.FLAGS


flags.DEFINE_integer('image_size', 64, 'image spatial size', lower_bound=32)
flags.DEFINE_integer('batch', 32, 'batch_size', lower_bound=1)
flags.DEFINE_integer('num_training_updates', 1e6, 'number of steps to pass', lower_bound=1000)
flags.DEFINE_integer('embedding_dim', 64, '', lower_bound=32)
flags.DEFINE_float('commitment_coef', 0.25, 'coefficient for commitment loss')
flags.DEFINE_float('wd', 0.99, 'weight_decay')
flags.DEFINE_float('lr', 3e-4, 'learning_rate')
flags.DEFINE_list('encoder_hiddens', ['8', '16', '32', '32', '64'], 'hidden layers of encoder')
flags.DEFINE_list('encoder_hiddens', ['32', '32', '16', '8', '3'], 'hidden layers of decoder')



def init_params():
    


    optim = optax.adam(FLAGS.lr)

    return optim 


def train():
    print(FLAGS)

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    rng = jax.random.PRNGKey(33)
    loader, variance = get_loader(FLAGS.batch)
    #params, state = forward.init(rng, next(loader))
    #opt_state = optimizer.init(params)

    





if __name__ == '__main__':
    app.run(train)
