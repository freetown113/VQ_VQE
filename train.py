import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cifar import get_loader
from loss import make_loss_fn

from absl import app
from absl import flags


FLAGS = flags.FLAGS


flags.DEFINE_integer('image_size', 64, 'image spatial size', lower_bound=32)
flags.DEFINE_integer('batch', 32, 'batch_size', lower_bound=1)
flags.DEFINE_integer('num_training_updates', int(1e6), 'number of steps to pass', lower_bound=1000)
flags.DEFINE_integer('emb_dim', 64, '', lower_bound=32)
flags.DEFINE_integer('num_emb', 512, '', lower_bound=128)
flags.DEFINE_float('commitment_coef', 0.25, 'coefficient for commitment loss')
flags.DEFINE_float('wd', 0.99, 'weight_decay')
flags.DEFINE_float('lr', 3e-4, 'learning_rate')
flags.DEFINE_string('name', 'VQ_VAE', 'Model name')
flags.DEFINE_bool('var', True, 'calculate variance of the dataset')
flags.DEFINE_list('enc_hid', ['8', '16', '32', '32', '64'], 'hidden layers of encoder')
flags.DEFINE_list('dec_hid', ['32', '32', '16', '8', '3'], 'hidden layers of decoder')



def init_params(rng, dummy, var, args):
    key, subkey = jax.random.split(rng, num=2)
    func = make_loss_fn(dummy, var, args)
    params = func.init(key, dummy)

    optim = optax.adam(args.lr)
    optim_params = optim.init(params)

    state = dict({
        'params': params,
        'optim': optim_params
    })

    return optim, func, state


def update(states, rng, ):



def train(argv):

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    rng = jax.random.PRNGKey(33)
    loader, variance = get_loader(FLAGS.batch, FLAGS.var)
    optim, fwd, state = init_params(rng, next(iter(loader)), variance, FLAGS)






if __name__ == '__main__':
    app.run(train)
