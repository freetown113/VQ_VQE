import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
from itertools import count
import cv2

from cifar import get_loader
from loss import build_model, get_loss_fn

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


def make_grid(samples, num_cols=8, rescale=True):
    batch_size, height, width, ch = samples.shape
    assert batch_size % num_cols == 0
    num_rows = batch_size // num_cols
    samples = jnp.split(samples, num_rows, axis=0)
    samples = jnp.concatenate(samples, axis=1)
    samples = jnp.split(samples, num_cols, axis=0)
    samples = jnp.concatenate(samples, axis=2).squeeze(0)
    return samples


class VQ:
    def __init__(self, args):
        self.args = args

    #@functools.partial(jax.jit, static_argnums=0)
    def init_params(self, rng, dummy, var):
        key, subkey = jax.random.split(rng, num=2)
        self.fwdpass = build_model(dummy, var, self.args)

        params = self.fwdpass.init(key, dummy)

        self.optim = optax.adam(self.args.lr)
        optim_params = self.optim.init(params)

        states = dict({
            'params': params,
            'optim': optim_params
        })

        return states


    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state, image):

        (loss, model_output), grads = jax.value_and_grad(self.fwdpass.apply, has_aux=True)(state['params'], image)

        updates, opt_state = self.optim.update(grads, state['optim'])
        params = optax.apply_updates(state['params'], updates)

        states = dict({
         'params': params,
         'optim': opt_state
        })

        return states, model_output


def train(argv):

    rng = jax.random.PRNGKey(33)
    loader, variance = get_loader(FLAGS.batch, FLAGS.var)
    vq = VQ(FLAGS)
    vq_states = vq.init_params(rng, next(iter(loader)), variance)

    for i in count():
        train_losses = []
        train_recon_errors = []
        train_perplexities = []
        train_vqvae_loss = []

        for step, img in enumerate(loader):
            vq_states, log = vq.update(vq_states, img)

            log = jax.device_get(log)
            train_losses.append(log['loss'])
            train_perplexities.append(log['perplex'])
            train_recon_errors.append(log['reconstruct'])


        print(f'[Step {i * len(loader) + step}] ' +
              ('train loss: %f ' % np.mean(train_losses)) +
              ('recon_error: %.3f ' % np.mean(train_recon_errors)) +
              ('perplexity: %.3f ' % np.mean(train_perplexities))
               )


        smp = jax.device_get(img[-16:])
        #Our model outputs values in [-1, 1] so scale it back to [0, 1].
        (loss, model_output), grads = jax.value_and_grad(vq.fwdpass.apply, has_aux=True)(vq_states['params'], img)
        smp = ((smp + 1.0) / 2.0) * 255.
        rec = jax.device_get(((model_output['reconstructed'] + 1) / 2.) * 255.)[-16:]
        #print(rec.shape, rec.max())
        res = np.concatenate((smp, rec), axis=1)
        #print(res.shape)
        res = np.array(make_grid(res, 16)).astype(np.uint8)
        #print(res.shape, res.max())
        cv2.imwrite('./resulting.jpg', res)



if __name__ == '__main__':
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    print(local_devices)
    print(global_devices)
    
    with jax.default_device(global_devices[0]):
        app.run(train)







