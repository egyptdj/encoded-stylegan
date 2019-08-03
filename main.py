import os
import sys
import utils
import pickle
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from network import NetworkEncodedStyleGAN


class EncodedStyleGAN(object):
    def __init__(self):
        super(EncodedStyleGAN, self).__init__()
        self.base_option = utils.option.parse()
        self.is_initialized = False


    def _initialize(self):
        tflib.init_tf()
        url = os.path.join(self.base_option['cache_dir'], 'karras2019stylegan-ffhq-1024x1024.pkl')
        with open(url, 'rb') as f:
            _, _, Gs = pickle.load(f)
        # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        # with dnnlib.util.open_url(url, cache_dir=self.base_option['cache_dir']) as f:
        #     _, _, Gs = pickle.load(f)

        self.network = NetworkEncodedStyleGAN(
            data_dir = self.base_option['data_dir'],
            minibatch_size = self.base_option['minibatch_size'],
            stylegan_model=Gs.components.synthesis)

        self.network.build()
        self.is_initialized = True

    def train(self):
        if not self.is_initialized: self._initialize()
        self.network.train()


if __name__=='__main__':
    experiment = EncodedStyleGAN()
    experiment.train()
