import os
import sys
import utils
import pickle
import tensorflow as tf
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from network import NetworkEncodedStyleGAN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))


class EncodedStyleGAN(object):
    def __init__(self):
        super(EncodedStyleGAN, self).__init__()
        self.base_option = utils.option.parse()


    def _initialize(self):
        tflib.init_tf()
        url = os.path.join(self.base_option['cache_dir'], 'karras2019stylegan-ffhq-1024x1024.pkl')
        with open(url, 'rb') as f:
            _, _, Gs = pickle.load(f)
        # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        # with dnnlib.util.open_url(url, cache_dir=self.base_option['cache_dir']) as f:
        #     _, _, Gs = pickle.load(f)

        self.network = NetworkEncodedStyleGAN(
            stylegan_model=Gs.components.synthesis,
            stylegan_graph=tf.get_default_graph(),
            stylegan_session=tf.get_default_session())


if __name__=='__main__':
    a = EncodedStyleGAN()
    a._initialize()
    import ipdb; ipdb.set_trace()
