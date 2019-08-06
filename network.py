from dataset import *
from model import *
from graph import *
from session import *


class NetworkEncodedStyleGAN(object):
    def __init__(self, data_dir, stylegan_model):
        super(NetworkEncodedStyleGAN, self).__init__()
        self.dataset = DatasetEncodedStyleGAN(data_dir)
        self.model = ModelEncodedStyleGAN(stylegan_model)
        self.graph = GraphEncodedStyleGAN()
        self.session = SessionEncodedStyleGAN()
        self.is_built = False

    def build(self):
        # self.dataset.build(nchw=True)
        # self.model.build(self.dataset.image)
        self.graph.build(self.dataset, self.model)
        self.session.build(self.graph)
        self.is_built = True

    def train(self, learning_rate, num_iter, save_iter, result_dir):
        self.session.train(learning_rate=learning_rate, num_iter=num_iter, save_iter=save_iter, result_dir=result_dir)
