from dataset import *
from model import *
from graph import *
from session import *


class NetworkEncodedStyleGAN(object):
    def __init__(self, stylegan_model, stylegan_graph, stylegan_session):
        super(NetworkEncodedStyleGAN, self).__init__()
        self.model = ModelEncodedStyleGAN(stylegan_model)
        self.graph = GraphEncodedStyleGAN(stylegan_graph)
        self.session = SessionEncodedStyleGAN(stylegan_session)

    def train(self):
        pass
