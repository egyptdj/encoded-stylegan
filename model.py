import tensorflow as tf


class ModelEncodedStyleGAN(object):
    def __init__(self, stylegan_model):
        super(ModelEncodedStyleGAN, self).__init__()
        self.generator = stylegan_model
        self.encoder = Encoder()



class Encoder(object):
    def __init__(self):
        pass
