import tensorflow as tf
from stylegan.training import dataset


class DatasetEncodedStyleGAN(object):
    def __init__(self, data_dir, minibatch_size):
        super(DatasetEncodedStyleGAN, self).__init__()
        self.data_dir = data_dir
        self.minibatch_size = minibatch_size
        self.dataset = dataset.load_dataset(data_dir=self.data_dir, tfrecord_dir='ffhq', verbose=True)
        self.dataset.configure(self.minibatch_size)
        self.is_built = False

    def build(self, uint=False, nchw=False):
        self.image, self.labels = self.dataset.get_minibatch_tf()
        if not uint: self.image = tf.cast(self.image, tf.float32)/255.0
        if not nchw: self.image = tf.transpose(self.image, perm=[0,2,3,1])
        self.is_built = True
