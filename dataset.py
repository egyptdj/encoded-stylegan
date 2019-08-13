import os
import PIL.Image
import numpy as np
import tensorflow as tf
from stylegan.training import dataset


class DatasetEncodedStyleGAN(object):
    def __init__(self, data_dir, testim_dir, minibatch_size):
        super(DatasetEncodedStyleGAN, self).__init__()
        self.data_dir = data_dir
        self.testim_dir = testim_dir
        self.minibatch_size = minibatch_size
        self.dataset = dataset.load_dataset(data_dir=self.data_dir, tfrecord_dir='ffhq', verbose=False)
        self.dataset.configure(self.minibatch_size)
        self.image_list = [image for image in os.listdir(self.testim_dir) if image.endswith("png") or image.endswith("jpg")]
        assert len(self.image_list)>0
        self.is_built = False

    def build(self, uint=False, nchw=False):
        self.image, self.labels = self.dataset.get_minibatch_tf()
        test_im = np.array(PIL.Image.open(self.testim_dir+"/"+self.image_list[0]))[np.newaxis,...]
        test_im = tf.image.resize(tf.constant(test_im), [1024,1024])
        if not uint: test_im = tf.cast(test_im, tf.float32)/255.0
        if nchw: test_im = tf.transpose(test_im, perm=[0,3,1,2])

        self.test_image = tf.Variable(initial_value=test_im, shape=[1,3,1024,1024], dtype=tf.float32, name='test_image')
        tf.add_to_collection("TEST_OPS", self.test_image)
        if not uint: self.image = tf.cast(self.image, tf.float32)/255.0
        if not nchw: self.image = tf.transpose(self.image, perm=[0,2,3,1])
        self.is_built = True
