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
        self.image_list = [image for image in os.listdir(self.testim_dir) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
        assert len(self.image_list)>0
        self.is_built = False

    def build(self, uint=False, nchw=False):
        self.image, self.labels = self.dataset.get_minibatch_tf()
        test_im = np.stack([np.array(PIL.Image.open(self.testim_dir+"/"+image_path).resize((1024,1024))) for image_path in self.image_list[:4]], axis=0)
        test_im = tf.transpose(tf.constant(test_im), perm=[0,3,1,2])
        self.test_image = tf.Variable(initial_value=test_im, dtype=tf.uint8, name='test_image')

        if not uint: self.image, self.test_image = tf.cast(self.image, tf.float32)/255.0, tf.cast(self.test_image, tf.float32)/255.0
        if not nchw: self.image, self.test_image = tf.transpose(self.image, perm=[0,2,3,1]), tf.transpose(self.test_image, perm=[0,2,3,1])
        tf.add_to_collection("TEST_OPS", self.test_image)
        self.is_built = True
