import os
import PIL.Image
import numpy as np
import tensorflow as tf


class DatasetEncodedStyleGAN(object):
    def __init__(self, data_dir, idx):
        super(DatasetEncodedStyleGAN, self).__init__()
        self.data_dir = data_dir
        self.idx = idx
        self.image_list = [image for image in os.listdir(self.data_dir) if image.endswith("png") or image.endswith("jpg")]
        assert len(self.image_list)>0
        self.is_built = False

    def build(self, uint=False, nchw=False):
        im = np.array(PIL.Image.open(self.data_dir+"/"+self.image_list[self.idx]))
        if len(im.shape)==2:
            im = im[np.newaxis,...]
            im = np.stack([im, im, im], axis=-1)
        elif len(im.shape)==3:
            im = im[np.newaxis,...]
        self.image = tf.image.resize(tf.constant(im), [1024,1024])
        if not uint: self.image = tf.cast(self.image, tf.float32)/255.0
        if nchw: self.image = tf.transpose(self.image, perm=[0,3,1,2])
        self.is_built = True
