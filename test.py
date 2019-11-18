import os
import sys
import utils
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'progan'))
from tqdm import tqdm
from vgg import Vgg16
from regularizer import modeseek
from lpips import lpips_tf
from laploss import laploss
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from stylegan.training import dataset
from stylegan.training.misc import save_pkl
from stylegan.training.networks_stylegan import *

def main():
    tflib.init_tf()

    url = '/media/bispl/workssd/model1024.pkl'
    with open(url, 'rb') as f: models = pickle.load(f)

    encoder, decoder = models
    im_in = tf.placeholder(tf.uint8, [1,1024,1024,3])
    im_preproc = tf.cast(tf.transpose(im_in, [0,3,1,2]), tf.float32)/255.0
    encoded_latents = encoder.get_output_for(im_preproc)

    im_obama = np.array(PIL.Image.open('images/validation/barack-obama_01.png').resize((1024,1024)))[np.newaxis,...]
    im_trump = np.array(PIL.Image.open('images/validation/donald-trump-2_01.png').resize((1024,1024)))[np.newaxis,...]

    sess = tf.get_default_session()
    obama_latents = sess.run(encoded_latents, feed_dict={im_in: im_obama})
    trump_latents = sess.run(encoded_latents, feed_dict={im_in: im_trump})
    merged_latents = []
    for i in range(10):
        merged_latents.append((obama_latents[i]+trump_latents[i])/2.0)
    merged_decoded = decoder.get_output_for(obama_latents[0],obama_latents[1],merged_latents[2],merged_latents[3],merged_latents[4],merged_latents[5],merged_latents[6],merged_latents[7],merged_latents[8],merged_latents[9],is_validation=True)
    merged_im = sess.run(merged_decoded)
    merged_im = np.transpose(np.squeeze(merged_im), [1,2,0])
    import matplotlib.pyplot as plt
    plt.imshow(merged_im)
    import ipdb; ipdb.set_trace()

if __name__=='__main__':
    main()
