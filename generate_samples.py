# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import argparse

def main():
    parser = argparse.ArgumentParser(description='OCT-PGAN - generate samples')
    parser.add_argument('-s', '--seed', type=int, default=None, help='random seed')
    parser.add_argument('-n', '--num_samples', type=int, default=8, help='number of samples to generate')
    parser.add_argument('-c', '--label', type=int, default=None, choices=[0,1], help='image label to generate (0: erosion, 1: rupture)')
    parser.add_argument('-t', '--target_dir', type=str, default='./', help='target directory to generate samples')
    parser.add_argument('-m', '--model', type=str, default='./network-final.pkl', help='path to the pretrained model pkl')
    args = parser.parse_args()

    # Initialize TensorFlow.
    tflib.init_tf()
    
    # Load pre-trained network.
    with open(args.model, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    Gs.print_layers()

    # set random seed
    rnd = np.random.RandomState(args.seed)
    
    # set labels
    assert args.label is not None
    labels = np.array([[0,0]])
    labels[0, args.label] = 1
    labels = np.zeros([1, 0], np.float32)

    for i in range(args.num_samples):
        # Pick latent vector.
        latents = rnd.randn(1, Gs.input_shape[1])
        
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, labels, output_transform=fmt)

        # Save image.
        #png_filename = os.path.join(args.target_dir, 'label{}_sample{}.png'.format(args.label, i))
        png_filename = os.path.join(args.target_dir, 'sample{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
