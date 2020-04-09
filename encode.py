import os
import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from training import dataset
import config
from training import misc
from dnnlib import tflib

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[-2], center[-1], w-center[-2], h-center[-1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[-2])**2 + (Y-center[-1])**2)

    mask = dist_from_center <= radius
    return mask

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def encode(network_pkl, target_dir):
    # Construct networks.
    tflib.init_tf()
    sess = tf.get_default_session()
    with tf.device('/gpu:0'):
        training_set = dataset.load_dataset(data_dir=config.data_dir, tfrecord_dir='hcp_t1_t2_ax_preproc_acpc_dc_restore', verbose=True, repeat=False)
        training_set.configure(1, 0)
        img, _ = training_set.get_minibatch_tf()
        # img = process_reals(img, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
        img = (tf.cast(img, tf.float32) * 2.0 / 255.0) - 1.0
        G, E, Dx, Dz, Gs, Es = misc.load_pkl(network_pkl)
        latents = []
        encoded_latent = Es.get_output_for(img, None)
        tflib.tfutil.init_uninitialized_vars()
        i = 0
        while True:
            try:
                if i%100 ==0:
                    print(i)
                latents.append(sess.run(encoded_latent))
                i += 1
            except tf.errors.OutOfRangeError:
                break
        latents = np.concatenate(latents, axis=0)
        np.save(os.path.join(target_dir, 'encoded_latent.npy'), latents)
#
# def encode(network_pkl):
#     slice = PIL.Image.open('/home/bispl/Pictures/t1mets.png')
#     img_shape = slice.size
#     ratio = 256.0/max(img_shape)
#     new_shape = tuple([int(x*ratio) for x in img_shape])
#
#     slice = slice.resize(new_shape)
#     padded_slice = PIL.Image.new('L', (256, 256))
#     padded_slice.paste(slice, ((256-new_shape[0])//2, (256-new_shape[1])//2))
#     padded_slice = np.asarray(padded_slice)/255.0
#     padded_slice = (padded_slice + 1.0) / 2.0
#     mets = padded_slice
#
#     slice = PIL.Image.open('/home/bispl/Pictures/brain_mri_transversal_t1_002.jpg')
#     img_shape = slice.size
#     ratio = 256.0/max(img_shape)
#     new_shape = tuple([int(x*ratio) for x in img_shape])
#
#     slice = slice.resize(new_shape)
#     padded_slice = PIL.Image.new('L', (256, 256))
#     padded_slice.paste(slice, ((256-new_shape[0])//2, (256-new_shape[1])//2))
#     padded_slice = np.asarray(padded_slice)/255.0
#     padded_slice = (padded_slice + 1.0) / 2.0
#
#     normal = padded_slice
#
#     # Construct networks.
#     with tf.Session() as sess:
#         with tf.device('/gpu:0'):
#             G, E, Dx, Dz, Gs, Es = misc.load_pkl(network_pkl)
#
#             input_image = tf.placeholder(tf.float32, Es.input_shape)
#             input_latent = tf.placeholder(tf.float32, Es.output_shape)
#
#             encoded_latent = Es.get_output_for(input_image, None)
#             encoded_image = Gs.get_output_for(input_latent, None)
#
#             # Load training set.
#             training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, tfrecord_dir='hcp_t1_t2_ax_preproc_acpc_dc_restore')
#             image_batch, label_batch = training_set.get_minibatch_np(2)
#             image_batch = np.float32(image_batch)/255.0
#             mask = np.invert(create_circular_mask(256, 256, (150, 128), 25))[np.newaxis, np.newaxis, ...]
#             corrupted_image_batch = image_batch*mask.astype(np.float32)
#
#             original_image_latent = sess.run(encoded_latent, {input_image: image_batch*2.0-1.0})
#             original_image_recon = sess.run(encoded_image, {input_latent: original_image_latent})
#             original_image_recon = (original_image_recon + 1.0) / 2.0
#
#             corrupted_image_latent = sess.run(encoded_latent, {input_image: corrupted_image_batch*2.0-1.0})
#             corrupted_image_recon = sess.run(encoded_image, {input_latent: corrupted_image_latent})
#             corrupted_image_recon = (corrupted_image_recon + 1.0) / 2.0
#
#             difference_latent = corrupted_image_latent - original_image_latent
#             difference_image_recon = sess.run(encoded_image, {input_latent: difference_latent})
#             difference_image_recon = (difference_image_recon + 1.0) / 2.0
#
#             middle_image_latent = sess.run(tf.reduce_mean(original_image_latent, axis=0, keep_dims=True))
#             middle_image_recon = sess.run(encoded_image, {input_latent: middle_image_latent})
#             middle_image_recon = (middle_image_recon + 1.0) / 2.0
#
#             mets_latent = sess.run(encoded_latent, {input_image: mets[np.newaxis, np.newaxis, ...]})
#             mets_image_recon = sess.run(encoded_image, {input_latent: mets_latent})
#             mets_image_recon = (mets_image_recon + 1.0) / 2.0
#
#             normal_latent = sess.run(encoded_latent, {input_image: normal[np.newaxis, np.newaxis, ...]})
#             normal_image_recon = sess.run(encoded_image, {input_latent: normal_latent})
#             normal_image_recon = (normal_image_recon + 1.0) / 2.0
#
#             random_latent = np.random.normal(size=[2, 512])
#             latents = np.linspace(random_latent[0], random_latent[1], 1000)
#
#             for idx, latent in enumerate(latents):
#                 image = sess.run(encoded_image, {input_latent: latent[np.newaxis,:]})
#                 plt.imsave('/home/bispl/Pictures/im{:04d}.png'.format(idx),image[0,0,...], cmap='gray')
#             # plt.imshow(original_image_recon[0,0,...])
#             # plt.imshow(corrupted_image_recon[0,0,...])
#             # plt.imshow(difference_image_recon[0,0,...])
#             # plt.imshow(mets, cmap='gray')
#             # plt.imshow(mets_image_recon[0,0,...], cmap='gray')
#             # plt.imshow(normal, cmap='gray')
#             # plt.imshow(normal_image_recon[0,0,...], cmap='gray')
#             # plt.imshow(original_image_recon[0,0,...], cmap='gray')
#             # plt.imshow(original_image_recon[1,0,...], cmap='gray')
#             # plt.imshow(middle_image_recon[0,0,...], cmap='gray')
