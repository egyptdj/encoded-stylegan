import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from training import dataset
import config
from training import misc

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[-2], center[-1], w-center[-2], h-center[-1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[-2])**2 + (Y-center[-1])**2)

    mask = dist_from_center <= radius
    return mask


def encode(network_pkl):
    slice = PIL.Image.open('/home/bispl/Pictures/t1mets.png')
    img_shape = slice.size
    ratio = 256.0/max(img_shape)
    new_shape = tuple([int(x*ratio) for x in img_shape])

    slice = slice.resize(new_shape)
    padded_slice = PIL.Image.new('L', (256, 256))
    padded_slice.paste(slice, ((256-new_shape[0])//2, (256-new_shape[1])//2))
    padded_slice = np.asarray(padded_slice)/255.0
    padded_slice = (padded_slice + 1.0) / 2.0
    mets = padded_slice

    slice = PIL.Image.open('/home/bispl/Pictures/brain_mri_transversal_t1_002.jpg')
    img_shape = slice.size
    ratio = 256.0/max(img_shape)
    new_shape = tuple([int(x*ratio) for x in img_shape])

    slice = slice.resize(new_shape)
    padded_slice = PIL.Image.new('L', (256, 256))
    padded_slice.paste(slice, ((256-new_shape[0])//2, (256-new_shape[1])//2))
    padded_slice = np.asarray(padded_slice)/255.0
    padded_slice = (padded_slice + 1.0) / 2.0

    normal = padded_slice

    # Construct networks.
    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            G, E, Dx, Dz, Gs, Es = misc.load_pkl(network_pkl)

            input_image = tf.placeholder(tf.float32, Es.input_shape)
            input_latent = tf.placeholder(tf.float32, Es.output_shape)

            encoded_latent = Es.get_output_for(input_image, None)
            encoded_image = Gs.get_output_for(input_latent, None)

            # Load training set.
            training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, tfrecord_dir='hcp_t1_t2_ax_preproc_acpc_dc_restore')
            image_batch, label_batch = training_set.get_minibatch_np(2)
            image_batch = np.float32(image_batch)/255.0
            mask = np.invert(create_circular_mask(256, 256, (150, 128), 25))[np.newaxis, np.newaxis, ...]
            corrupted_image_batch = image_batch*mask.astype(np.float32)

            original_image_latent = sess.run(encoded_latent, {input_image: image_batch*2.0-1.0})
            original_image_recon = sess.run(encoded_image, {input_latent: original_image_latent})
            original_image_recon = (original_image_recon + 1.0) / 2.0

            corrupted_image_latent = sess.run(encoded_latent, {input_image: corrupted_image_batch*2.0-1.0})
            corrupted_image_recon = sess.run(encoded_image, {input_latent: corrupted_image_latent})
            corrupted_image_recon = (corrupted_image_recon + 1.0) / 2.0

            difference_latent = corrupted_image_latent - original_image_latent
            difference_image_recon = sess.run(encoded_image, {input_latent: difference_latent})
            difference_image_recon = (difference_image_recon + 1.0) / 2.0

            middle_image_latent = sess.run(tf.reduce_mean(original_image_latent, axis=0, keep_dims=True))
            middle_image_recon = sess.run(encoded_image, {input_latent: middle_image_latent})
            middle_image_recon = (middle_image_recon + 1.0) / 2.0

            mets_latent = sess.run(encoded_latent, {input_image: mets[np.newaxis, np.newaxis, ...]})
            mets_image_recon = sess.run(encoded_image, {input_latent: mets_latent})
            mets_image_recon = (mets_image_recon + 1.0) / 2.0

            normal_latent = sess.run(encoded_latent, {input_image: normal[np.newaxis, np.newaxis, ...]})
            normal_image_recon = sess.run(encoded_image, {input_latent: normal_latent})
            normal_image_recon = (normal_image_recon + 1.0) / 2.0

            random_latent = np.random.normal(size=[2, 512])
            latents = np.linspace(random_latent[0], random_latent[1], 1000)

            for idx, latent in enumerate(latents):
                image = sess.run(encoded_image, {input_latent: latent[np.newaxis,:]})
                plt.imsave('/home/bispl/Pictures/im{:04d}.png'.format(idx),image[0,0,...], cmap='gray')
            # plt.imshow(original_image_recon[0,0,...])
            # plt.imshow(corrupted_image_recon[0,0,...])
            # plt.imshow(difference_image_recon[0,0,...])
            # plt.imshow(mets, cmap='gray')
            # plt.imshow(mets_image_recon[0,0,...], cmap='gray')
            # plt.imshow(normal, cmap='gray')
            # plt.imshow(normal_image_recon[0,0,...], cmap='gray')
            # plt.imshow(original_image_recon[0,0,...], cmap='gray')
            # plt.imshow(original_image_recon[1,0,...], cmap='gray')
            # plt.imshow(middle_image_recon[0,0,...], cmap='gray')
