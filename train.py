
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'progan'))
import time
import numpy as np
import tensorflow as tf

import config
from stylegan.training import misc, dataset
from stylegan.dnnlib.tflib import tfutil

from stylegan import dnnlib
from stylegan.dnnlib import tflib
from vgg import Vgg16

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(G, training_set,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    gw = 1; gh = 1
    if size == '1080p':
        gw = np.clip(1920 // G.output_shape[3], 3, 32)
        gh = np.clip(1080 // G.output_shape[2], 2, 32)
    if size == '4k':
        gw = np.clip(3840 // G.output_shape[3], 7, 32)
        gh = np.clip(2160 // G.output_shape[2], 4, 32)

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            if layout == 'row_per_class' and training_set.label_size > 0:
                if label[0, y % training_set.label_size] == 0.0:
                    continue
            reals[idx] = real[0]
            labels[idx] = label[0]
            break

    # Generate latents.
    latents = misc.random_latents(gw * gh, G)
    return (gw, gh), reals, labels, latents

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

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
            x = tfutil.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
        minibatch_base          = 4,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {4: 2, 8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 2, 512: 2, 1024: 2},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {256: 16, 512: 8, 1024: 4},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base            = 0.001,    # Learning rate for the generator.
        G_lrate_dict            = {1024: 0.0015},       # Resolution-specific overrides.
        D_lrate_base            = 0.001,    # Learning rate for the discriminator.
        D_lrate_dict            = {1024: 0.0015},       # Resolution-specific overrides.
        tick_kimg_base          = 160,      # Default interval of progress snapshots.
        tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}): # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * num_gpus)

        # Other parameters.
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_progressive_autoencoder(
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    D_repeats               = 1,            # How many times the discriminator is trained per G iteration.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 12000,        # Total length of the training, measured in thousands of real images.
    mirror_augment          = True,        # Enable mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 1,            # How often to export image snapshots?
    network_snapshot_ticks  = 10,           # How often to export network snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,          # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):         # Assumed wallclock time at the beginning. Affects reporting.

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir='/media/bispl/workdisk/FFHQ_flickrface/tfrecords', tfrecord_dir='ffhq', verbose=True)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            encoder, avg_encoder, generator, avg_generator, latent_critic, image_critic = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            encoder = tflib.Network("encoder", func_name='encoder.E_basic', out_shape=[512], num_channels=3, resolution=training_set.shape[1], structure='linear')
            avg_encoder = encoder.clone('avg_encoder')
            generator = tflib.Network("generator", func_name='stylegan.training.networks_progan.G_paper', out_shape=[512], num_channels=3, resolution=training_set.shape[1])
            avg_generator = generator.clone('avg_generator')
            latent_critic = tflib.Network("z_critic", func_name='stylegan.training.networks_stylegan.G_mapping', dlatent_size=1, mapping_layers=8, latent_size=512, normalize_latents=False)
            image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_progan.D_paper', num_channels=3, resolution=1024)
        avg_encoder_update_op = avg_encoder.setup_as_moving_average_of(encoder, beta=G_smoothing)
        avg_generator_update_op = avg_generator.setup_as_moving_average_of(generator, beta=G_smoothing)
    encoder.print_layers(); generator.print_layers(); latent_critic.print_layers(); image_critic.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, num_gpus)
        labels_split    = tf.split(labels, num_gpus)

    encoder_optimizer = tflib.Optimizer(name='encoder_optimizer', learning_rate=lrate_in, beta1=0.0, beta2=0.99, epsilon=1e-8)
    generator_optimizer = tflib.Optimizer(name='generator_optimizer', learning_rate=lrate_in, beta1=0.0, beta2=0.99, epsilon=1e-8)
    z_critic_optimizer = tflib.Optimizer(name='z_critic_optimizer', learning_rate=lrate_in, beta1=0.0, beta2=0.99, epsilon=1e-8)
    y_critic_optimizer = tflib.Optimizer(name='y_critic_optimizer', learning_rate=lrate_in, beta1=0.0, beta2=0.99, epsilon=1e-8)

    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            encoder_gpu = encoder if gpu == 0 else encoder.clone(encoder.name + '_shadow')
            generator_gpu = generator if gpu == 0 else generator.clone(generator.name + '_shadow')
            latent_critic_gpu = latent_critic if gpu == 0 else latent_critic_iter.clone(latent_critic.name + '_shadow')
            image_critic_gpu = image_critic if gpu == 0 else image_critic_iter.clone(image_critic.name + '_shadow')
            lod_assign_ops = [tf.assign(encoder_gpu.find_var('lod'), lod_in), tf.assign(generator_gpu.find_var('lod'), lod_in), tf.assign(image_critic_gpu.find_var('lod'), lod_in)]
            reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)

            with tf.name_scope('loss'), tf.control_dependencies(lod_assign_ops):
                MSE = tf.keras.losses.MeanSquaredError()
                MAE = tf.keras.losses.MeanAbsoluteError()
                z_fakes_gpu = encoder_gpu.get_output_for(reals_gpu)
                fakes_gpu = generator_gpu.get_output_for(z_fakes_gpu, None, is_validation=True, use_noise=False, randomize_noise=False)
                tf.add_to_collection('IMAGE_ENCODED', (fakes_gpu+1.0)/2)
                with tf.name_scope('regression_loss'):
                    regression_loss = 0.0

                    # L2 Loss
                    if coeff_lambda['l2'] > 0.0:
                        l2_loss = MSE(reals_gpu, fakes_gpu)
                        tf.add_to_collection('LOSS_L2', l2_loss)
                        regression_loss += coeff_lambda['l2']*l2_loss

                    # L1 Loss
                    if coeff_lambda['l1'] > 0.0:
                        l1_loss = MAE(reals_gpu, fakes_gpu)
                        tf.add_to_collection('LOSS_L1', l1_loss)
                        regression_loss += coeff_lambda['l1']*l1_loss

                    # VGG loss
                    if coeff_lambda['vgg'] > 0.0:
                        image_vgg = Vgg16('cache'+'/vgg16.npy')
                        image_vgg.build(tf.image.resize(tf.transpose(reals_gpu, perm=[0,2,3,1]), [224,224]))
                        image_perception = [image_vgg.conv1_1, image_vgg.conv1_2, image_vgg.conv3_2, image_vgg.conv4_2]
                        encoded_vgg = Vgg16('cache'+'/vgg16.npy')
                        encoded_vgg.build(tf.image.resize(tf.transpose(fakes_gpu, perm=[0,2,3,1]), [224,224]))
                        encoded_perception = [encoded_vgg.conv1_1, encoded_vgg.conv1_2, encoded_vgg.conv3_2, encoded_vgg.conv4_2]
                        vgg_loss = tf.reduce_sum([MSE(image, encoded) for image, encoded in zip(image_perception, encoded_perception)]) # https://github.com/machrisaa/tensorflow-vgg
                        tf.add_to_collection('LOSS_VGG', vgg_loss)
                        regression_loss += coeff_lambda['vgg']*vgg_loss

                    # LPIPS loss
                    if coeff_lambda['lpips'] > 0.0:
                        lpips_url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'
                        with dnnlib.util.open_url(lpips_url, cache_dir='cache') as lpips:
                            lpips_network =  pickle.load(lpips)
                        lpips_loss = lpips_network.get_output_for(reals_gpu, fakes_gpu)
                        tf.add_to_collection('LOSS_LPIPS', lpips_loss)
                        regression_loss += coeff_lambda['lpips']*lpips_loss

                    # MSSIM loss
                    if coeff_lambda['mssim'] > 0.0:
                        mssim_loss = 1-tf.image.ssim_multiscale(tf.transpose(reals_gpu, perm=[0,2,3,1]), tf.transpose(fakes_gpu, perm=[0,2,3,1]), 1.0)
                        tf.add_to_collection('LOSS_MSSIM', mssim_loss)
                        regression_loss += coeff_lambda['mssim']*mssim_loss

                    # LOGCOSH loss
                    if coeff_lambda['logcosh'] > 0.0:
                        logcosh_loss = tf.keras.losses.logcosh(tf.transpose(reals_gpu, perm=[0,2,3,1]), tf.transpose(fakes_gpu, perm=[0,2,3,1]))
                        tf.add_to_collection('LOSS_LOGCOSH', logcosh_loss)
                        regression_loss += coeff_lambda['logcosh']*logcosh_loss


                with tf.name_scope('z_domain_loss'):
                    fake_latent = tf.random.normal(shape=tf.shape(z_fakes_gpu), name='z_rand')
                    real_latent = tf.identity(z_fakes_gpu, name='z_real')

                    fake_latent_critic_out = latent_critic_gpu.get_output_for(tf.reshape(fake_latent, [-1,512]), None)
                    real_latent_critic_out = latent_critic_gpu.get_output_for(tf.reshape(real_latent, [-1,512]), None)

                    with tf.name_scope("fake_loss"):
                        fake_latent_loss = tf.losses.compute_weighted_loss(\
                            losses=fake_latent_critic_out, \
                            weights=1.0, scope='fake_latent_loss')

                    with tf.name_scope("real_loss"):
                        real_latent_critic_loss = tf.losses.compute_weighted_loss(\
                            losses=real_latent_critic_out, \
                            weights=1.0, scope='real_latent_critic_loss')

                        fake_latent_critic_loss = tf.losses.compute_weighted_loss(\
                            losses=fake_latent_critic_out, \
                            weights=1.0, scope='fake_latent_critic_loss')

                        # WASSERSTEIN GAN - GRADIENT PENALTY
                        with tf.name_scope('latent_gradient_penalty'):
                            epsilon = tf.random.uniform([], name='epsilon')
                            gradient_latent = tf.identity((epsilon * real_latent + (1-epsilon) * fake_latent), name='latent_gradient')
                            latent_critic_gradient_out = latent_critic_gpu.get_output_for(tf.reshape(gradient_latent, [-1,512]), None)
                            latent_gradients = tf.gradients(latent_critic_gradient_out, gradient_latent, name='latent_gradients')
                            latent_gradients_norm = tf.norm(latent_gradients[0], ord=2, name='latent_gradient_norm')
                            latent_gradient_penalty = tf.square(latent_gradients_norm -1)

                    z_critic_real_loss = -real_latent_critic_loss + fake_latent_critic_loss + coeff_lambda['gp']*latent_gradient_penalty
                    z_critic_fake_loss = -fake_latent_loss

                with tf.name_scope('y_domain_loss'), tf.control_dependencies(lod_assign_ops):
                    fake_image = generator_gpu.get_output_for(tf.random.normal(shape=tf.shape(z_fakes_gpu), name='z_rand'), None, is_validation=True, use_noise=False, randomize_noise=False)
                    real_image = tf.identity(reals_gpu, name='y_real')

                    fake_image_critic_out = image_critic_gpu.get_output_for(fake_image, None)
                    real_image_critic_out = image_critic_gpu.get_output_for(real_image, None)
                    with tf.name_scope("fake_loss"):
                        fake_image_loss = tf.losses.compute_weighted_loss(\
                            losses=fake_image_critic_out, \
                            weights=1.0, scope='fake_image_loss')

                    with tf.name_scope("real_loss"):
                        real_image_critic_loss = tf.losses.compute_weighted_loss(\
                            losses=real_image_critic_out, \
                            weights=1.0, scope='real_image_critic_loss')

                        fake_image_critic_loss = tf.losses.compute_weighted_loss(\
                            losses=fake_image_critic_out, \
                            weights=1.0, scope='fake_image_critic_loss')

                        # WASSERSTEIN GAN - GRADIENT PENALTY
                        with tf.name_scope('image_gradient_penalty'):
                            epsilon = tf.random.uniform([], name='epsilon')
                            gradient_image = tf.identity((epsilon * real_image + (1-epsilon) * fake_image), name='image_gradient')
                            image_critic_gradient_out = image_critic_gpu.get_output_for(gradient_image, None)
                            image_gradients = tf.gradients(image_critic_gradient_out, gradient_image, name='image_gradients')
                            image_gradients_norm = tf.norm(image_gradients[0], ord=2, name='image_gradient_norm')
                            image_gradient_penalty = tf.square(image_gradients_norm -1)

                    y_critic_real_loss = -real_image_critic_loss + fake_image_critic_loss + coeff_lambda['gp']*image_gradient_penalty
                    y_critic_fake_loss = -fake_image_loss

            with tf.name_scope('final_losses'):
                encoder_loss = regression_loss + z_critic_fake_loss
                generator_loss = regression_loss + y_critic_fake_loss
                z_critic_loss = tf.identity(z_critic_real_loss)
                y_critic_loss = tf.identity(y_critic_real_loss)
                tf.add_to_collection("LOSS_REGRESSION", regression_loss)
                tf.add_to_collection("LOSS_ENCODER", encoder_loss)
                tf.add_to_collection("LOSS_GENERATOR", generator_loss)
                tf.add_to_collection("LOSS_Z_CRITIC_REAL", z_critic_real_loss)
                tf.add_to_collection("LOSS_Z_CRITIC_REAL_SEP", real_latent_critic_loss)
                tf.add_to_collection("LOSS_Z_CRITIC_GP", latent_gradient_penalty)
                tf.add_to_collection("LOSS_Y_CRITIC_REAL", y_critic_real_loss)
                tf.add_to_collection("LOSS_Y_CRITIC_REAL_SEP", real_image_critic_loss)
                tf.add_to_collection("LOSS_Y_CRITIC_GP", image_gradient_penalty)
                tf.add_to_collection("LOSS_Z_CRITIC_FAKE", z_critic_fake_loss)
                tf.add_to_collection("LOSS_Y_CRITIC_FAKE", y_critic_fake_loss)

            with tf.name_scope('metrics'):
                psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(reals_gpu, perm=[0,2,3,1]), tf.transpose(fakes_gpu, perm=[0,2,3,1]), 1.0))
                ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(reals_gpu, perm=[0,2,3,1]), tf.transpose(fakes_gpu, perm=[0,2,3,1]), 1.0))
                tf.add_to_collection('METRIC_PSNR', psnr)
                tf.add_to_collection('METRIC_SSIM', ssim)

            with tf.name_scope('backprop'):
                print("================== ENCODER VARS ==================")
                print("\n".join([v.name for v in [*encoder_gpu.trainables.values()]]))
                encoder_optimizer.register_gradients(encoder_loss, encoder_gpu.trainables)

                print("================== GENERATOR VARS ==================")
                print("\n".join([v.name for v in [*generator_gpu.trainables.values()]]))
                generator_optimizer.register_gradients(generator_loss, generator_gpu.trainables)

                print("================== Z_CRITIC VARS ==================")
                print("\n".join([v.name for v in [*latent_critic_gpu.trainables.values()]]))
                z_critic_optimizer.register_gradients(z_critic_loss, latent_critic_gpu.trainables)

                print("================== Y_CRITIC VARS ==================")
                print("\n".join([v.name for v in [*image_critic_gpu.trainables.values()]]))
                y_critic_optimizer.register_gradients(y_critic_loss, image_critic_gpu.trainables)

    encoder_optimize = encoder_optimizer.apply_updates()
    generator_optimize = generator_optimizer.apply_updates()

    z_critic_optimize = z_critic_optimizer.apply_updates()
    y_critic_optimize = y_critic_optimizer.apply_updates()


    with tf.name_scope('summary'):
        _ = tf.summary.scalar('regression', tf.reduce_mean(tf.get_collection('LOSS_REGRESSION')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if coeff_lambda['l2'] > 0.0: _ = tf.summary.scalar('l2', tf.reduce_mean(tf.get_collection('LOSS_L2')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if coeff_lambda['l1'] > 0.0: _ = tf.summary.scalar('l1', tf.reduce_mean(tf.get_collection('LOSS_L1')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if coeff_lambda['vgg'] > 0.0: _ = tf.summary.scalar('vgg', tf.reduce_mean(tf.get_collection('LOSS_VGG')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if coeff_lambda['lpips'] > 0.0: _ = tf.summary.scalar('lpips', tf.reduce_mean(tf.get_collection('LOSS_LPIPS')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if coeff_lambda['mssim'] > 0.0: _ = tf.summary.scalar('mssim', tf.reduce_mean(tf.get_collection('LOSS_MSSIM')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if coeff_lambda['logcosh'] > 0.0: _ = tf.summary.scalar('logcosh', tf.reduce_mean(tf.get_collection('LOSS_LOGCOSH')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('encoder', tf.reduce_mean(tf.get_collection('LOSS_ENCODER')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('generator', tf.reduce_mean(tf.get_collection('LOSS_GENERATOR')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_REAL')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_real_separate', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_REAL_SEP')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_gp', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_GP')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_REAL')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_real_separate', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_REAL_SEP')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_gp', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_GP')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_FAKE')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_FAKE')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', tf.reduce_mean(tf.get_collection('METRIC_PSNR')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', tf.reduce_mean(tf.get_collection('METRIC_SSIM')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        original_image_summary = tf.summary.image('original', tf.image.resize(tf.clip_by_value(tf.transpose(tf.cast(reals, tf.float32)/255.0, perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=8, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        recovered_image_summary = tf.summary.image('recovered', tf.image.resize(tf.clip_by_value(tf.transpose(tf.concat(tf.get_collection('IMAGE_ENCODED'), axis=0), perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=8, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
        val_summary = tf.summary.merge(tf.get_collection('VAL_SUMMARY'))
        full_summary = tf.summary.merge_all()

    os.makedirs(result_dir+'/model', exist_ok=True)
    os.makedirs(result_dir+'/summary', exist_ok=True)
    train_summary_writer = tf.summary.FileWriter(result_dir+'/summary/train')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:

        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set)
        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                encoder_optimizer.reset_optimizer_state(); generator_optimizer.reset_optimizer_state(); z_critic_optimizer.reset_optimizer_state(); y_critic_optimizer.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for repeat in range(minibatch_repeats):
            tfutil.run([encoder_optimize], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
            for _ in range(5):
                tfutil.run([z_critic_optimize, avg_encoder_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch

            tfutil.run([generator_optimize], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})
            for _ in range(5):
                tfutil.run([y_critic_optimize, avg_generator_update_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

        train_scalar_summary = sess.run(scalar_summary)
        train_summary_writer.add_summary(train_scalar_summary, cur_nimg)


        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (cur_tick, cur_nimg / 1000.0, sched.lod, sched.minibatch, misc.format_time(total_time), tick_time, tick_time / tick_kimg, maintenance_time))
            train_image_summary = sess.run(image_summary)
            train_summary_writer.add_summary(train_image_summary, cur_nimg)

            # Save snapshots.
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl((encoder, avg_encoder, generator, avg_generator, latent_critic, image_critic), os.path.join(result_dir, 'model', 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc.save_pkl((encoder, avg_encoder, generator, avg_generator, latent_critic, image_critic), os.path.join(result_dir, 'model', 'network-final.pkl'))

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    np.random.seed(1000)
    print('Initializing TensorFlow...')
    # os.environ.update(config.env)
    num_gpus = 1
    result_dir = 'results/progressive_test'

    tfutil.init_tf({'graph_options.place_pruned_graph': True})
    print('Running')
    coeff_lambda = dict(l2=1.0, l1=0.0, vgg=0.0, lpips=0.0, mssim=0.0, logcosh=0.0, gp=10)
    train_progressive_autoencoder()
    print('Exiting...')

#----------------------------------------------------------------------------
