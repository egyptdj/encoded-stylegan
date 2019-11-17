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
    args = utils.option.parse()
    tf.random.set_random_seed(args.seed)
    tf.config.set_soft_device_placement(True)

    tflib.init_tf()
    gpus = np.arange(args.num_gpus)

    # LOAD LSUN DININGROOM DATASET
    print("LOADING CELEBA DATASET")
    import tensorflow_datasets as tfds

    def resize(height=args.resolution, width=args.resolution):
        def transformation_func(x):
            return tf.image.resize_with_crop_or_pad(tf.image.resize(x['image'], [157,128]), height, width)
        return transformation_func

    dataset_builder = tfds.builder("celeb_a")
    dataset_builder.download_and_prepare(download_dir=str(args.data_dir))
    dataset_train = dataset_builder.as_dataset(split="train")
    dataset_train = dataset_train.map(resize())
    dataset_train = dataset_train.repeat().shuffle(buffer_size=1024, seed=args.seed).batch(args.minibatch_size)
    train_iterator = dataset_train.make_one_shot_iterator()
    get_train_image = tf.transpose(tf.cast(train_iterator.get_next(), tf.float32), perm=[0,3,1,2])/255.0

    dataset_val = dataset_builder.as_dataset(split="validation")
    dataset_val = dataset_val.map(resize())
    dataset_val = dataset_val.repeat().batch(args.minibatch_size)
    val_iterator = dataset_val.make_one_shot_iterator()
    get_val_image = tf.transpose(tf.cast(val_iterator.get_next(), tf.float32), perm=[0,3,1,2])/255.0

    # DEFINE INPUTS
    with tf.device('/cpu:0'):
        image_input = tf.placeholder(tf.float32, [None,3,args.resolution,args.resolution], name='image_input')
        gpu_image_input = tf.split(image_input, args.num_gpus, axis=0)
        encoder_learning_rate = tf.placeholder(tf.float32, [], name='encoder_learning_rate')
        generator_learning_rate = tf.placeholder(tf.float32, [], name='generator_learning_rate')
        # Gs_beta = 0.5 ** tf.div(tf.cast(args.minibatch_size*args.num_gpus, tf.float32), 10000.0)
        tf.add_to_collection('KEY_NODES', image_input)
        tf.add_to_collection('KEY_NODES', encoder_learning_rate)
        tf.add_to_collection('KEY_NODES', generator_learning_rate)

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimizers'):
        # encoder_optimizer = tflib.Optimizer(name='encoder_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        # generator_optimizer = tflib.Optimizer(name='generator_optimizer', learning_rate=generator_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        autoencoder_optimizer = tflib.Optimizer(name='autoencoder_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        # z_critic_optimizer = tflib.Optimizer(name='z_critic_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        # y_critic_optimizer = tflib.Optimizer(name='y_critic_optimizer', learning_rate=generator_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)

    # CONSTRUCT MODELS
    for gpu_idx in gpus:
        with tf.name_scope('GPU{}'.format(gpu_idx)), tf.device('/gpu:{}'.format(gpu_idx)):
            print("CONSTRUCTING MODEL WITH GPU: {}".format(gpu_idx))

            # DEFINE ENCODER AND GENERATOR
            if bool(args.blur_filter): blur = [1,2,1]
            else: blur=None

            autoencoder = tflib.Network("autoencoder", out_shape=[512], func_name='autoencoder.AE_basic', nonlinearity=args.nonlinearity, use_wscale=args.use_wscale, mbstd_group_size=args.mbstd_group_size, mbstd_num_features=args.mbstd_num_features, fused_scale=args.fused_scale, blur_filter=blur, feature_collection='FEATURES')
            # encoder = tflib.Network("encoder", out_shape=[512], func_name='encoder.E_basic', nonlinearity=args.nonlinearity, use_wscale=args.use_wscale, mbstd_group_size=args.mbstd_group_size, mbstd_num_features=args.mbstd_num_features, fused_scale=args.fused_scale, blur_filter=blur, feature_collection='FEATURES')
            # generator = tflib.Network("generator", func_name='stylegan.training.networks_progan.G_paper', num_channels=3, resolution=args.resolution, structure='linear', feature_collection='FEATURES')

            # CONSTRUCT NETWORK
            images = gpu_image_input[gpu_idx]
            # encoded_latents = encoder.get_output_for(images)
            # if gpu_idx==0:
            #     latent_manipulator = tf.placeholder_with_default(tf.zeros_like(encoded_latents), encoded_latents.shape, name='latent_manipulator')
            # encoded_images = generator.get_output_for(encoded_latents+latent_manipulator, None, is_validation=True, use_noise=False, randomize_noise=False)
            encoded_images = autoencoder.get_output_for(images, None, is_validation=True, use_noise=False, randomize_noise=False)
            # tf.add_to_collection('KEY_NODES', latent_manipulator)
            # tf.add_to_collection('KEY_NODES', encoded_latents)
            tf.add_to_collection('KEY_NODES', encoded_images)
            tf.add_to_collection('IMAGE_ENCODED', encoded_images)
            """ images is the X and Y domain,
                encoded_latents is the z domain,
                encoder is the mapping E,
                generator is the mapping D,
                encoded_images is the DEx domain """

            with tf.name_scope('loss'):
                MSE = tf.keras.losses.MeanSquaredError()
                MAE = tf.keras.losses.MeanAbsoluteError()

                with tf.name_scope('regression_loss'):
                    regression_loss = 0.0

                    # L2 Loss
                    if args.l2_lambda > 0.0:
                        l2_loss = MSE(images, encoded_images)
                        tf.add_to_collection('LOSS_L2', l2_loss)
                        regression_loss += args.l2_lambda*l2_loss

                    # L1 Loss
                    if args.l1_lambda > 0.0:
                        l1_loss = MAE(images, encoded_images)
                        tf.add_to_collection('LOSS_L1', l1_loss)
                        regression_loss += args.l1_lambda*l1_loss

                    # VGG loss
                    if args.vgg_lambda > 0.0:
                        image_vgg = Vgg16(args.cache_dir+'/vgg16.npy')
                        image_vgg.build(tf.image.resize(tf.transpose(images, perm=[0,2,3,1]), [args.vgg_shape,args.vgg_shape]))
                        image_perception = [image_vgg.conv1_1, image_vgg.conv1_2, image_vgg.conv3_2, image_vgg.conv4_2]
                        encoded_vgg = Vgg16(args.cache_dir+'/vgg16.npy')
                        encoded_vgg.build(tf.image.resize(tf.transpose(encoded_images, perm=[0,2,3,1]), [args.vgg_shape,args.vgg_shape]))
                        encoded_perception = [encoded_vgg.conv1_1, encoded_vgg.conv1_2, encoded_vgg.conv3_2, encoded_vgg.conv4_2]
                        vgg_loss = tf.reduce_sum([MSE(image, encoded) for image, encoded in zip(image_perception, encoded_perception)]) # https://github.com/machrisaa/tensorflow-vgg
                        tf.add_to_collection('LOSS_VGG', vgg_loss)
                        regression_loss += args.vgg_lambda*vgg_loss

                    # LPIPS loss
                    if args.lpips_lambda > 0.0:
                        lpips_url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'
                        with dnnlib.util.open_url(lpips_url, cache_dir=args.cache_dir) as lpips:
                            lpips_network =  pickle.load(lpips)
                        lpips_loss = lpips_network.get_output_for(images, encoded_images)
                        tf.add_to_collection('LOSS_LPIPS', lpips_loss)
                        regression_loss += args.lpips_lambda*lpips_loss

                    # MSSIM loss
                    if args.mssim_lambda > 0.0:
                        mssim_loss = 1-tf.image.ssim_multiscale(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0)
                        tf.add_to_collection('LOSS_MSSIM', mssim_loss)
                        regression_loss += args.mssim_lambda*mssim_loss

                    # LOGCOSH loss
                    if args.logcosh_lambda > 0.0:
                        logcosh_loss = tf.keras.losses.logcosh(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]))
                        tf.add_to_collection('LOSS_LOGCOSH', logcosh_loss)
                        regression_loss += args.logcosh_lambda*logcosh_loss


                # with tf.name_scope('z_domain_loss'):
                #     latent_critic = tflib.Network("z_critic", func_name='stylegan.training.networks_stylegan.G_mapping', dlatent_size=1, mapping_layers=args.latent_critic_layers, latent_size=512, normalize_latents=False)
                #     fake_latent = tf.random.normal(shape=tf.shape(encoded_latents), name='z_rand')
                #     real_latent = tf.identity(encoded_latents, name='z_real')
                #
                #     fake_latent_critic_out = latent_critic.get_output_for(tf.reshape(fake_latent, [-1,512]), None)
                #     real_latent_critic_out = latent_critic.get_output_for(tf.reshape(real_latent, [-1,512]), None)
                #
                #     with tf.name_scope("fake_loss"):
                #         fake_latent_loss = tf.losses.compute_weighted_loss(\
                #             losses=fake_latent_critic_out, \
                #             weights=1.0, scope='fake_latent_loss')
                #
                #     with tf.name_scope("real_loss"):
                #         real_latent_critic_loss = tf.losses.compute_weighted_loss(\
                #             losses=real_latent_critic_out, \
                #             weights=1.0, scope='real_latent_critic_loss')
                #
                #         fake_latent_critic_loss = tf.losses.compute_weighted_loss(\
                #             losses=fake_latent_critic_out, \
                #             weights=1.0, scope='fake_latent_critic_loss')
                #
                #         # WASSERSTEIN GAN - GRADIENT PENALTY
                #         with tf.name_scope('gradient_penalty'):
                #             epsilon = tf.random.uniform([], name='epsilon')
                #             gradient_latent = tf.identity((epsilon * real_latent + (1-epsilon) * fake_latent), name='gradient_latent')
                #             critic_gradient_out = latent_critic.get_output_for(tf.reshape(gradient_latent, [-1,512]), None)
                #             gradients = tf.gradients(critic_gradient_out, gradient_latent, name='gradients')
                #             gradients_norm = tf.norm(gradients[0], ord=2, name='gradient_norm')
                #             gradient_penalty = tf.square(gradients_norm -1)
                #
                #     z_critic_real_loss = -real_latent_critic_loss + fake_latent_critic_loss + gradient_penalty
                #     z_critic_fake_loss = -fake_latent_loss
                #
                # with tf.name_scope('y_domain_loss'):
                #     if args.progan:
                #         image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_progan.D_paper', num_channels=3, resolution=args.resolution, structure=None)
                #     else:
                #         image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_stylegan.D_basic', num_channels=3, resolution=args.resolution, structure=None)
                #
                #     fake_image = generator.get_output_for(tf.random.normal(shape=tf.shape(encoded_latents), name='z_rand'), None, is_validation=True, use_noise=False, randomize_noise=False)
                #     real_image = tf.identity(images, name='y_real')
                #     tf.add_to_collection('IMAGE_GENERATED', fake_image)
                #
                #     fake_image_critic_out = image_critic.get_output_for(fake_image, None)
                #     real_image_critic_out = image_critic.get_output_for(real_image, None)
                #     with tf.name_scope("fake_loss"):
                #         fake_image_loss = tf.losses.compute_weighted_loss(\
                #             losses=fake_image_critic_out, \
                #             weights=1.0, scope='fake_image_loss')
                #
                #     with tf.name_scope("real_loss"):
                #         real_image_critic_loss = tf.losses.compute_weighted_loss(\
                #             losses=real_image_critic_out, \
                #             weights=1.0, scope='real_image_critic_loss')
                #
                #         fake_image_critic_loss = tf.losses.compute_weighted_loss(\
                #             losses=fake_image_critic_out, \
                #             weights=1.0, scope='fake_image_critic_loss')
                #
                #         # WASSERSTEIN GAN - GRADIENT PENALTY
                #         with tf.name_scope('gradient_penalty'):
                #             epsilon = tf.random.uniform([], name='epsilon')
                #             gradient_image = tf.identity((epsilon * real_image + (1-epsilon) * fake_image), name='gradient_image')
                #             critic_gradient_out = image_critic.get_output_for(gradient_image, None)
                #             gradients = tf.gradients(critic_gradient_out, gradient_image, name='gradients')
                #             gradients_norm = tf.norm(gradients[0], ord=2, name='gradient_norm')
                #             gradient_penalty = tf.square(gradients_norm -1)
                #
                #     y_critic_real_loss = -real_image_critic_loss + fake_image_critic_loss + gradient_penalty
                #     y_critic_fake_loss = -fake_image_loss

                with tf.name_scope('final_losses'):
                    # encoder_loss = regression_loss + z_critic_fake_loss
                    # generator_loss = regression_loss + y_critic_fake_loss
                    # autoencoder_loss = regression_loss + z_critic_fake_loss + y_critic_fake_loss
                    # z_critic_loss = tf.identity(z_critic_real_loss)
                    # y_critic_loss = tf.identity(y_critic_real_loss)
                    tf.add_to_collection("LOSS_REGRESSION", regression_loss)
                    # tf.add_to_collection("LOSS_ENCODER", encoder_loss)
                    # tf.add_to_collection("LOSS_GENERATOR", generator_loss)
                    # tf.add_to_collection("LOSS_AUTOENCODER", autoencoder_loss)
                    # tf.add_to_collection("LOSS_Z_CRITIC_REAL", z_critic_real_loss)
                    # tf.add_to_collection("LOSS_Y_CRITIC_REAL", y_critic_real_loss)
                    # tf.add_to_collection("LOSS_Z_CRITIC_FAKE", z_critic_fake_loss)
                    # tf.add_to_collection("LOSS_Y_CRITIC_FAKE", y_critic_fake_loss)

                with tf.name_scope('metrics'):
                    psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    tf.add_to_collection('METRIC_PSNR', psnr)
                    tf.add_to_collection('METRIC_SSIM', ssim)

                with tf.name_scope('backprop'):
                    # print("================== ENCODER VARS ==================")
                    # print("\n".join([v.name for v in [*encoder.trainables.values()]]))
                    # encoder_optimizer.register_gradients(encoder_loss, encoder.trainables)
                    #
                    # print("================== GENERATOR VARS ==================")
                    # print("\n".join([v.name for v in [*generator.trainables.values()]]))
                    # generator_optimizer.register_gradients(generator_loss, generator.trainables)

                    print("================== AUTOENCODER VARS ==================")
                    print("\n".join([v.name for v in [*autoencoder.trainables.values()]]))
                    print("================== AUTOENCODER LAYERSS ==================")
                    print(autoencoder.print_layers())
                    autoencoder_optimizer.register_gradients(regression_loss, autoencoder.trainables)

                    # print("================== Z_CRITIC VARS ==================")
                    # print("\n".join([v.name for v in [*latent_critic.trainables.values()]]))
                    # z_critic_optimizer.register_gradients(z_critic_loss, latent_critic.trainables)
                    #
                    # print("================== Y_CRITIC VARS ==================")
                    # print("\n".join([v.name for v in [*image_critic.trainables.values()]]))
                    # y_critic_optimizer.register_gradients(y_critic_loss, image_critic.trainables)

    with tf.name_scope('summary'):
        _ = tf.summary.scalar('regression', tf.reduce_mean(tf.get_collection('LOSS_REGRESSION')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.l2_lambda > 0.0: _ = tf.summary.scalar('l2', tf.reduce_mean(tf.get_collection('LOSS_L2')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.l1_lambda > 0.0: _ = tf.summary.scalar('l1', tf.reduce_mean(tf.get_collection('LOSS_L1')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.vgg_lambda > 0.0: _ = tf.summary.scalar('vgg', tf.reduce_mean(tf.get_collection('LOSS_VGG')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.lpips_lambda > 0.0: _ = tf.summary.scalar('lpips', tf.reduce_mean(tf.get_collection('LOSS_LPIPS')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.mssim_lambda > 0.0: _ = tf.summary.scalar('mssim', tf.reduce_mean(tf.get_collection('LOSS_MSSIM')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.logcosh_lambda > 0.0: _ = tf.summary.scalar('logcosh', tf.reduce_mean(tf.get_collection('LOSS_LOGCOSH')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('encoder', tf.reduce_mean(tf.get_collection('LOSS_ENCODER')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('generator', tf.reduce_mean(tf.get_collection('LOSS_GENERATOR')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('autoencoder', tf.reduce_mean(tf.get_collection('LOSS_AUTOENCODER')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('latent_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_REAL')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('image_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_REAL')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('latent_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_FAKE')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        # _ = tf.summary.scalar('image_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_FAKE')), family='loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', tf.reduce_mean(tf.get_collection('METRIC_PSNR')), family='metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', tf.reduce_mean(tf.get_collection('METRIC_SSIM')), family='metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('encoder', encoder_learning_rate, family='lr', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('generator', generator_learning_rate, family='lr', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        original_image_summary = tf.summary.image('original', tf.clip_by_value(tf.transpose(image_input, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        recovered_image_summary = tf.summary.image('recovered', tf.clip_by_value(tf.transpose(tf.concat(tf.get_collection('IMAGE_ENCODED'), axis=0), perm=[0,2,3,1]), 0.0, 1.0), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        # generated_image_summary = tf.summary.image('generated', tf.clip_by_value(tf.transpose(tf.concat(tf.get_collection('IMAGE_GENERATED'), axis=0), perm=[0,2,3,1]), 0.0, 1.0), max_outputs=args.image_output, family='images', collections=[tf.GraphKeys.SUMMARIES])
        scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
        val_summary = tf.summary.merge(tf.get_collection('VAL_SUMMARY'))
        full_summary = tf.summary.merge_all()

    # DEFINE OPTIMIZE OPS
    with tf.name_scope('optimize'):
        # encoder_optimize = encoder_optimizer.apply_updates()
        # generator_optimize = generator_optimizer.apply_updates()
        autoencoder_optimize = autoencoder_optimizer.apply_updates()
        # generator_optimize = avg_generator.setup_as_moving_average_of(generator, beta=Gs_beta)

        # z_critic_optimize = z_critic_optimizer.apply_updates()
        # y_critic_optimize = y_critic_optimizer.apply_updates()

    os.makedirs(args.result_dir+'/model', exist_ok=True)
    os.makedirs(args.result_dir+'/summary', exist_ok=True)
    train_summary_writer = tf.summary.FileWriter(args.result_dir+'/summary/train')
    val_summary_writer = tf.summary.FileWriter(args.result_dir+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    encoder_lr = args.encoder_learning_rate
    generator_lr = args.generator_learning_rate
    for iter in tqdm(range(args.num_iter)):
        train_imbatch = sess.run(get_train_image)
        val_imbatch = sess.run(get_val_image)
        _ = sess.run(autoencoder_optimize, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr})
        # for _ in range(args.critic_iter):
        #     _ = sess.run(z_critic_optimize, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr})
        #     _ = sess.run(y_critic_optimize, feed_dict={image_input: train_imbatch, generator_learning_rate: generator_lr})

        train_scalar_summary = sess.run(scalar_summary, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr, generator_learning_rate: generator_lr})
        train_summary_writer.add_summary(train_scalar_summary, iter)
        val_scalar_summary = sess.run(val_summary, feed_dict={image_input: val_imbatch, encoder_learning_rate: encoder_lr, generator_learning_rate: generator_lr})
        val_summary_writer.add_summary(val_scalar_summary, iter)

        if iter%args.save_iter==0:
            train_image_summary = sess.run(image_summary, feed_dict={image_input: train_imbatch})
            train_summary_writer.add_summary(train_image_summary, iter)
            val_image_summary = sess.run(image_summary, feed_dict={image_input: val_imbatch})
            val_summary_writer.add_summary(val_image_summary, iter)
            # gen_summary = sess.run(generated_image_summary)
            # train_summary_writer.add_summary(gen_summary, iter)

            # save_pkl((encoder, generator, latent_critic, image_critic), args.result_dir+'/model/model.pkl')
            save_pkl(autoencoder, args.result_dir+'/model/model.pkl')



if __name__=='__main__':
    main()
