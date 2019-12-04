import os
import sys
import utils
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'progan'))

from collections import OrderedDict
from tqdm import tqdm
from vgg import Vgg16
from losses import *
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

    # if args.progan:
    #     print('AUTOENCODING ON THE PROGRESSIVE GAN')
    #     url = os.path.join(args.cache_dir, 'karras2018iclr-celebahq-1024x1024.pkl')
    #     with open(url, 'rb') as f: _, _, Gs = pickle.load(f)
    # else:
    #     print('AUTOENCODING ON THE STYLE GAN')
    #     url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #     with dnnlib.util.open_url(url, cache_dir=args.cache_dir) as f: _, _, Gs = pickle.load(f)

    # LOAD DATASET
    print("LOADING FFHQ DATASET")
    ffhq = dataset.load_dataset(data_dir=args.data_dir, tfrecord_dir='ffhq', verbose=False)
    ffhq.configure(args.num_gpus*args.minibatch_size)
    get_images, get_labels = ffhq.get_minibatch_tf()
    get_images = tf.cast(get_images, tf.float32)/255.0
    get_images = get_images * 2.0 - 1.0

    # PREPARE VALIDATION IMAGE BATCH
    image_list = [image for image in os.listdir(args.validation_dir) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
    assert len(image_list)>0
    val_imbatch = np.transpose(np.stack([np.array(PIL.Image.open(args.validation_dir+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0), [0,3,1,2])/255.0
    val_imbatch = val_imbatch * 2.0 - 1.0

    # DEFINE INPUTS
    with tf.device('/cpu:0'):
        image_input = tf.placeholder(tf.float32, [None,3,1024,1024], name='image_input')
        gpu_image_input = tf.split(image_input, args.num_gpus, axis=0)
        encoder_learning_rate = tf.placeholder(tf.float32, [], name='encoder_learning_rate')
        generator_learning_rate = tf.placeholder(tf.float32, [], name='generator_learning_rate')
        # Gs_beta = 0.5 ** tf.div(tf.cast(args.minibatch_size*args.num_gpus, tf.float32), 10000.0)
        # tf.add_to_collection('KEY_NODES', image_input)
        # tf.add_to_collection('KEY_NODES', encoder_learning_rate)
        # tf.add_to_collection('KEY_NODES', generator_learning_rate)

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimizers'):
        encoder_optimizer = tflib.Optimizer(name='encoder_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        generator_optimizer = tflib.Optimizer(name='generator_optimizer', learning_rate=generator_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        z_critic_optimizer = tflib.Optimizer(name='z_critic_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        y_critic_optimizer = tflib.Optimizer(name='y_critic_optimizer', learning_rate=generator_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)

    # CONSTRUCT MODELS
    for gpu_idx in gpus:
        with tf.name_scope('GPU{}'.format(gpu_idx)), tf.device('/gpu:{}'.format(gpu_idx)):
            print("CONSTRUCTING MODEL WITH GPU: {}".format(gpu_idx))

            # DEFINE ENCODER AND GENERATOR
            # generator = Gs.clone(name='generator')
            # avg_generator = Gs.clone(name='avg_generator')
            encoders = []
            generators = []
            for lod in range(10, 1, -1):
                encoders.append(tflib.Network("encoder{}x{}".format(2**lod, 2**lod), func_name='encoder.E_basic', out_shape=[512], num_channels=3, resolution=2**lod))
            for lod in range(2, 11, 1):
                generators.append(tflib.Network("generator{}x{}".format(2**lod, 2**lod), func_name='stylegan.training.networks_progan.G_paper', latent_size=512, num_channels=3, resolution=2**lod))

            # CONSTRUCT NETWORK
            x = gpu_image_input[gpu_idx]
            encoder_features = []
            generator_features = []
            for encoder in encoders:
                x = encoder.get_output_for(x)
                encoder_features.append(x)
            encoded_latents = x
            for generator in generators:
                x = generator.get_output_for(x)
                generator_features.append(x)
            encoded_images = x
            generator_features = generator_features[::-1]
            # if gpu_idx==0:
            #     latent_manipulator = tf.placeholder_with_default(tf.zeros_like(encoded_latents), encoded_latents.shape, name='latent_manipulator')
            # encoded_images = generator.get_output_for(encoded_latents+latent_manipulator, None, is_validation=True, use_noise=False, randomize_noise=False)
            # tf.add_to_collection('KEY_NODES', latent_manipulator)
            # tf.add_to_collection('KEY_NODES', encoded_latents)
            # tf.add_to_collection('KEY_NODES', encoded_images)
            tf.add_to_collection('IMAGE_ENCODED', encoded_images)
            """ images is the X and Y domain,
                encoded_latents is the z domain,
                encoder is the mapping E,
                generator is the mapping D,
                encoded_images is the DEx domain """

            with tf.name_scope('loss'):
                MSE = tf.keras.losses.MeanSquaredError()
                MAE = tf.keras.losses.MeanAbsoluteError()

                with tf.name_scope('features_loss'):
                    feature_loss = 0.0

                    for e_feat, g_feat in zip(encoder_features, generator_features):
                        feature_loss += MSE(e_feat, g_feat)

                with tf.name_scope('regression_loss'):
                    regression_loss = tf.identity(feature_loss)
                    # regression_loss = 0.0

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

                with tf.name_scope('z_domain_loss'):
                    latent_critic = tflib.Network("z_critic", func_name='stylegan.training.networks_stylegan.G_mapping', dlatent_size=1, mapping_layers=args.latent_critic_layers, latent_size=512, normalize_latents=False)

                    z_critic_fake_loss = G_lsgan(G=encoder, D=latent_critic, opt=z_critic_optimizer, latents=images, labels=None)
                    z_critic_real_loss = D_lsgan(G=encoder, D=latent_critic, opt=z_critic_optimizer, latents=images, reals=tf.random_normal(shape=tf.shape(encoded_latents), name='z_real'))

                with tf.name_scope('y_domain_loss'):
                    image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_progan.D_paper', num_channels=3, resolution=1024, structure='fixed')

                    y_critic_fake_loss = G_wgan(G=generator, D=image_critic, opt=y_critic_optimizer, latent_shape=tf.shape(encoded_latents), labels=None)
                    y_critic_real_loss = D_wgan_gp(G=generator, D=image_critic, opt=y_critic_optimizer, latent_shape=tf.shape(encoded_latents), labels=None, reals=tf.identity(images, name='y_real'))

                with tf.name_scope('final_losses'):
                    encoder_loss = regression_loss + z_critic_fake_loss
                    generator_loss = regression_loss + y_critic_fake_loss
                    z_critic_loss = tf.identity(z_critic_real_loss)
                    y_critic_loss = tf.identity(y_critic_real_loss)
                    tf.add_to_collection("LOSS_REGRESSION", regression_loss)
                    tf.add_to_collection("LOSS_ENCODER", encoder_loss)
                    tf.add_to_collection("LOSS_GENERATOR", generator_loss)
                    tf.add_to_collection("LOSS_Z_CRITIC_REAL", z_critic_real_loss)
                    tf.add_to_collection("LOSS_Y_CRITIC_REAL", y_critic_real_loss)
                    tf.add_to_collection("LOSS_Z_CRITIC_FAKE", z_critic_fake_loss)
                    tf.add_to_collection("LOSS_Y_CRITIC_FAKE", y_critic_fake_loss)

                with tf.name_scope('metrics'):
                    psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    tf.add_to_collection('METRIC_PSNR', psnr)
                    tf.add_to_collection('METRIC_SSIM', ssim)

                with tf.name_scope('backprop'):
                    encoder_trainables = OrderedDict()
                    generator_trainables = OrderedDict()

                    for encoder in encoders:
                        encoder_trainables.update(encoder.trainables)

                    print("================== ENCODER VARS ==================")
                    print("\n".join([v.name for v in [*encoder_trainables.values()]]))
                    encoder_optimizer.register_gradients(encoder_loss, encoder_trainables)

                    print("================== GENERATOR VARS ==================")

                    for generator in generators:
                        generator_trainables.update(generator.trainables)
                    print("\n".join([v.name for v in [*generator_trainables.values()]]))
                    generator_optimizer.register_gradients(generator_loss, generator_trainables)

                    print("================== Z_CRITIC VARS ==================")
                    print("\n".join([v.name for v in [*latent_critic.trainables.values()]]))
                    z_critic_optimizer.register_gradients(z_critic_loss, latent_critic.trainables)

                    print("================== Y_CRITIC VARS ==================")
                    print("\n".join([v.name for v in [*image_critic.trainables.values()]]))
                    y_critic_optimizer.register_gradients(y_critic_loss, image_critic.trainables)

    with tf.name_scope('summary'):
        _ = tf.summary.scalar('regression', tf.reduce_mean(tf.get_collection('LOSS_REGRESSION')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.l2_lambda > 0.0: _ = tf.summary.scalar('l2', tf.reduce_mean(tf.get_collection('LOSS_L2')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.l1_lambda > 0.0: _ = tf.summary.scalar('l1', tf.reduce_mean(tf.get_collection('LOSS_L1')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.vgg_lambda > 0.0: _ = tf.summary.scalar('vgg', tf.reduce_mean(tf.get_collection('LOSS_VGG')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.lpips_lambda > 0.0: _ = tf.summary.scalar('lpips', tf.reduce_mean(tf.get_collection('LOSS_LPIPS')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.mssim_lambda > 0.0: _ = tf.summary.scalar('mssim', tf.reduce_mean(tf.get_collection('LOSS_MSSIM')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.logcosh_lambda > 0.0: _ = tf.summary.scalar('logcosh', tf.reduce_mean(tf.get_collection('LOSS_LOGCOSH')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('encoder', tf.reduce_mean(tf.get_collection('LOSS_ENCODER')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('generator', tf.reduce_mean(tf.get_collection('LOSS_GENERATOR')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_REAL')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_REAL')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_FAKE')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_FAKE')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', tf.reduce_mean(tf.get_collection('METRIC_PSNR')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', tf.reduce_mean(tf.get_collection('METRIC_SSIM')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('encoder', encoder_learning_rate, family='03_lr', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('generator', generator_learning_rate, family='03_lr', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        original_image_summary = tf.summary.image('original', tf.image.resize(tf.clip_by_value(tf.transpose((image_input + 1.0) / 2.0, perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        recovered_image_summary = tf.summary.image('recovered', tf.image.resize(tf.clip_by_value(tf.transpose((tf.concat(tf.get_collection('IMAGE_ENCODED'), axis=0) + 1.0) / 2.0, perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
        val_summary = tf.summary.merge(tf.get_collection('VAL_SUMMARY'))
        full_summary = tf.summary.merge_all()

    # DEFINE OPTIMIZE OPS
    with tf.name_scope('optimize'):
        encoder_optimize = encoder_optimizer.apply_updates()
        generator_optimize = generator_optimizer.apply_updates()
        # generator_optimize = avg_generator.setup_as_moving_average_of(generator, beta=Gs_beta)

        z_critic_optimize = z_critic_optimizer.apply_updates()
        y_critic_optimize = y_critic_optimizer.apply_updates()

    os.makedirs(args.result_dir+'/model', exist_ok=True)
    os.makedirs(args.result_dir+'/summary', exist_ok=True)
    train_summary_writer = tf.summary.FileWriter(args.result_dir+'/summary/train')
    val_summary_writer = tf.summary.FileWriter(args.result_dir+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    encoder_lr = args.encoder_learning_rate
    generator_lr = args.generator_learning_rate
    for iter in tqdm(range(args.num_iter)):
        train_imbatch = sess.run(get_images)
        for _ in range(args.encoder_iter):
            _ = sess.run(encoder_optimize, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr})
            for _ in range(args.latent_critic_iter):
                _ = sess.run(z_critic_optimize, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr})

        for _ in range(args.generator_iter):
            _ = sess.run(generator_optimize, feed_dict={image_input: train_imbatch, generator_learning_rate: generator_lr})
            for _ in range(args.image_critic_iter):
                _ = sess.run(y_critic_optimize, feed_dict={image_input: train_imbatch, generator_learning_rate: generator_lr})

        train_scalar_summary = sess.run(scalar_summary, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr, generator_learning_rate: generator_lr})
        train_summary_writer.add_summary(train_scalar_summary, iter)
        val_scalar_summary = sess.run(val_summary, feed_dict={image_input: val_imbatch, encoder_learning_rate: encoder_lr, generator_learning_rate: generator_lr})
        val_summary_writer.add_summary(val_scalar_summary, iter)

        if iter%args.save_iter==0:
            train_image_summary = sess.run(image_summary, feed_dict={image_input: train_imbatch})
            train_summary_writer.add_summary(train_image_summary, iter)
            val_image_summary = sess.run(recovered_image_summary, feed_dict={image_input: val_imbatch})
            val_summary_writer.add_summary(val_image_summary, iter)
            if iter==0:
                val_original_image_summary = sess.run(original_image_summary, feed_dict={image_input: val_imbatch})
                val_summary_writer.add_summary(val_original_image_summary, iter)

            save_pkl((encoder, generator, latent_critic, image_critic), args.result_dir+'/model/model.pkl')



if __name__=='__main__':
    main()
