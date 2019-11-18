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

    # LOAD DATASET
    print("LOADING FFHQ DATASET")
    ffhq = dataset.load_dataset(data_dir=args.data_dir, tfrecord_dir='ffhq', verbose=False)
    ffhq.configure(args.num_gpus*args.minibatch_size)
    get_images, get_labels = ffhq.get_minibatch_tf()
    get_images = tf.cast(get_images, tf.float32)/255.0

    # PREPARE VALIDATION IMAGE BATCH
    image_list = [image for image in os.listdir(args.validation_dir) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
    assert len(image_list)>0
    val_imbatch = np.transpose(np.stack([np.array(PIL.Image.open(args.validation_dir+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0), [0,3,1,2])/255.0

    # DEFINE INPUTS
    with tf.device('/cpu:0'):
        image_input = tf.placeholder(tf.float32, [None,3,1024,1024], name='image_input')
        gpu_image_input = tf.split(image_input, args.num_gpus, axis=0)
        encoder_learning_rate = tf.placeholder(tf.float32, [], name='encoder_learning_rate')
        # generator_learning_rate = tf.placeholder(tf.float32, [], name='generator_learning_rate')
        # Gs_beta = 0.5 ** tf.div(tf.cast(args.minibatch_size*args.num_gpus, tf.float32), 10000.0)
        tf.add_to_collection('KEY_NODES', image_input)
        tf.add_to_collection('KEY_NODES', encoder_learning_rate)
        # tf.add_to_collection('KEY_NODES', generator_learning_rate)

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimizers'):
        # encoder_optimizer = tflib.Optimizer(name='encoder_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        # generator_optimizer = tflib.Optimizer(name='generator_optimizer', learning_rate=generator_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        # z_critic_optimizer = tflib.Optimizer(name='z_critic_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        # y_critic_optimizer = tflib.Optimizer(name='y_critic_optimizer', learning_rate=generator_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        autoencoder_optimizer = tflib.Optimizer(name='autoencoder_optimizer', learning_rate=encoder_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)

    # CONSTRUCT MODELS
    for gpu_idx in gpus:
        with tf.name_scope('GPU{}'.format(gpu_idx)), tf.device('/gpu:{}'.format(gpu_idx)):
            print("CONSTRUCTING MODEL WITH GPU: {}".format(gpu_idx))

            # DEFINE ENCODER AND GENERATOR
            if bool(args.blur_filter): blur = [1,2,1]
            else: blur=None
            encoder = tflib.Network("encoder", func_name='autoencoder.E_basic', out_shape=[512], nonlinearity=args.nonlinearity, use_wscale=args.use_wscale, mbstd_group_size=args.mbstd_group_size, mbstd_num_features=args.mbstd_num_features, fused_scale=args.fused_scale, blur_filter=blur)
            decoder = tflib.Network("decoder", func_name='autoencoder.D_basic', num_channels=3, resolution=1024, structure='linear')

            # CONSTRUCT NETWORK
            images = gpu_image_input[gpu_idx]
            encoded_latents = encoder.get_output_for(images)
            encoded_images = decoder.get_output_for(encoded_latents[0],encoded_latents[1],encoded_latents[2],encoded_latents[3],encoded_latents[4],encoded_latents[5],encoded_latents[6],encoded_latents[7],encoded_latents[8],encoded_latents[9], is_validation=True)
            tf.add_to_collection('KEY_NODES', encoded_latents)
            tf.add_to_collection('KEY_NODES', encoded_images)
            tf.add_to_collection('IMAGE_ENCODED', encoded_images)

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

                with tf.name_scope('final_losses'):
                    autoencoder_loss = regression_loss
                    tf.add_to_collection("LOSS_AUTOENCODER", autoencoder_loss)

                with tf.name_scope('metrics'):
                    psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    tf.add_to_collection('METRIC_PSNR', psnr)
                    tf.add_to_collection('METRIC_SSIM', ssim)

                with tf.name_scope('backprop'):
                    print("================== ENCODER VARS ==================")
                    encoder_trainables = [*encoder.trainables.values()]
                    print("\n".join([v.name for v in encoder_trainables]))

                    print("================== DECODER VARS ==================")
                    decoder_trainables = [*decoder.trainables.values()]
                    print("\n".join([v.name for v in decoder_trainables]))

                    autoencoder_optimizer.register_gradients(autoencoder_loss, encoder_trainables+decoder_trainables)

    with tf.name_scope('summary'):
        if args.l2_lambda > 0.0: _ = tf.summary.scalar('l2', tf.reduce_mean(tf.get_collection('LOSS_L2')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.l1_lambda > 0.0: _ = tf.summary.scalar('l1', tf.reduce_mean(tf.get_collection('LOSS_L1')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.vgg_lambda > 0.0: _ = tf.summary.scalar('vgg', tf.reduce_mean(tf.get_collection('LOSS_VGG')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.lpips_lambda > 0.0: _ = tf.summary.scalar('lpips', tf.reduce_mean(tf.get_collection('LOSS_LPIPS')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.mssim_lambda > 0.0: _ = tf.summary.scalar('mssim', tf.reduce_mean(tf.get_collection('LOSS_MSSIM')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        if args.logcosh_lambda > 0.0: _ = tf.summary.scalar('logcosh', tf.reduce_mean(tf.get_collection('LOSS_LOGCOSH')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('autoencoder', tf.reduce_mean(tf.get_collection('LOSS_AUTOENCODER')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', tf.reduce_mean(tf.get_collection('METRIC_PSNR')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', tf.reduce_mean(tf.get_collection('METRIC_SSIM')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('autoencoder', encoder_learning_rate, family='03_lr', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        original_image_summary = tf.summary.image('original', tf.image.resize(tf.clip_by_value(tf.transpose(image_input, perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        recovered_image_summary = tf.summary.image('recovered', tf.image.resize(tf.clip_by_value(tf.transpose(tf.concat(tf.get_collection('IMAGE_ENCODED'), axis=0), perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
        val_summary = tf.summary.merge(tf.get_collection('VAL_SUMMARY'))
        full_summary = tf.summary.merge_all()

    # DEFINE OPTIMIZE OPS
    with tf.name_scope('optimize'):
        autoencoder_optimize = autoencoder_optimizer.apply_updates()
        # encoder_optimize = encoder_optimizer.apply_updates()
        # generator_optimize = generator_optimizer.apply_updates()
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
    # generator_lr = args.generator_learning_rate
    for iter in tqdm(range(args.num_iter)):
        train_imbatch = sess.run(get_images)
        _ = sess.run(autoencoder_optimize, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr})

        train_scalar_summary = sess.run(scalar_summary, feed_dict={image_input: train_imbatch, encoder_learning_rate: encoder_lr})
        train_summary_writer.add_summary(train_scalar_summary, iter)
        val_scalar_summary = sess.run(val_summary, feed_dict={image_input: val_imbatch, encoder_learning_rate: encoder_lr})
        val_summary_writer.add_summary(val_scalar_summary, iter)

        if iter%args.save_iter==0:
            train_image_summary = sess.run(image_summary, feed_dict={image_input: train_imbatch})
            train_summary_writer.add_summary(train_image_summary, iter)
            val_image_summary = sess.run(recovered_image_summary, feed_dict={image_input: val_imbatch})
            val_summary_writer.add_summary(val_image_summary, iter)
            if iter==0:
                val_original_image_summary = sess.run(original_image_summary, feed_dict={image_input: val_imbatch})
                val_summary_writer.add_summary(val_original_image_summary, iter)

            save_pkl((encoder, decoder), args.result_dir+'/model/model.pkl')



if __name__=='__main__':
    main()
