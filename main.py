import os
import sys
import utils
import pickle
import random
import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'progan'))
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    tf.config.set_soft_device_placement(True)

    tflib.init_tf()
    gpus = np.arange(args.num_gpus)

    if not args.stylegan:
        print('AUTOENCODING ON THE PROGRESSIVE GAN')
        url = os.path.join(args.cache_dir, 'karras2018iclr-celebahq-1024x1024.pkl')
        with open(url, 'rb') as f: _, _, Gs = pickle.load(f)
    else:
        print('AUTOENCODING ON THE STYLE GAN')
        url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        with dnnlib.util.open_url(url, cache_dir=args.cache_dir) as f: _, _, Gs = pickle.load(f)

    # LOAD DATASET
    print("LOADING FFHQ DATASET")
    ffhq = dataset.load_dataset(data_dir=args.data_dir, tfrecord_dir='ffhq', verbose=False)
    ffhq.configure(args.num_gpus*args.minibatch_size)
    get_images, get_labels = ffhq.get_minibatch_tf()
    get_images = tf.cast(get_images, tf.float32)/255.0
    empty_label = tf.placeholder(tf.float32, shape=[None,0], name='empty_label')
    train_labelbatch = np.zeros([args.minibatch_size,0], np.float32)

    # PREPARE VALIDATION BATCH
    image_list = [image for image in os.listdir(args.validation_dir) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
    assert len(image_list)>0
    val_imbatch = np.transpose(np.stack([np.float32(PIL.Image.open(args.validation_dir+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0), [0,3,1,2])/255.0
    val_labelbatch = np.zeros([len(image_list)//args.num_gpus,0], np.float32)
    val_latentbatch = np.random.normal(size=[val_imbatch.shape[0], 512])

    # DEFINE INPUTS
    with tf.device('/cpu:0'):
        image_input = tf.placeholder(tf.float32, [None,3,1024,1024], name='image_input')
        gpu_image_input = tf.split(image_input, args.num_gpus, axis=0)
        latent_input = tf.placeholder(tf.float32, [None,512], name='latent_input')
        gpu_latent_input = tf.split(latent_input, args.num_gpus, axis=0)
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        # Gs_beta = 0.5 ** tf.div(tf.cast(args.minibatch_size*args.num_gpus, tf.float32), 10000.0)
        tf.add_to_collection('KEY_NODES', image_input)
        tf.add_to_collection('KEY_NODES', latent_input)
        tf.add_to_collection('KEY_NODES', empty_label)
        tf.add_to_collection('KEY_NODES', learning_rate)

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimizers'):
        encoder_generator_optimizer = tflib.Optimizer(name='encoder_generator_optimizer', learning_rate=learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        critic_optimizer = tflib.Optimizer(name='critic_optimizer', learning_rate=learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)

    # CONSTRUCT MODELS
    for gpu_idx in gpus:
        with tf.name_scope('GPU{}'.format(gpu_idx)), tf.device('/gpu:{}'.format(gpu_idx)):
            print("CONSTRUCTING MODEL WITH GPU: {}".format(gpu_idx))

            # DEFINE ENCODER AND GENERATOR
            generator = Gs.clone(name='generator')
            # avg_generator = Gs.clone(name='avg_generator')
            if not args.stylegan:
                encoder = tflib.Network("encoder", func_name='encoder.E_basic', out_shape=[512], num_channels=3, resolution=1024, structure=args.structure)
            else:
                encoder = tflib.Network("encoder", func_name='encoder.E_basic', out_shape=[18, 512], num_channels=3, resolution=1024, structure=args.structure)

            # CONSTRUCT ILI NETWORK
            images = gpu_image_input[gpu_idx]
            encoded_latents = encoder.get_output_for(images)
            if gpu_idx==0:
                latent_manipulator = tf.placeholder_with_default(tf.zeros_like(encoded_latents), encoded_latents.shape, name='latent_manipulator')
            encoded_images = generator.get_output_for(encoded_latents+latent_manipulator, empty_label, is_validation=True, use_noise=False, randomize_noise=False)

            # CONSTRUCT LIL NETWORK
            latents = gpu_latent_input[gpu_idx]
            generated_images = generator.get_output_for(latents, empty_label, is_validation=True, use_noise=False, random_noise=False)
            generated_latents = encoder.get_output_for(generated_images)

            tf.add_to_collection('KEY_NODES', latent_manipulator)
            tf.add_to_collection('KEY_NODES', encoded_latents)
            tf.add_to_collection('KEY_NODES', encoded_images)
            tf.add_to_collection('LATENT_ENCODED', encoded_latents)
            tf.add_to_collection('IMAGE_ENCODED', encoded_images)
            tf.add_to_collection('IMAGE_GENERATED', generated_images)
            tf.add_to_collection('LATENT_GENERATED', generated_latents)

            with tf.name_scope('losses'):
                MSE = tf.keras.losses.MeanSquaredError()
                MAE = tf.keras.losses.MeanAbsoluteError()

                with tf.name_scope('ILI'):
                    with tf.name_scope('gan_losses'):
                        latent_critic = tflib.Network("z_critic", func_name='stylegan.training.networks_stylegan.G_mapping', dlatent_size=1, mapping_layers=args.latent_critic_layers, latent_size=512, normalize_latents=False)
                        z_critic_fake_loss = G_lsgan(G=encoder, D=latent_critic, opt=z_critic_optimizer, latents=images, labels=None)
                        z_critic_real_loss = D_lsgan(G=encoder, D=latent_critic, opt=z_critic_optimizer, latents=images, reals=tf.random_normal(shape=tf.shape(encoded_latents), name='z_real'))
                    with tf.name_scope('consistency_losses'):
                        ili_consistency_loss = 0.0
                        # L2 loss
                        l2_loss = MSE(images, encoded_images)
                        ili_consistency_loss += l2_loss

                        # VGG loss
                        image_vgg = Vgg16(args.cache_dir+'/vgg16.npy')
                        image_vgg.build(tf.image.resize(tf.transpose(images, perm=[0,2,3,1]), [args.vgg_shape,args.vgg_shape]))
                        image_perception = [image_vgg.conv1_1, image_vgg.conv1_2, image_vgg.conv3_2, image_vgg.conv4_2]
                        encoded_vgg = Vgg16(args.cache_dir+'/vgg16.npy')
                        encoded_vgg.build(tf.image.resize(tf.transpose(encoded_images, perm=[0,2,3,1]), [args.vgg_shape,args.vgg_shape]))
                        encoded_perception = [encoded_vgg.conv1_1, encoded_vgg.conv1_2, encoded_vgg.conv3_2, encoded_vgg.conv4_2]
                        vgg_loss = tf.reduce_sum([MSE(image, encoded) for image, encoded in zip(image_perception, encoded_perception)]) # https://github.com/machrisaa/tensorflow-vgg
                        ili_consistency_loss += vgg_loss

                        # # LPIPS loss
                        # lpips_url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'
                        # with dnnlib.util.open_url(lpips_url, cache_dir=args.cache_dir) as lpips:
                        #     lpips_network =  pickle.load(lpips)
                        # lpips_loss = lpips_network.get_output_for(images, encoded_images)
                        # tf.add_to_collection('LOSS_LPIPS', lpips_loss)
                        # ili_consistency_loss += args.lpips_lambda*lpips_loss
                        #
                        # # MSSIM loss
                        # mssim_loss = 1-tf.image.ssim_multiscale(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0)
                        # tf.add_to_collection('LOSS_MSSIM', mssim_loss)
                        # ili_consistency_loss += args.mssim_lambda*mssim_loss
                        #
                        # # LOGCOSH loss
                        # logcosh_loss = tf.keras.losses.logcosh(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]))
                        # tf.add_to_collection('LOSS_LOGCOSH', logcosh_loss)
                        # ili_consistency_loss += args.logcosh_lambda*logcosh_loss

                with tf.name_scope('LIL'):
                    with tf.name_scope('gan_losses'):
                        if not args.stylegan:
                            image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_progan.D_paper', num_channels=3, resolution=1024, structure=args.structure)
                        else:
                            image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_stylegan.D_basic', num_channels=3, resolution=1024, structure=args.structure)

                        y_critic_fake_loss = G_lsgan(G=generator, D=image_critic, opt=y_critic_optimizer, latents=latents, labels=empty_label)
                        y_critic_real_loss = D_lsgan(G=generator, D=image_critic, opt=y_critic_optimizer, latents=latents, labels=empty_label, reals=tf.identity(images, name='y_real'))

                    with tf.name_scope('consistency_losses'):
                        lil_consistency_loss = 0.0
                        # L1 loss
                        l1_loss = MAE(latents, generated_latents)
                        lil_consistency_loss += l1_loss

                with tf.name_scope('final_losses'):
                    tf.add_to_collection("LOSS_Z_CRITIC_REAL", z_critic_real_loss)
                    tf.add_to_collection("LOSS_Y_CRITIC_REAL", y_critic_real_loss)
                    tf.add_to_collection("LOSS_Z_CRITIC_FAKE", z_critic_fake_loss)
                    tf.add_to_collection("LOSS_Y_CRITIC_FAKE", y_critic_fake_loss)
                    tf.add_to_collection("LOSS_ILI_CONSISTENCY", ili_consistency_loss)
                    tf.add_to_collection("LOSS_LIL_CONSISTENCY", lil_consistency_loss)

                with tf.name_scope('metrics'):
                    psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
                    tf.add_to_collection('METRIC_PSNR', psnr)
                    tf.add_to_collection('METRIC_SSIM', ssim)

                with tf.name_scope('backprop'):
                    encoder_generator_optimizer.register_gradients(z_critic_fake_loss+ili_consistency_loss+y_critic_fake_loss+lil_consistency_loss, [*encoder.trainables.values()])
                    critic_optimizer.register_gradients(z_critic_real_loss+y_critic_real_loss, [*latent_critic.trainables.values()]+[*image_critic.trainables.values()])

    with tf.name_scope('summary'):
        _ = tf.summary.histogram('latents', latent_input, family='latents', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.histogram('encoded_latents', tf.concat(tf.get_collection('LATENT_ENCODED'), axis=0), family='latents', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.histogram('generated_latents', tf.concat(tf.get_collection('LATENT_GENERATED'), axis=0), family='latents', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.histogram('images', image_input, family='images', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.histogram('encoded_images', tf.concat(tf.get_collection('IMAGE_ENCODED'), axis=0), family='images', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.histogram('generated_images', tf.concat(tf.get_collection('IMAGE_GENERATED'), axis=0), family='images', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_REAL')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_real', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_REAL')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('latent_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Z_CRITIC_FAKE')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('image_critic_fake', tf.reduce_mean(tf.get_collection('LOSS_Y_CRITIC_FAKE')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ili_consistency', tf.reduce_mean(tf.get_collection('LOSS_ILI_CONSISTENCY')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('lil_consistency', tf.reduce_mean(tf.get_collection('LOSS_LIL_CONSISTENCY')), family='02_loss', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', tf.reduce_mean(tf.get_collection('METRIC_PSNR')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', tf.reduce_mean(tf.get_collection('METRIC_SSIM')), family='01_metric', collections=['SCALAR_SUMMARY', 'VAL_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('learning_rate', learning_rate, family='03_lr', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        original_image_summary = tf.summary.image('original', tf.image.resize(tf.clip_by_value(tf.transpose(image_input, perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        recovered_image_summary = tf.summary.image('recovered', tf.image.resize(tf.clip_by_value(tf.transpose(tf.concat(tf.get_collection('IMAGE_ENCODED'), axis=0), perm=[0,2,3,1]), 0.0, 1.0), [256,256]), max_outputs=args.image_output, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
        val_summary = tf.summary.merge(tf.get_collection('VAL_SUMMARY'))
        full_summary = tf.summary.merge_all()

    # DEFINE OPTIMIZE OPS
    with tf.name_scope('optimize'):
        encoder_generator_optimize = encoder_generator_optimizer.apply_updates()
        critic_optimize = critic_optimizer.apply_updates()

    os.makedirs(args.result_dir+'/model', exist_ok=True)
    os.makedirs(args.result_dir+'/summary', exist_ok=True)
    train_summary_writer = tf.summary.FileWriter(args.result_dir+'/summary/train')
    val_summary_writer = tf.summary.FileWriter(args.result_dir+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    lr = args.learning_rate
    for iter in tqdm(range(args.num_iter), ncols=50):
        train_imbatch = sess.run(get_images)
        train_latentbatch = np.random.normal(size=[args.num_gpus*args.minibatch_size, 512])
        _ = sess.run(critic_optimize, feed_dict={image_input: train_imbatch, latent_input: train_latentbatch, learning_rate: lr, empty_label: train_labelbatch})
        _ = sess.run(encoder_generator_optimize, feed_dict={image_input: train_imbatch, latent_input: train_latentbatch, learning_rate: lr, empty_label: train_labelbatch})

        train_scalar_summary = sess.run(scalar_summary, feed_dict={image_input: train_imbatch, latent_input: train_latentbatch, learning_rate: lr, empty_label: train_labelbatch})
        train_summary_writer.add_summary(train_scalar_summary, iter)
        val_scalar_summary = sess.run(val_summary, feed_dict={image_input: val_imbatch, latent_input: val_latentbatch, learning_rate: lr, empty_label: val_labelbatch})
        val_summary_writer.add_summary(val_scalar_summary, iter)

        if iter%args.save_iter==0:
            train_image_summary = sess.run(image_summary, feed_dict={image_input: train_imbatch, empty_label: train_labelbatch})
            train_summary_writer.add_summary(train_image_summary, iter)
            val_image_summary = sess.run(recovered_image_summary, feed_dict={image_input: val_imbatch, empty_label: val_labelbatch})
            val_summary_writer.add_summary(val_image_summary, iter)
            if iter==0:
                val_original_image_summary = sess.run(original_image_summary, feed_dict={image_input: val_imbatch, empty_label: val_labelbatch})
                val_summary_writer.add_summary(val_original_image_summary, iter)

            save_pkl((encoder, generator, latent_critic, image_critic), args.result_dir+'/model/model.pkl')



if __name__=='__main__':
    main()
