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
from encoder import encode
from regularizer import modeseek
from lpips import lpips_tf
from laploss import laploss
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from stylegan.training.misc import save_pkl
from stylegan.training.networks_stylegan import *

def main():
    base_option = utils.option.parse()
    tf.random.set_random_seed(base_option['seed'])

    tflib.init_tf()
    url = os.path.join(base_option['cache_dir'], 'karras2018iclr-lsun-diningtable-256x256.pkl')
    with open(url, 'rb') as f: _, _, Gs = pickle.load(f)

    if bool(base_option['blur_filter']): blur = [1,2,1]
    else: blur=None

    labels = tf.constant(value=[], shape=[base_option['minibatch_size'], 0])

    # LOAD LSUN DININGROOM DATASET
    print("LOADING LSUN DININGROOM DATASET")
    import tensorflow_datasets as tfds

    def resize(height=256, width=256):
        def transformation_func(x):
            return tf.image.resize_with_crop_or_pad(x['image'], height, width)
        return transformation_func

    dataset_builder = tfds.builder("lsun/dining_room")
    dataset_builder.download_and_prepare(download_dir=str(base_option['data_dir']))
    dataset_train = dataset_builder.as_dataset(split="train")
    dataset_train = dataset_train.map(resize())
    dataset_train = dataset_train.repeat().shuffle(buffer_size=1024, seed=base_option['seed']).batch(base_option['minibatch_size'])
    train_iterator = dataset_train.make_one_shot_iterator()
    get_train_image = train_iterator.get_next()

    dataset_val = dataset_builder.as_dataset(split="validation")
    dataset_val = dataset_val.map(resize())
    dataset_val = dataset_val.repeat().batch(base_option['minibatch_size'])
    val_iterator = dataset_val.make_one_shot_iterator()
    get_val_image = val_iterator.get_next()


    generator = Gs.clone(name='generator')
    image_input = tf.placeholder(tf.uint8, [None,256,256,3], name='image_input')
    images = tf.transpose(tf.cast(image_input, tf.float32)/255.0, perm=[0,3,1,2])
    encoded_latents = encode(images, reuse=False, nonlinearity=base_option['nonlinearity'], use_wscale=base_option['use_wscale'], mbstd_group_size=base_option['mbstd_group_size'], mbstd_num_features=base_option['mbstd_num_features'], fused_scale=base_option['fused_scale'], blur_filter=blur)
    latent_manipulator = tf.placeholder_with_default(tf.zeros_like(encoded_latents), encoded_latents.shape, name='latent_manipulator')
    encoded_images = generator.get_output_for(encoded_latents+latent_manipulator, labels, is_validation=True, use_noise=False, randomize_noise=False)
    """ images is the X and Y domain,
        encoded_latents is the z domain,
        encoder is the mapping E,
        generator is the mapping D,
        encoded_images is the DEx domain """

    with tf.name_scope('metric'):
        psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))

    with tf.name_scope('loss'):
        MSE = tf.keras.losses.MeanSquaredError()
        MAE = tf.keras.losses.MeanAbsoluteError()

        with tf.name_scope('regression_loss'):
            regression_loss = 0.0

            # L2 Loss
            l2_loss = MSE(images, encoded_images)
            regression_loss += base_option['l2_lambda']*l2_loss

            # VGG loss
            image_vgg = Vgg16(base_option['cache_dir']+'/vgg16.npy')
            image_vgg.build(tf.image.resize(tf.transpose(images, perm=[0,2,3,1]), [224,224]))
            image_perception = [image_vgg.conv1_1, image_vgg.conv1_2, image_vgg.conv3_2, image_vgg.conv4_2]
            encoded_vgg = Vgg16(base_option['cache_dir']+'/vgg16.npy')
            encoded_vgg.build(tf.image.resize(tf.transpose(encoded_images, perm=[0,2,3,1]), [224,224]))
            encoded_perception = [encoded_vgg.conv1_1, encoded_vgg.conv1_2, encoded_vgg.conv3_2, encoded_vgg.conv4_2]
            vgg_loss = tf.reduce_sum([MSE(image, encoded) for image, encoded in zip(image_perception, encoded_perception)]) # https://github.com/machrisaa/tensorflow-vgg

            regression_loss += base_option['vgg_lambda']*vgg_loss

        with tf.name_scope('z_domain_loss'):
            latent_critic = tflib.Network("z_critic", func_name='stylegan.training.networks_stylegan.G_mapping', dlatent_size=1, mapping_layers=8, latent_size=512, normalize_latents=False)
            fake_latent = tf.random.normal(shape=tf.shape(encoded_latents), name='z_rand')
            real_latent = tf.identity(encoded_latents, name='z_real')

            fake_latent_critic_out = latent_critic.get_output_for(tf.reshape(fake_latent, [-1,512]), None)
            real_latent_critic_out = latent_critic.get_output_for(tf.reshape(real_latent, [-1,512]), None)

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
                with tf.name_scope('gradient_penalty'):
                    epsilon = tf.random.uniform([], name='epsilon')
                    gradient_latent = tf.identity((epsilon * real_latent + (1-epsilon) * fake_latent), name='gradient_latent')
                    critic_gradient_out = latent_critic.get_output_for(tf.reshape(gradient_latent, [-1,512]), None)
                    gradients = tf.gradients(critic_gradient_out, gradient_latent, name='gradients')
                    gradients_norm = tf.norm(gradients[0], ord=2, name='gradient_norm')
                    gradient_penalty = tf.square(gradients_norm -1)

            z_critic_real_loss = -real_latent_critic_loss + fake_latent_critic_loss + gradient_penalty
            z_critic_fake_loss = -fake_latent_loss

        with tf.name_scope('y_domain_loss'):
            image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_progan.D_paper', num_channels=3, resolution=256, structure=None)

            fake_image = generator.get_output_for(tf.random.normal(shape=tf.shape(encoded_latents), name='z_rand'), labels, is_validation=True, use_noise=False, randomize_noise=False)
            real_image = tf.identity(images, name='y_real')

            fake_image_critic_out = image_critic.get_output_for(fake_image, None)
            real_image_critic_out = image_critic.get_output_for(real_image, None)
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
                with tf.name_scope('gradient_penalty'):
                    epsilon = tf.random.uniform([], name='epsilon')
                    gradient_image = tf.identity((epsilon * real_image + (1-epsilon) * fake_image), name='gradient_image')
                    critic_gradient_out = image_critic.get_output_for(gradient_image, None)
                    gradients = tf.gradients(critic_gradient_out, gradient_image, name='gradients')
                    gradients_norm = tf.norm(gradients[0], ord=2, name='gradient_norm')
                    gradient_penalty = tf.square(gradients_norm -1)

            y_critic_real_loss = -real_image_critic_loss + fake_image_critic_loss + gradient_penalty
            y_critic_fake_loss = -fake_image_loss

        with tf.name_scope('final_losses'):
            encoder_loss = regression_loss + z_critic_fake_loss
            generator_loss = regression_loss + y_critic_fake_loss
            z_critic_loss = tf.identity(z_critic_real_loss)
            y_critic_loss = tf.identity(y_critic_real_loss)

    # DEFINE SUMMARIES
    encoder_learning_rate = tf.placeholder(tf.float32, [], name='encoder_learning_rate')
    generator_learning_rate = tf.placeholder(tf.float32, [], name='generator_learning_rate')

    with tf.name_scope('summary'):
        _ = tf.summary.scalar('encoder_learning_rate', encoder_learning_rate, family='metrics', collections=['ENCODER_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('generator_learning_rate', generator_learning_rate, family='metrics', collections=['GENERATOR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('l2_loss', l2_loss, family='loss', collections=['ENCODER_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('vgg_loss', vgg_loss, family='loss', collections=['ENCODER_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('regression_loss', regression_loss, family='loss', collections=['ENCODER_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('z_critic_real_loss', z_critic_real_loss, family='loss', collections=['Z_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('y_critic_real_loss', y_critic_real_loss, family='loss', collections=['Y_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('z_critic_fake_loss', z_critic_fake_loss, family='loss', collections=['Z_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('y_critic_fake_loss', y_critic_fake_loss, family='loss', collections=['Y_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', psnr, family='metrics', collections=['GENERATOR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', ssim, family='metrics', collections=['GENERATOR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('target', tf.clip_by_value(tf.transpose(images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=4, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('encoded', tf.clip_by_value(tf.transpose(encoded_images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=4, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])

        encoder_summary = tf.summary.merge(tf.get_collection('ENCODER_SUMMARY'))
        generator_summary = tf.summary.merge(tf.get_collection('GENERATOR_SUMMARY'))
        z_critic_summary = tf.summary.merge(tf.get_collection('Z_CRITIC_SUMMARY'))
        y_critic_summary = tf.summary.merge(tf.get_collection('Y_CRITIC_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection("IMAGE_SUMMARY"))
        full_summary = tf.summary.merge_all()

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimize'):
        encoder_vars = tf.trainable_variables('encoder')
        print("================== ENCODER VARS ==================")
        print("\n".join([v.name for v in encoder_vars]))
        encoder_optimizer = tf.train.AdamOptimizer(learning_rate=encoder_learning_rate, name='encoder_optimizer')
        encoder_gv = encoder_optimizer.compute_gradients(loss=encoder_loss, var_list=encoder_vars)
        encoder_optimize = encoder_optimizer.apply_gradients(encoder_gv, name='encoder_optimize')

        generator_vars = tf.trainable_variables('generator')
        print("================== GENERATOR VARS ==================")
        print("\n".join([v.name for v in generator_vars]))
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate, name='generator_optimizer')
        generator_gv = generator_optimizer.compute_gradients(loss=generator_loss, var_list=generator_vars)
        generator_optimize = generator_optimizer.apply_gradients(generator_gv, name='generator_optimize')

        z_critic_vars = tf.trainable_variables('z_critic')
        print("================== Z_CRITIC VARS ==================")
        print("\n".join([v.name for v in z_critic_vars]))
        z_critic_optimizer = tf.train.AdamOptimizer(learning_rate=encoder_learning_rate, name='z_critic_optimizer')
        z_critic_gv = z_critic_optimizer.compute_gradients(loss=z_critic_loss, var_list=z_critic_vars)
        z_critic_optimize = z_critic_optimizer.apply_gradients(z_critic_gv, name='z_critic_optimize')

        y_critic_vars = tf.trainable_variables('y_critic')
        print("================== Y_CRITIC VARS ==================")
        print("\n".join([v.name for v in y_critic_vars]))
        y_critic_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate, name='y_critic_optimizer')
        y_critic_gv = y_critic_optimizer.compute_gradients(loss=y_critic_loss, var_list=y_critic_vars)
        y_critic_optimize = y_critic_optimizer.apply_gradients(y_critic_gv, name='y_critic_optimize')

    saver = tf.train.Saver(var_list=tf.global_variables(), name='saver')
    train_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/train')
    val_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    encoder_lr = base_option['encoder_learning_rate']
    generator_lr = base_option['generator_learning_rate']
    for iter in tqdm(range(base_option['num_iter'])):
        train_image_batch = sess.run(get_train_image)
        iter_image_summary = sess.run(image_summary, feed_dict={image_input:train_image_batch})
        iter_encoder_summary, _ = sess.run([encoder_summary, encoder_optimize], feed_dict={image_input: train_image_batch, encoder_learning_rate: encoder_lr})
        for _ in range(base_option['critic_iter']):
            iter_z_critic_summary, _ = sess.run([z_critic_summary, z_critic_optimize], feed_dict={image_input: train_image_batch, encoder_learning_rate: encoder_lr})

        iter_generator_summary, _ = sess.run([generator_summary, generator_optimize], feed_dict={image_input: train_image_batch, generator_learning_rate: generator_lr})
        for _ in range(base_option['critic_iter']):
            iter_y_critic_summary, _ = sess.run([y_critic_summary, y_critic_optimize], feed_dict={image_input: train_image_batch, generator_learning_rate: generator_lr})

        train_summary_writer.add_summary(iter_image_summary, iter)
        train_summary_writer.add_summary(iter_encoder_summary, iter)
        train_summary_writer.add_summary(iter_generator_summary, iter)
        train_summary_writer.add_summary(iter_z_critic_summary, iter)
        train_summary_writer.add_summary(iter_y_critic_summary, iter)

        val_image_batch = sess.run(get_val_image)
        if iter%base_option['save_iter']==0 or iter==0:
            val_summary = sess.run(full_summary, feed_dict={image_input:val_image_batch, encoder_learning_rate: encoder_lr, generator_learning_rate: generator_lr})
            val_summary_writer.add_summary(val_summary, iter)
            saver.save(sess, base_option['result_dir']+'/model/encoded_stylegan.ckpt')
            # save_pkl((generator, z_critic, y_critic), base_option['result_dir']+'/model/encoded_stylegan.pkl')
        else:
            val_encoder_summary, val_generator_summary, val_z_critic_summary, val_y_critic_summary = sess.run([encoder_summary,generator_summary,z_critic_summary,y_critic_summary], feed_dict={image_input:val_image_batch, encoder_learning_rate: encoder_lr, generator_learning_rate: generator_lr})
            val_summary_writer.add_summary(val_encoder_summary, iter)
            val_summary_writer.add_summary(val_generator_summary, iter)
            val_summary_writer.add_summary(val_z_critic_summary, iter)
            val_summary_writer.add_summary(val_y_critic_summary, iter)



if __name__=='__main__':
    main()
