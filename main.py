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
from stylegan.training.networks_stylegan import *
from stylegan.training.misc import save_pkl

def main():
    base_option = utils.option.parse()
    tf.random.set_random_seed(base_option['seed'])
    tf.config.set_soft_device_placement(True)

    tflib.init_tf()
    gpus = np.arange(base_option['num_gpus'])

    if base_option['progan']:
        url = os.path.join(base_option['cache_dir'], 'karras2018iclr-celebahq-1024x1024.pkl')
        with open(url, 'rb') as f: _, _, Gs = pickle.load(f)
    else:
        url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        with dnnlib.util.open_url(url, cache_dir=base_option['cache_dir']) as f: _, _, Gs = pickle.load(f)

    # DEFINE OPTIMIZERS
    encoder_learning_rate = tf.placeholder(tf.float32, [], name='encoder_learning_rate')
    generator_learning_rate = tf.placeholder(tf.float32, [], name='generator_learning_rate')
    with tf.name_scope('optimizers'):
        encoder_optimizer = tf.train.AdamOptimizer(learning_rate=encoder_learning_rate, name='encoder_optimizer')
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate, name='generator_optimizer')
        z_critic_optimizer = tf.train.AdamOptimizer(learning_rate=encoder_learning_rate, name='z_critic_optimizer')
        y_critic_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate, name='y_critic_optimizer')

    for gpu_idx in gpus:
        with tf.device('/gpu:{}'.format(gpu_idx)):
            with tf.name_scope('model_gpu{}'.format(gpu_idx)):
                print("CONSTRUCTING MODEL WITH GPU: {}".format(gpu_idx))

                if bool(base_option['blur_filter']): blur = [1,2,1]
                else: blur=None

                empty_label = tf.constant(value=[], shape=[base_option['minibatch_size'], 0])
                if base_option['dataset_generated']:
                    # DEFINE NODES
                    print("SAMPLING DATASET FROM THE GENERATOR")
                    generator = Gs.clone('generator')
                    if base_option['uniform_noise']:
                        latents = tf.random.uniform(([base_option['minibatch_size']] + generator.input_shape[1:]), -1.0*base_option['noise_range'], 1.0*base_option['noise_range'])
                    else:
                        latents = tf.random.normal(([base_option['minibatch_size']] + generator.input_shape[1:]), stddev=1.0*base_option['noise_range'])
                    images = generator.get_output_for(latents, empty_label, is_validation=True, use_noise=False, randomize_noise=False)
                    encoder = tflib.Network("encoder", func_name='encoder.E_basic', nonlinearity=base_option['nonlinearity'], use_wscale=base_option['use_wscale'], mbstd_group_size=base_option['mbstd_group_size'], mbstd_num_features=base_option['mbstd_num_features'], fused_scale=base_option['fused_scale'], blur_filter=blur)
                    encoded_latents = encoder.get_output_for(images)
                    encoded_images = generator.get_output_for(encoded_latents, empty_label, is_validation=True, use_noise=False, randomize_noise=False)
                else:
                    # LOAD FFHQ DATASET
                    print("LOADING FFHQ DATASET")
                    from stylegan.training import dataset
                    generator = Gs.clone(name='generator')
                    ffhq = dataset.load_dataset(data_dir=base_option['data_dir'], tfrecord_dir='ffhq', verbose=False)
                    ffhq.configure(base_option['minibatch_size'])
                    images, labels = ffhq.get_minibatch_tf()
                    images = tf.cast(images, tf.float32)/255.0
                    if base_option['progan']:
                        encoder = tflib.Network("encoder", out_shape=[512], func_name='encoder.E_basic', nonlinearity=base_option['nonlinearity'], use_wscale=base_option['use_wscale'], mbstd_group_size=base_option['mbstd_group_size'], mbstd_num_features=base_option['mbstd_num_features'], fused_scale=base_option['fused_scale'], blur_filter=blur)
                    else:
                        encoder = tflib.Network("encoder", out_shape=[18, 512], func_name='encoder.E_basic', nonlinearity=base_option['nonlinearity'], use_wscale=base_option['use_wscale'], mbstd_group_size=base_option['mbstd_group_size'], mbstd_num_features=base_option['mbstd_num_features'], fused_scale=base_option['fused_scale'], blur_filter=blur)
                    encoded_latents = encoder.get_output_for(images)
                    encoded_images = generator.get_output_for(encoded_latents, labels, is_validation=True, use_noise=False, randomize_noise=False)
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
                        l2_loss = MSE(images, encoded_images)
                        _ = tf.summary.scalar('l2_loss', l2_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
                        regression_loss += base_option['l2_lambda']*l2_loss

                        # VGG loss
                        image_vgg = Vgg16(base_option['cache_dir']+'/vgg16.npy')
                        image_vgg.build(tf.image.resize(tf.transpose(images, perm=[0,2,3,1]), [224,224]))
                        image_perception = [image_vgg.conv1_1, image_vgg.conv1_2, image_vgg.conv3_2, image_vgg.conv4_2]
                        encoded_vgg = Vgg16(base_option['cache_dir']+'/vgg16.npy')
                        encoded_vgg.build(tf.image.resize(tf.transpose(encoded_images, perm=[0,2,3,1]), [224,224]))
                        encoded_perception = [encoded_vgg.conv1_1, encoded_vgg.conv1_2, encoded_vgg.conv3_2, encoded_vgg.conv4_2]
                        vgg_loss = tf.reduce_sum([MSE(image, encoded) for image, encoded in zip(image_perception, encoded_perception)]) # https://github.com/machrisaa/tensorflow-vgg
                        _ = tf.summary.scalar('vgg_loss', vgg_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])

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
                        if base_option['progan']:
                            image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_progan.D_paper', num_channels=3, resolution=1024, structure=None)
                        else:
                            image_critic = tflib.Network("y_critic", func_name='stylegan.training.networks_stylegan.D_basic', num_channels=3, resolution=1024, structure=None)

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

                    with tf.name_scope('optimize'):
                        encoder_gv = encoder_optimizer.compute_gradients(loss=encoder_loss, var_list=[*encoder.trainables.values()])
                        tf.add_to_collection('ENCODER_GV', encoder_gv)

                        generator_gv = generator_optimizer.compute_gradients(loss=generator_loss, var_list=[*generator.trainables.values()])
                        tf.add_to_collection('GENERATOR_GV', generator_gv)

                        z_critic_gv = z_critic_optimizer.compute_gradients(loss=z_critic_loss, var_list=[*latent_critic.trainables.values()])
                        tf.add_to_collection('Z_CRITIC_GV', z_critic_gv)

                        y_critic_gv = y_critic_optimizer.compute_gradients(loss=y_critic_loss, var_list=[*image_critic.trainables.values()])
                        tf.add_to_collection('Y_CRITIC_GV', y_critic_gv)


    with tf.name_scope('metric'):
        psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1]), 1.0))

    # DEFINE GRAPH NEEDED FOR TESTING
    with tf.name_scope("test_encode"):
        image_list = [image for image in os.listdir(base_option['validation_dir']) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
        assert len(image_list)>0
        # G_synth_test = generator.components.synthesis.clone()
        test_image_input = tf.placeholder(tf.float32, [None,1024,1024,3], name='image_input')
        test_encoded_latent = encoder.get_output_for(tf.transpose(test_image_input, perm=[0,3,1,2]))
        latent_manipulator = tf.placeholder_with_default(tf.zeros_like(test_encoded_latent), test_encoded_latent.shape, name='latent_manipulator')
        test_recovered_image = generator.get_output_for(test_encoded_latent+latent_manipulator, tf.constant(value=[], shape=[len(image_list), 0]), is_validation=True)

        tf.add_to_collection('TEST_NODES', test_image_input)
        tf.add_to_collection('TEST_NODES', test_encoded_latent)
        tf.add_to_collection('TEST_NODES', test_recovered_image)
        tf.add_to_collection('TEST_NODES', latent_manipulator)


        val_imbatch = np.stack([np.array(PIL.Image.open(base_option['validation_dir']+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0)/255.0
        val_psnr = tf.reduce_mean(tf.image.psnr(test_image_input, tf.transpose(test_recovered_image, perm=[0,2,3,1]), 1.0))
        val_ssim = tf.reduce_mean(tf.image.ssim(test_image_input, tf.transpose(test_recovered_image, perm=[0,2,3,1]), 1.0))
        _ = tf.summary.scalar('psnr', val_psnr, family='metrics', collections=['TEST_SUMMARY', 'TEST_SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', val_ssim, family='metrics', collections=['TEST_SUMMARY', 'TEST_SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('recovered', tf.clip_by_value(tf.transpose(test_recovered_image, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=64, family='images', collections=['TEST_SUMMARY', 'TEST_IMAGE_SUMMARY', 'VAL_IMAGE_SUMMARY'])
        original_image_summary = tf.summary.image('original', test_image_input, max_outputs=64, family='images', collections=['TEST_SUMMARY', 'TEST_IMAGE_SUMMARY'])


    # DEFINE SUMMARIES
    with tf.name_scope('summary'):
        _ = tf.summary.scalar('encoder_learning_rate', encoder_learning_rate, family='metrics', collections=['ENCODER_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('generator_learning_rate', generator_learning_rate, family='metrics', collections=['GENERATOR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('regression_loss', regression_loss, family='loss', collections=['ENCODER_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('z_critic_real_loss', z_critic_real_loss, family='loss', collections=['Z_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('y_critic_real_loss', y_critic_real_loss, family='loss', collections=['Y_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('z_critic_fake_loss', z_critic_fake_loss, family='loss', collections=['Z_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('y_critic_fake_loss', y_critic_fake_loss, family='loss', collections=['Y_CRITIC_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', psnr, family='metrics', collections=['GENERATOR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', ssim, family='metrics', collections=['GENERATOR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('target', tf.clip_by_value(tf.transpose(images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('encoded', tf.clip_by_value(tf.transpose(encoded_images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])

        encoder_summary = tf.summary.merge(tf.get_collection('ENCODER_SUMMARY'))
        generator_summary = tf.summary.merge(tf.get_collection('GENERATOR_SUMMARY'))
        z_critic_summary = tf.summary.merge(tf.get_collection('Z_CRITIC_SUMMARY'))
        y_critic_summary = tf.summary.merge(tf.get_collection('Y_CRITIC_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection("IMAGE_SUMMARY"))
        test_scalar_summary = tf.summary.merge(tf.get_collection('TEST_SCALAR_SUMMARY'))
        val_image_summary = tf.summary.merge(tf.get_collection('VAL_IMAGE_SUMMARY'))

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimize'):
        print("================== ENCODER VARS ==================")
        print("\n".join([v.name for v in [*encoder.trainables.values()]]))
        average_encoder_gv = utils.multigpu.average_gradients(tf.get_collection("ENCODER_GV"))
        encoder_optimize = encoder_optimizer.apply_gradients(average_encoder_gv, name='encoder_optimize')

        print("================== GENERATOR VARS ==================")
        print("\n".join([v.name for v in [*generator.trainables.values()]]))
        average_generator_gv = utils.multigpu.average_gradients(tf.get_collection("GENERATOR_GV"))
        generator_optimize = generator_optimizer.apply_gradients(average_generator_gv, name='generator_optimize')

        print("================== Z_CRITIC VARS ==================")
        print("\n".join([v.name for v in [*latent_critic.trainables.values()]]))
        average_z_critic_gv = utils.multigpu.average_gradients(tf.get_collection("Z_CRITIC_GV"))
        z_critic_optimize = z_critic_optimizer.apply_gradients(average_z_critic_gv, name='z_critic_optimize')

        print("================== Y_CRITIC VARS ==================")
        print("\n".join([v.name for v in [*image_critic.trainables.values()]]))
        average_y_critic_gv = utils.multigpu.average_gradients(tf.get_collection("Y_CRITIC_GV"))
        y_critic_optimize = y_critic_optimizer.apply_gradients(average_y_critic_gv, name='y_critic_optimize')

    os.makedirs(base_option['result_dir']+'/model', exist_ok=True)
    os.makedirs(base_option['result_dir']+'/summary', exist_ok=True)
    train_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/train')
    val_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    original_image_summary = sess.run(original_image_summary, feed_dict={test_image_input: val_imbatch})
    val_summary_writer.add_summary(original_image_summary)
    encoder_lr = base_option['encoder_learning_rate']
    generator_lr = base_option['generator_learning_rate']
    for iter in tqdm(range(base_option['num_iter'])):
        iter_encoder_summary, _ = sess.run([encoder_summary, encoder_optimize], feed_dict={encoder_learning_rate: encoder_lr})
        for _ in range(base_option['critic_iter']):
            iter_z_critic_summary, _ = sess.run([z_critic_summary, z_critic_optimize], feed_dict={encoder_learning_rate: encoder_lr})

        iter_generator_summary, _ = sess.run([generator_summary, generator_optimize], feed_dict={generator_learning_rate: generator_lr})
        for _ in range(base_option['critic_iter']):
            iter_y_critic_summary, _ = sess.run([y_critic_summary, y_critic_optimize], feed_dict={generator_learning_rate: generator_lr})

        train_summary_writer.add_summary(iter_encoder_summary, iter)
        train_summary_writer.add_summary(iter_generator_summary, iter)
        train_summary_writer.add_summary(iter_z_critic_summary, iter)
        train_summary_writer.add_summary(iter_y_critic_summary, iter)

        if iter%base_option['save_iter']==0 or iter==0:
            iter_image_summary = sess.run(image_summary)
            train_summary_writer.add_summary(iter_image_summary, iter)
            val_iter_image_summary = sess.run(val_image_summary, feed_dict={test_image_input: val_imbatch})
            val_summary_writer.add_summary(val_iter_image_summary, iter)
            save_pkl((encoder, generator, latent_critic, image_critic), base_option['result_dir']+'/model/model.pkl')



if __name__=='__main__':
    main()
