import os
import sys
import utils
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
from tqdm import tqdm
from vgg import Vgg16
from encoder import encode
from lpips import lpips_tf
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from stylegan.training.networks_stylegan import *


def main():
    base_option = utils.option.parse()

    tflib.init_tf()
    try:
        url = os.path.join(base_option['cache_dir'], 'karras2019stylegan-ffhq-1024x1024.pkl')
        with open(url, 'rb') as f: _, _, Gs = pickle.load(f)
    except:
        url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        with dnnlib.util.open_url(url, cache_dir=base_option['cache_dir']) as f: _, _, Gs = pickle.load(f)

    if base_option['dataset_generated']:
        # DEFINE NODES
        print("SAMPLING DATASET FROM THE GENERATOR")
        if base_option['uniform_noise']:
            noise_latents = tf.random.uniform(([base_option['minibatch_size']] + Gs.input_shape[1:]), -1.0*base_option['noise_range'], 1.0*base_option['noise_range'])
        else:
            noise_latents = tf.random.normal(([base_option['minibatch_size']] + Gs.input_shape[1:]), stddev=1.0*base_option['noise_range'])
        images = Gs.get_output_for(noise_latents, None, is_validation=True, use_noise=False, randomize_noise=False)
        latents = tf.get_default_graph().get_tensor_by_name('Gs_1/G_mapping/dlatents_out:0')
        encoded_latents = encode(images, reuse=False)
        encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=False, randomize_noise=False)
    else:
        # LOAD FFHQ DATASET
        print("LOADING FFHQ DATASET")
        from stylegan.training import dataset
        ffhq = dataset.load_dataset(data_dir=base_option['data_dir'], tfrecord_dir='ffhq', verbose=False)
        ffhq.configure(base_option['minibatch_size'])
        images, _ = ffhq.get_minibatch_tf()
        images = tf.cast(images, tf.float32)/255.0
        encoded_latents = encode(images, reuse=False)
        encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=False, randomize_noise=False)

    recovered_encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=True, randomize_noise=True)

    # LOAD LATENT DIRECTIONS
    latent_smile = tf.stack([tf.cast(tf.constant(np.load('latents/smile.npy'), name='latent_smile'), tf.float32)]*base_option['minibatch_size'], axis=0)
    latent_encoded_smile = tf.identity(encoded_latents)
    latent_encoded_smile += 2.0 * latent_smile
    smile_encoded_images = Gs.components.synthesis.get_output_for(latent_encoded_smile, None, is_validation=True, use_noise=True, randomize_noise=True)

    with tf.name_scope('metric'):
        psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 1.0))

    # DEFINE GRAPH NEEDED FOR TESTING
    with tf.name_scope("test_encode"):
        # G_synth_test = Gs.components.synthesis.clone()
        test_image_input = tf.placeholder(tf.float32, [None,1024,1024,3], name='image_input')
        test_encoded_latent = encode(tf.transpose(test_image_input, perm=[0,3,1,2]), reuse=True)
        latent_manipulator = tf.placeholder_with_default(tf.zeros_like(test_encoded_latent), test_encoded_latent.shape, name='latent_manipulator')
        test_recovered_image = Gs.components.synthesis.get_output_for(test_encoded_latent+latent_manipulator, None, is_validation=True, use_noise=True, randomize_noise=False)

        tf.add_to_collection('TEST_NODES', test_image_input)
        tf.add_to_collection('TEST_NODES', test_encoded_latent)
        tf.add_to_collection('TEST_NODES', test_recovered_image)
        tf.add_to_collection('TEST_NODES', latent_manipulator)

        image_list = [image for image in os.listdir(base_option['validation_dir']) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
        assert len(image_list)>0

        val_imbatch = np.stack([np.array(PIL.Image.open(base_option['validation_dir']+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0)/255.0
        val_psnr = tf.reduce_mean(tf.image.psnr(test_image_input, tf.transpose(test_recovered_image, perm=[0,2,3,1]), 1.0))
        val_ssim = tf.reduce_mean(tf.image.ssim(test_image_input, tf.transpose(test_recovered_image, perm=[0,2,3,1]), 1.0))
        _ = tf.summary.scalar('psnr', val_psnr, family='metrics', collections=['TEST_SUMMARY', 'TEST_SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', val_ssim, family='metrics', collections=['TEST_SUMMARY', 'TEST_SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('recovered', tf.clip_by_value(tf.transpose(test_recovered_image, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=64, family='images', collections=['TEST_SUMMARY', 'TEST_IMAGE_SUMMARY', 'VAL_IMAGE_SUMMARY'])
        original_image_summary = tf.summary.image('original', test_image_input, max_outputs=64, family='images', collections=['TEST_SUMMARY', 'TEST_IMAGE_SUMMARY'])

    with tf.name_scope('loss'):
        total_loss = 0.0
        mse = tf.keras.losses.MeanSquaredError()
        mae = tf.keras.losses.MeanAbsoluteError()

        if base_option['vgg_lambda']:
            image_vgg = Vgg16('/media/bispl/dbx/Dropbox/Academic/01_Research/99_DATASET/VGG16_MODEL/vgg16.npy')
            image_vgg.build(tf.image.resize(tf.transpose(images, perm=[0,2,3,1]), [224,224]))
            image_perception = [image_vgg.conv1_1, image_vgg.conv1_2, image_vgg.conv3_2, image_vgg.conv4_2]
            encoded_vgg = Vgg16('/media/bispl/dbx/Dropbox/Academic/01_Research/99_DATASET/VGG16_MODEL/vgg16.npy')
            encoded_vgg.build(tf.image.resize(tf.transpose(encoded_images, perm=[0,2,3,1]), [224,224]))
            encoded_perception = [encoded_vgg.conv1_1, encoded_vgg.conv1_2, encoded_vgg.conv3_2, encoded_vgg.conv4_2]
            vgg_loss = tf.reduce_sum([mse(image, encoded) for image, encoded in zip(image_perception, encoded_perception)]) # https://github.com/machrisaa/tensorflow-vgg
            _ = tf.summary.scalar('vgg_loss', vgg_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            total_loss += base_option['vgg_lambda']*vgg_loss

        if base_option['encoding_lambda'] and base_option['dataset_generated']:
            encoding_loss = mse(latents, encoded_latents)
            _ = tf.summary.scalar('encoding_loss', encoding_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            total_loss += base_option['encoding_lambda']*encoding_loss

        if base_option['lpips_lambda']:
            lpips_loss =  tf.reduce_mean(lpips_tf.lpips(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1])))
            _ = tf.summary.scalar('lpips_loss', lpips_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            total_loss += base_option['lpips_lambda']*lpips_loss

        if base_option['l2_lambda']:
            l2_loss = mse(images, encoded_images)
            _ = tf.summary.scalar('l2_loss', l2_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            total_loss += base_option['l2_lambda']*l2_loss

        if base_option['l1_latent_lambda'] and base_option['dataset_generated']:
            l1_latent_loss = mae(latents, encoded_latents)
            _ = tf.summary.scalar('l1_latent_loss', l1_latent_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            total_loss += base_option['l1_latent_lambda']*l1_latent_loss

        if base_option['l1_image_lambda']:
            l1_image_loss = mae(images, encoded_images)
            _ = tf.summary.scalar('l1_image_loss', l1_image_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            total_loss += base_option['l1_image_lambda']*l1_image_loss

    # DEFINE SUMMARIES
    with tf.name_scope('summary'):
        _ = tf.summary.scalar('total_loss', total_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('psnr', psnr, family='metrics', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('ssim', ssim, family='metrics', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('target', tf.clip_by_value(tf.transpose(images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('encoded', tf.clip_by_value(tf.transpose(encoded_images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('recovered(withnoise)', tf.clip_by_value(tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.image('recovered_smile', tf.clip_by_value(tf.transpose(smile_encoded_images, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
        scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
        image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
        test_scalar_summary = tf.summary.merge(tf.get_collection('TEST_SCALAR_SUMMARY'))
        val_image_summary = tf.summary.merge(tf.get_collection('VAL_IMAGE_SUMMARY'))

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimize'):
        encoder_vars = tf.trainable_variables('encoder')
        optimizer = tf.train.AdamOptimizer(learning_rate=base_option['learning_rate'], name='optimizer')
        gv = optimizer.compute_gradients(loss=total_loss, var_list=encoder_vars)
        optimize = optimizer.apply_gradients(gv, name='optimize')

    saver = tf.train.Saver(var_list=tf.global_variables('encoder'), name='saver')
    train_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/train')
    val_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    original_image_summary = sess.run(original_image_summary, feed_dict={test_image_input: val_imbatch})
    val_summary_writer.add_summary(original_image_summary)
    for iter in tqdm(range(base_option['num_iter'])):
        iter_scalar_summary, val_iter_scalar_summary, _ = sess.run([scalar_summary, test_scalar_summary, optimize], feed_dict=val_feed_dict)
        train_summary_writer.add_summary(iter_scalar_summary, iter)
        val_summary_writer.add_summary(val_iter_scalar_summary, iter)
        if iter%base_option['save_iter']==0 or iter==0:
            iter_image_summary = sess.run(image_summary)
            train_summary_writer.add_summary(iter_image_summary, iter)
            val_iter_image_summary = sess.run(val_image_summary)
            val_summary_writer.add_summary(val_iter_image_summary, iter)
            saver.save(sess, base_option['result_dir']+'/model/encoded_stylegan.ckpt')



if __name__=='__main__':
    main()
