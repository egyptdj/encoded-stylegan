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
from lpips import lpips_tf
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from stylegan.training.networks_stylegan import *


def encode(
    input,                              # First input: Images [minibatch, channel, height, width].
    out_shape           = [18, 512],
    reuse               = False,
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'fixed',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    assert isinstance(out_shape, list) or isinstance(out_shape, tuple)
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]
    out_fmap = np.prod(out_shape)

    with tf.variable_scope('encoder', reuse=reuse):
        # Building blocks.
        def block(x, res): # res = 2..resolution_log2
            with tf.variable_scope('%dx%d' % (2**res, 2**res)):
                if res >= 3: # 8x8 and up
                    with tf.variable_scope('Conv0'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
                else: # 4x4
                    if mbstd_group_size > 1:
                        x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                    with tf.variable_scope('Conv'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense0'):
                        x = act(apply_bias(dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense1'):
                        x = apply_bias(dense(x, fmaps=out_fmap, gain=1, use_wscale=use_wscale))
                        x = tf.reshape(x, [-1]+out_shape)
                return x

        x = input
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)
        output = block(x, 2)

        assert output.dtype == tf.as_dtype(dtype)
        output = tf.identity(output, name='output')
    return output


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
        noise_latents = tf.random_normal([base_option['minibatch_size']] + Gs.input_shape[1:])
        images = Gs.get_output_for(noise_latents, None, is_validation=True, use_noise=False, randomize_noise=False)
        latents = tf.get_default_graph().get_tensor_by_name('Gs_1/G_mapping/dlatents_out:0')
        _D_out = D.get_output_for(images, None)
        D_intermediate = tf.get_default_graph().get_tensor_by_name('D_1/'+'cond/'*int(np.log2(base_option['disc_resolution'])-2)+'{}x{}/Conv1_down/LeakyReLU/IdentityN:0'.format(base_option['disc_resolution'], base_option['disc_resolution']))
        encoded_latents = encode(D_intermediate, resolution=base_option['disc_resolution'], reuse=False)
        encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=False, randomize_noise=False)
    else:
        # LOAD FFHQ DATASET
        print("LOADING FFHQ DATASET")
        from stylegan.training import dataset
        ffhq = dataset.load_dataset(data_dir=base_option['data_dir'], tfrecord_dir='ffhq', verbose=False)
        ffhq.configure(base_option['minibatch_size'])
        images, _ = ffhq.get_minibatch_tf()
        images = tf.cast(images, tf.float32)/255.0
        _D_out = D.get_output_for(images, None)
        D_intermediate = tf.get_default_graph().get_tensor_by_name('D_1/'+'cond/'*int(np.log2(base_option['disc_resolution'])-2)+'{}x{}/Conv1_down/LeakyReLU/IdentityN:0'.format(base_option['disc_resolution'], base_option['disc_resolution']))
        encoded_latents = encode(D_intermediate, resolution=base_option['disc_resolution'], reuse=False)
        encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=False, randomize_noise=False)

    recovered_encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=True, randomize_noise=True)
    # LOAD LATENT DIRECTIONS
    latent_smile = tf.stack([tf.cast(tf.constant(np.load('latents/smile.npy'), name='latent_smile'), tf.float32)]*base_option['minibatch_size'], axis=0)
    latent_encoded_smile = tf.identity(encoded_latents)
    latent_encoded_smile += 2.0 * latent_smile
    smile_encoded_images = Gs.components.synthesis.get_output_for(latent_encoded_smile, None, is_validation=True, use_noise=True, randomize_noise=True)

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

    with tf.name_scope('metric'):
        psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 1.0))

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
        summary = tf.summary.merge_all()

    # DEFINE GRAPH NEEDED FOR TESTING
    with tf.name_scope("test_encode"):
        # G_synth_test = Gs.components.synthesis.clone()
        test_image_input = tf.placeholder(tf.float32, [None,1024,1024,3], name='image_input')
        _D_out = D.get_output_for(tf.transpose(test_image_input, perm=[0,3,1,2]), None)
        D_intermediate2 = tf.get_default_graph().get_tensor_by_name('test_encode/D/'+'cond/'*int(np.log2(base_option['disc_resolution'])-2)+'{}x{}/Conv1_down/LeakyReLU/IdentityN:0'.format(base_option['disc_resolution'], base_option['disc_resolution']))
        test_encoded_latent = encode(D_intermediate2, resolution=base_option['disc_resolution'], reuse=True)
        latent_manipulator = tf.placeholder_with_default(tf.zeros_like(test_encoded_latent), test_encoded_latent.shape, name='latent_manipulator')
        test_recovered_image = Gs.components.synthesis.get_output_for(test_encoded_latent+latent_manipulator, None, is_validation=True, use_noise=True, randomize_noise=False)

        tf.add_to_collection('TEST_NODES', test_image_input)
        tf.add_to_collection('TEST_NODES', test_encoded_latent)
        tf.add_to_collection('TEST_NODES', test_recovered_image)
        tf.add_to_collection('TEST_NODES', latent_manipulator)

        image_list = [image for image in os.listdir(base_option['validation_dir']) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
        assert len(image_list)>0

        val_imbatch = np.stack([np.array(PIL.Image.open(base_option['validation_dir']+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0)/255.0
        val_feed_dict = {test_image_input: val_imbatch}
        _ = tf.summary.image('recovered', tf.clip_by_value(tf.transpose(test_recovered_image, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=64, family='images', collections=['TEST_SUMMARY'])
        _ = tf.summary.image('original', test_image_input, max_outputs=64, family='images', collections=['TEST_SUMMARY'])
        test_image_summary = tf.summary.merge(tf.get_collection('TEST_SUMMARY'))

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
    for iter in tqdm(range(base_option['num_iter'])):
        iter_scalar_summary, _ = sess.run([scalar_summary, optimize])
        train_summary_writer.add_summary(iter_scalar_summary, iter)
        if iter%base_option['save_iter']==0 or iter==0:
            iter_image_summary = sess.run(image_summary)
            train_summary_writer.add_summary(iter_image_summary, iter)
            val_iter_image_summary = sess.run(test_image_summary, feed_dict=val_feed_dict)
            val_summary_writer.add_summary(val_iter_image_summary, iter)
            saver.save(sess, base_option['result_dir']+'/model/encoded_stylegan.ckpt')
