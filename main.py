import os
import sys
import utils
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
from tqdm import tqdm
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
        input.set_shape([None, num_channels, resolution, resolution])
        input = tf.cast(input, dtype)
        lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
        output = None

        # Building blocks.
        def fromrgb(x, res): # res = 2..resolution_log2
            with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
                x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Noise'):
                    x_noise = [conv2d(x, fmaps=1, kernel=1, gain=gain, use_wscale=False)]
                return x, x_noise
        def block(x, res): # res = 2..resolution_log2
            x_noise = []
            with tf.variable_scope('%dx%d' % (2**res, 2**res)):
                if res >= 3: # 8x8 and up
                    with tf.variable_scope('Conv0'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                        with tf.variable_scope('Noise'):
                            x_noise.append(conv2d(x, fmaps=1, kernel=1, gain=gain, use_wscale=False))
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
                        with tf.variable_scope('Noise'):
                            x_noise.append(conv2d(x, fmaps=1, kernel=1, gain=gain, use_wscale=False))
                else: # 4x4
                    if mbstd_group_size > 1:
                        x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                    with tf.variable_scope('Conv'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                        with tf.variable_scope('Noise'):
                            x_noise.append(conv2d(x, fmaps=1, kernel=1, gain=gain, use_wscale=False))
                    with tf.variable_scope('Dense0'):
                        x = act(apply_bias(dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense1'):
                        x = apply_bias(dense(x, fmaps=out_fmap, gain=1, use_wscale=use_wscale))
                        x = tf.reshape(x, [-1]+out_shape)
                return x, x_noise

        noise_list = []
        x, noise = fromrgb(input, resolution_log2)
        noise_list += noise
        for res in range(resolution_log2, 2, -1):
            x, noise = block(x, res)
            noise_list += noise
        output, noise = block(x, 2)
        noise_list += noise

        assert output.dtype == tf.as_dtype(dtype)
        output = tf.identity(output, name='output')
    return output, noise_list



def main():
    base_option = utils.option.parse()

    tflib.init_tf()
    url = os.path.join(base_option['cache_dir'], 'karras2019stylegan-ffhq-1024x1024.pkl')
    with open(url, 'rb') as f: _, _, Gs = pickle.load(f)

    # DEFINE NODES
    noise_latents = tf.random_normal([base_option['minibatch_size']] + Gs.input_shape[1:])
    images = Gs.get_output_for(noise_latents, None, is_validation=True, use_noise=True, randomize_noise=True)
    latents = tf.get_default_graph().get_tensor_by_name('Gs_1/G_mapping/dlatents_out:0')
    encoded_latents, encoded_noise = encode(images)
    Gs.components.synthesis.num_inputs=2
    encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=True, randomize_noise=False)
    noise_vars = [v for v in reversed(tf.global_variables()) if 'G_synthesis_1/noise' in v.name]


    # LOAD LATENT DIRECTIONS
    latent_smile = tf.stack([tf.cast(tf.constant(np.load('latents/smile.npy'), name='latent_smile'), tf.float32)]*base_option['minibatch_size'], axis=0)
    latent_encoded_smile = tf.identity(encoded_latents)
    latent_encoded_smile += 2.0 * latent_smile

    recovered_encoded_images = Gs.components.synthesis.get_output_for(encoded_latents, None, is_validation=True, use_noise=True, randomize_noise=False)
    smile_encoded_images = Gs.components.synthesis.get_output_for(latent_encoded_smile, None, is_validation=True, use_noise=True, randomize_noise=False)
    recovered_noise_vars = [v for v in reversed(tf.global_variables()) if 'G_synthesis_2/noise' in v.name]
    recovered_smile_noise_vars = [v for v in reversed(tf.global_variables()) if 'G_synthesis_3/noise' in v.name]
    with tf.control_dependencies([noise_v.assign(noise) for noise_v, noise in zip(recovered_noise_vars, encoded_noise)]):
        recovered_encoded_images = tf.identity(recovered_encoded_images)
    with tf.control_dependencies([noise_v.assign(noise) for noise_v, noise in zip(recovered_smile_noise_vars, encoded_noise)]):
        smile_encoded_images = tf.identity(smile_encoded_images)


    with tf.name_scope('loss'):
        mse = tf.keras.losses.MeanSquaredError()
        encoding_loss = mse(latents, encoded_latents)
        l2_loss = mse(images, encoded_images)
        lpips_loss =  tf.reduce_mean(lpips_tf.lpips(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(encoded_images, perm=[0,2,3,1])))
        total_loss = (base_option['encoding_lambda']*encoding_loss) + lpips_loss + (base_option['l2_lambda']*l2_loss)

    with tf.name_scope('metric'):
        psnr = tf.reduce_mean(tf.image.psnr(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(tf.transpose(images, perm=[0,2,3,1]), tf.transpose(recovered_encoded_images, perm=[0,2,3,1]), 1.0))

    # DEFINE SUMMARIES
    with tf.name_scope('summary'):
        _ = tf.summary.scalar('encoding_loss', encoding_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('lpips_loss', lpips_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
        _ = tf.summary.scalar('l2_loss', l2_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
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
        latent_manipulator = tf.placeholder(encoded_latents.dtype, encoded_latents.shape)
        test_image_input = tf.placeholder(tf.float32, [None,1024,1024,3], name='image_input')
        test_encoded_latent, test_encoded_noise = encode(tf.transpose(test_image_input, perm=[0,3,1,2]), reuse=True)
        test_recovered_image = Gs.components.synthesis.get_output_for(test_encoded_latent+latent_manipulator, None, is_validation=True, use_noise=True, randomize_noise=False)
        test_noise_vars = [v for v in reversed(tf.global_variables()) if 'G_synthesis_4/noise' in v.name]
        with tf.control_dependencies([noise_v.assign(noise) for noise_v, noise in zip(test_noise_vars, test_encoded_noise)]):
            test_recovered_image = tf.identity(test_recovered_image)
            tf.add_to_collection('TEST_NODES', test_image_input)
            tf.add_to_collection('TEST_NODES', test_encoded_latent)
            tf.add_to_collection('TEST_NODES', test_recovered_image)
            tf.add_to_collection('TEST_NODES', latent_manipulator)

            image_list = [image for image in os.listdir(base_option['test_dir']) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
            assert len(image_list)>0

            imbatch = np.stack([np.array(PIL.Image.open(base_option['test_dir']+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0)/255.0
            test_feed_dict = {test_image_input: imbatch, latent_manipulator:np.zeros(latent_manipulator.shape.as_list())}

            test_image_summary = tf.summary.image('recovered_test', tf.clip_by_value(tf.transpose(test_recovered_image, perm=[0,2,3,1]), 0.0, 1.0), max_outputs=1, family='images', collections=['TEST_SUMMARY'])

    # DEFINE OPTIMIZERS
    with tf.name_scope('optimize'):
        with tf.control_dependencies([noise_v.assign(noise) for noise_v, noise in zip(noise_vars, encoded_noise)]):
            encoder_vars = tf.trainable_variables('encoder')
            optimizer = tf.train.AdamOptimizer(learning_rate=base_option['learning_rate'], name='optimizer')
            gv = optimizer.compute_gradients(loss=total_loss, var_list=encoder_vars)
            optimize = optimizer.apply_gradients(gv, name='optimize')

    saver = tf.train.Saver(var_list=tf.global_variables('encoder'), name='saver')
    train_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/train')
    test_summary_writer = tf.summary.FileWriter(base_option['result_dir']+'/summary/validation')
    sess = tf.get_default_session()
    tflib.tfutil.init_uninitialized_vars()
    for iter in tqdm(range(base_option['num_iter'])):
        iter_scalar_summary, _ = sess.run([scalar_summary, optimize])
        train_summary_writer.add_summary(iter_scalar_summary, iter)
        if iter%base_option['save_iter']==0 or iter==0:
            iter_image_summary = sess.run(image_summary)
            train_summary_writer.add_summary(iter_image_summary, iter)
            test_iter_image_summary = sess.run(test_image_summary, feed_dict=test_feed_dict)
            test_summary_writer.add_summary(test_iter_image_summary, iter)
            saver.save(sess, base_option['result_dir']+'/model/encoded_stylegan.ckpt')



if __name__=='__main__':
    main()
