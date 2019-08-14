import numpy as np
import tensorflow as tf
from stylegan.training.networks_stylegan import *


class ModelEncodedStyleGAN(object):
    def __init__(self, stylegan_model):
        super(ModelEncodedStyleGAN, self).__init__()
        self.generator = Generator(stylegan_model)
        self.encoder = Encoder()
        self.perceptor = Perceptor('vgg16')
        self.is_built = False

    def build(self, input, test_input):
        self.encoded_latent = self.encoder.build(input)
        self.recovered_image = self.generator.build(self.encoded_latent)
        self.original_image = tf.transpose(input, perm=[0,2,3,1])
        self.perceptual_features_original = self.perceptor.build(tf.image.resize(self.original_image, size=[224,224]))
        self.perceptual_features_recovered = self.perceptor.build(tf.image.resize(self.recovered_image, size=[224,224]))
        self.recovered_test_image = self.generator.build(self.encoder.build(test_input, reuse=True))
        self.original_test_image = tf.transpose(test_input, perm=[0,2,3,1])
        self.is_built = True


class Encoder(object):
    def __init__(self):
        super(Encoder, self).__init__()

    def build(self,
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
                    return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
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

            x = fromrgb(input, resolution_log2)
            for res in range(resolution_log2, 2, -1):
                x = block(x, res)
            output = block(x, 2)

            assert output.dtype == tf.as_dtype(dtype)
            output = tf.identity(output, name='output')
        return output


class Generator(object):
    def __init__(self, stylegan_model):
        super(Generator, self).__init__()
        self.stylegan_model = stylegan_model

    def build(self, input, nchw=False):
        output = self.stylegan_model.get_output_for(input,
            is_validation=True,
            style_mixing_prob=None,
            randomize_noise=False,
            structure='fixed') # refer to function G_synthesis defined in ./stylegan/training/networks_stylegan.py for arguments
        if not nchw: output = tf.transpose(output, perm=[0,2,3,1])
        return output


class Perceptor(object):
    def __init__(self, model):
        super(Perceptor, self).__init__()
        if model=='vgg16':
            self.vgg = Vgg16('/media/bispl/dbx/Dropbox/Academic/01_Research/99_DATASET/VGG16_MODEL/vgg16.npy')
        else: raise ValueError('perceptor model {} not available for use'.format(model))

    def build(self, input):
        self.vgg.build(input)
        output = [self.vgg.conv1_1, self.vgg.conv1_2, self.vgg.conv3_2, self.vgg.conv4_2]
        return output



# https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg16:
    def __init__(self, vgg16_npy_path):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        # self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
