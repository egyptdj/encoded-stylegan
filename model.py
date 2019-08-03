import tensorflow as tf


class ModelEncodedStyleGAN(object):
    def __init__(self, stylegan_model):
        super(ModelEncodedStyleGAN, self).__init__()
        self.generator = Generator(stylegan_model)
        self.encoder = Encoder()
        self.perceptor = Perceptor('vgg16')
        self.is_built = False

    def build(self, input):
        self.encoded_latent = self.encoder.build(input)
        self.recovered_image = self.generator.build(self.encoded_latent)
        self.perceptual_features = self.perceptor.build(self.recovered_image), self.perceptor.build(input)
        self.is_built = True


class Encoder(object):
    def __init__(self):
        super(Encoder, self).__init__()

    def build(self,
        input,                              # First input: Images [minibatch, channel, height, width].
        out_shape           = [512],
        labels_in           = None,         # Second input: Labels [minibatch, label_size].
        num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
        resolution          = 32,           # Input resolution. Overridden based on dataset.
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
        structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
        is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
        **_kwargs):                         # Ignore unrecognized keyword args.

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def blur(x): return blur2d(x, blur_filter) if blur_filter else x
        if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
        act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]

        images_in.set_shape([None, num_channels, resolution, resolution])
        # labels_in.set_shape([None, label_size])
        images_in = tf.cast(images_in, dtype)
        # labels_in = tf.cast(labels_in, dtype)
        lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
        scores_out = None

        # Building blocks.
        def fromrgb(x, res): # res = 2..resolution_log2
            with tf.variable_scope('encoder'):
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
                        out_shape_product = 1
                        for i in out_shape:
                            out_shape_product *= i
                        x = apply_bias(dense(x, fmaps=out_shape_product, gain=1, use_wscale=use_wscale))
                        x = tf.reshape(x, [-1]+out_shape)
                return x

        # Fixed structure: simple and efficient, but does not support progressive growing.
        if structure == 'fixed':
            print('=====FIXED=====')
            with tf.variable_scope('encoder'):
                x = fromrgb(images_in, resolution_log2)
                for res in range(resolution_log2, 2, -1):
                    x = block(x, res)
                scores_out = block(x, 2)

        # Linear structure: simple but inefficient.
        if structure == 'linear':
            print('=====LINEAR=====')
            with tf.variable_scope('encoder'):
                img = images_in
                x = fromrgb(img, resolution_log2)
                for res in range(resolution_log2, 2, -1):
                    lod = resolution_log2 - res
                    x = block(x, res)
                    img = downscale2d(img)
                    y = fromrgb(img, res - 1)
                    with tf.variable_scope('Grow_lod%d' % lod):
                        x = tflib.lerp_clip(x, y, lod_in - lod)
                scores_out = block(x, 2)

        # Recursive structure: complex but efficient.
        if structure == 'recursive':
            print('=====RECURSIVE=====')
            with tf.variable_scope('encoder'):
                def cset(cur_lambda, new_cond, new_lambda):
                    return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
                def grow(res, lod):
                    x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
                    if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
                    x = block(x(), res); y = lambda: x
                    if res > 2: y = cset(y, (lod_in > lod), lambda: tflib.lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
                    return y()
                scores_out = grow(2, resolution_log2 - 2)

        # # Label conditioning from "Which Training Methods for GANs do actually Converge?"
        # if label_size:
        #     with tf.variable_scope('LabelSwitch'):
        #         scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)
        assert scores_out.dtype == tf.as_dtype(dtype)
        scores_out = tf.identity(scores_out, name='scores_out')
        return scores_out

        # output = tf.reshape(tf.layers.dense(input, 18*512), [18, 512])
        # return output
        pass


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
        if not nchw: tf.transpose(output, perm=[0,2,3,1])
        return output


class Perceptor(object):
    def __init__(self, model):
        super(Perceptor, self).__init__()
        if model=='vgg16':
            self.perceptual_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.vgg16.preprocess_input
        elif model=='vgg19':
            self.perceptual_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.vgg19.preprocess_input
        else: raise ValueError('perceptor model {} not available for use'.format(model))

    def build(self, input, layers):
        assert isinstance(layers, list) or isinstance(layers, tuple)
        extractor_models = [tf.keras.models.Model(inputs=self.perceptual_model.input, outputs=self.perceptual_model.layers[layer].output) for layer in layers]
        output = [model.predict(self.preprocess_input(input)) for model in extractor_models]
        return output
