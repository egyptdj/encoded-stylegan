import tensorflow as tf


class GraphEncodedStyleGAN(object):
    def __init__(self):
        super(GraphEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, dataset, model):
        if not dataset.is_built: dataset.build(nchw=True)
        if not model.is_built: model.build(dataset.image)

        # DEFINE BASIC GRAPH VARIABLES
        self.original_image = tf.clip_by_value(model.original_image, 0.0, 1.0)
        self.recovered_image = tf.clip_by_value(model.recovered_image, 0.0, 1.0)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.noise_vars = [v for v in tf.global_variables() if 'G_synthesis/noise' in v.name]

        # DEFINE LOSS
        with tf.name_scope('loss'):
            self.perceptual_loss = 0.0
            self.mse_loss = 0.0
            mse = tf.keras.losses.MeanSquaredError()
            for original, recovered in zip(model.perceptual_features_original, model.perceptual_features_recovered):
                self.perceptual_loss += mse(original, recovered)
            self.mse_loss = mse(model.original_image, model.recovered_image)
            self.total_loss = self.mse_loss + self.perceptual_loss

        # DEFINE METRICS
        with tf.name_scope('metric'):
            self.psnr = tf.reduce_mean(tf.image.psnr(self.original_image, self.recovered_image, 1.0))
            self.ssim = tf.reduce_mean(tf.image.ssim(self.original_image, self.recovered_image, 1.0))

        # DEFINE SUMMARIES
        with tf.name_scope('summary'):
            _ = tf.summary.scalar('total_loss', self.total_loss, family='loss', collections=['SCALAR_SUMMARY'])
            _ = tf.summary.scalar('mse_loss', self.mse_loss, family='loss', collections=['SCALAR_SUMMARY'])
            _ = tf.summary.scalar('perceptual_loss', self.perceptual_loss, family='loss', collections=['SCALAR_SUMMARY'])
            _ = tf.summary.scalar('psnr', self.psnr, family='metrics', collections=['SCALAR_SUMMARY'])
            _ = tf.summary.scalar('ssim', self.ssim, family='metrics', collections=['SCALAR_SUMMARY'])
            _ = tf.summary.image('original', self.original_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY'])
            _ = tf.summary.image('recovered', self.recovered_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY'])
            self.scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
            self.image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
            self.summary = tf.summary.merge_all()

        # DEFINE OPTIMIZERS
        with tf.name_scope('optimize'):
            encoder_tvars = tf.trainable_variables('encoder')
            noise_tvars = tf.trainable_variables('noise_')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
            gv = optimizer.compute_gradients(loss=self.total_loss, var_list=encoder_tvars+noise_tvars)
            with tf.control_dependencies([noise_var.assign(noise) for noise_var, noise in zip(self.noise_vars, model.encoded_noise)]):
                self.optimize = optimizer.apply_gradients(gv, name='optimize')

        # DEFINE SAVERS
        with tf.name_scope('save'):
            self.saver = tf.train.Saver(var_list=tf.global_variables('encoder'), name='saver')

        self.is_built = True
