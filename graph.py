import tensorflow as tf


class GraphEncodedStyleGAN(object):
    def __init__(self):
        super(GraphEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, dataset, model):
        if not dataset.is_built: dataset.build(nchw=True)
        if not model.is_built: model.build()
        self.dataset = dataset
        self.model = model

        # DEFINE BASIC GRAPH VARIABLES
        self.original_image = tf.clip_by_value(model.original_image, 0.0, 1.0)
        self.searchlight_recovered_image = tf.clip_by_value(model.searchlight_recovered_image, 0.0, 1.0)
        self.encoder_recovered_image = tf.clip_by_value(model.encoder_recovered_image, 0.0, 1.0)
        self.searchlight_learning_rate = tf.placeholder(tf.float32, shape=[], name='searchlight_learning_rate')
        self.encoder_learning_rate = tf.placeholder(tf.float32, shape=[], name='encoder_learning_rate')

        # DEFINE LOSS
        mse = tf.keras.losses.MeanSquaredError()
        with tf.name_scope('searchlight_loss'):
            self.searchlight_perceptual_loss = 0.0
            self.searchlight_mse_loss = 0.0
            for original, recovered in zip(model.perceptual_features_original, model.perceptual_features_searchlight_recovered):
                self.searchlight_perceptual_loss += mse(original, recovered)
            self.searchlight_mse_loss = mse(model.original_image, model.searchlight_recovered_image)
            self.searchlight_total_loss = self.searchlight_mse_loss + self.searchlight_perceptual_loss

        with tf.name_scope('encoder_loss'):
            self.encoder_perceptual_loss = 0.0
            self.encoder_mse_loss = 0.0
            for original, recovered in zip(model.perceptual_features_original, model.perceptual_features_encoder_recovered):
                self.encoder_perceptual_loss += mse(original, recovered)
            self.encoder_mse_loss = mse(model.original_image, model.encoder_recovered_image)
            self.encoder_searchlight_loss = mse(model.encoded_latent, model.searchlight)
            self.encoder_total_loss = self.encoder_mse_loss + self.encoder_perceptual_loss + 1e5*self.encoder_searchlight_loss

        # DEFINE METRICS
        with tf.name_scope('metric'):
            self.searchlight_psnr = tf.reduce_mean(tf.image.psnr(self.original_image, self.searchlight_recovered_image, 1.0))
            self.searchlight_ssim = tf.reduce_mean(tf.image.ssim(self.original_image, self.searchlight_recovered_image, 1.0))
            self.encoder_psnr = tf.reduce_mean(tf.image.psnr(self.original_image, self.encoder_recovered_image, 1.0))
            self.encoder_ssim = tf.reduce_mean(tf.image.ssim(self.original_image, self.encoder_recovered_image, 1.0))

        # DEFINE SUMMARIES
        with tf.name_scope('summary'):
            _ = tf.summary.scalar('total_loss', self.searchlight_total_loss, family='searchlight_loss', collections=['SEARCHLIGHT_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('mse_loss', self.searchlight_mse_loss, family='searchlight_loss', collections=['SEARCHLIGHT_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('perceptual_loss', self.searchlight_perceptual_loss, family='searchlight_loss', collections=['SEARCHLIGHT_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('psnr', self.searchlight_psnr, family='searchlight_metrics', collections=['SEARCHLIGHT_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('ssim', self.searchlight_ssim, family='searchlight_metrics', collections=['SEARCHLIGHT_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('total_loss', self.encoder_total_loss, family='encoder_loss', collections=['ENCODER_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('mse_loss', self.encoder_mse_loss, family='encoder_loss', collections=['ENCODER_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('perceptual_loss', self.encoder_perceptual_loss, family='encoder_loss', collections=['ENCODER_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('psnr', self.encoder_psnr, family='encoder_metrics', collections=['ENCODER_SCALAR_SUMMARY'])
            _ = tf.summary.scalar('ssim', self.encoder_ssim, family='encoder_metrics', collections=['ENCODER_SCALAR_SUMMARY'])
            _ = tf.summary.image('original', self.original_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY'])
            _ = tf.summary.image('searchlight_recovered', self.searchlight_recovered_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY'])
            _ = tf.summary.image('encoder_recovered', self.encoder_recovered_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY'])
            self.searchlight_scalar_summary = tf.summary.merge(tf.get_collection('SEARCHLIGHT_SCALAR_SUMMARY'))
            self.encoder_scalar_summary = tf.summary.merge(tf.get_collection('ENCODER_SCALAR_SUMMARY'))
            self.image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
            self.summary = tf.summary.merge_all()

        # DEFINE OPTIMIZERS
        with tf.name_scope('searchlight_optimize'):
            searchlight_optimizer = tf.train.AdamOptimizer(learning_rate=self.searchlight_learning_rate, name='optimizer')
            searchlight_gv = searchlight_optimizer.compute_gradients(loss=self.searchlight_total_loss, var_list=self.model.searchlight)
            self.searchlight_optimize = searchlight_optimizer.apply_gradients(searchlight_gv, name='searchlight_optimize')

        with tf.name_scope('encoder_optimize'):
            encoder_vars = tf.trainable_variables('encoder')
            encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.encoder_learning_rate, name='encoder_optimizer')
            encoder_gv = encoder_optimizer.compute_gradients(loss=self.encoder_total_loss, var_list=encoder_vars)
            self.encoder_optimize = encoder_optimizer.apply_gradients(encoder_gv, name='encoder_optimize')

        # DEFINE SAVERS
        with tf.name_scope('save'):
            self.saver = tf.train.Saver(var_list=tf.global_variables('encoder'), name='saver')

        self.is_built = True
