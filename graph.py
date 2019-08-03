import tensorflow as tf


class GraphEncodedStyleGAN(object):
    def __init__(self):
        super(GraphEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, dataset, model):
        if not dataset.is_built: dataset.build(nchw=True)
        if not model.is_built: model.build(dataset.image)

        # DEFINE BASIC VARIABLES
        self.input_image = dataset.image
        self.recovered_image = tf.clip_by_value(model.recovered_image, 0.0, 1.0)

        # DEFINE LOSS
        self.perceptual_loss = None
        self.mse_loss = None
        mse = tf.keras.losses.MeanSquaredError()
        for recovered, original in zip(model.perceptual_features):
            self.perceptual_loss += mse(original, recovered)
        self.mse_loss = mse(dataset.images, model.recovered_image)
        self.total_loss = self.mse_loss + self.perceptual_loss

        # DEFINE METRICS
        self.psnr = tf.image.psnr(dataset.images, model.recovered_image)
        self.ssim = tf.image.ssim(dataset.images, model.recovered_image, 1.0)

        # DEFINE SUMMARIES
        _ = tf.summary.scalar('mse_loss', self.mse_loss, family='loss')
        _ = tf.summary.scalar('perceptual_loss', self.perceptual_loss, family='loss')
        _ = tf.summary.scalar('psnr', self.psnr, family='metrics')
        _ = tf.summary.scalar('ssim', self.ssim, family='metrics')
        _ = tf.summary.image('original', dataset.images, max_outputs=1, family='images')
        _ = tf.summary.image('recovered', self.recovered_image, max_outputs=1, family='images')
        self.summary = tf.summary.merge_all()

        # DEFINE OPTIMIZERS
        encoder_vars = tf.trainable_variabels('encoder')
        optimizer = tf.train.AdamOptimizer(learning_rate=tf.get_variable('learning_rate', shape=[], dtype=tf.float32), name='optimizer')
        gv = optimizer.compute_gradients(loss=self.total_loss, var_list=encoder_vars)
        self.optimize = optimizer.apply_gradients(gv, name='optimize')

        # DEFINE SAVERS
        self.saver = tf.train.Saver(var_list=tf.global_variables('encoder'), name='saver')

        self.is_built = True
