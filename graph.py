import tensorflow as tf
from stylegan.dnnlib import tflib


class GraphEncodedStyleGAN(object):
    def __init__(self):
        super(GraphEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, model):
        if not model.is_built: model.build()
        self.model = model

        # DEFINE BASIC GRAPH VARIABLES
        sess = tf.get_default_session()
        tflib.tfutil.init_uninitialized_vars()
        self.original_image = tf.clip_by_value(model.original_image, 0.0, 1.0)
        self.recovered_image = tf.clip_by_value(model.encoded_latent_recovered, 0.0, 1.0)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        # tf.add_to_collection("TEST_OPS", self.original_test_image)
        # tf.add_to_collection("TEST_OPS", self.recovered_test_image)
        # tf.add_to_collection("TEST_OPS", self.model.input_latent)
        # tf.add_to_collection("TEST_OPS", self.model.input_image)
        tf.add_to_collection("TEST_OPS", self.original_image)
        tf.add_to_collection("TEST_OPS", self.recovered_image)

        # DEFINE LOSS
        with tf.name_scope('loss'):
            self.perceptual_loss = []
            mse = tf.keras.losses.MeanSquaredError()
            for original, recovered in zip(model.perceptual_features_original, model.perceptual_features_recovered):
                self.perceptual_loss.append(mse(original, recovered))
            self.mse_loss = mse(model.original_image, model.encoded_latent_recovered)
            self.mse_loss = mse(self.model.random_latent, self.model.encoded_latent)
            self.mse_lambda = tf.Variable(initial_value=1.0, trainable=False, dtype=tf.float32, shape=[], name='mse_lambda')
            self.total_loss = self.mse_lambda * self.mse_loss + tf.reduce_sum(self.perceptual_loss)

        # DEFINE METRICS
        with tf.name_scope('metric'):
            self.psnr = tf.reduce_mean(tf.image.psnr(self.original_image, self.recovered_image, 1.0))
            self.ssim = tf.reduce_mean(tf.image.ssim(self.original_image, self.recovered_image, 1.0))

        # DEFINE SUMMARIES
        with tf.name_scope('summary'):
            _ = tf.summary.scalar('total_loss', self.total_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            _ = tf.summary.scalar('mse_loss', self.mse_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            _ = [tf.summary.scalar('perceptual_loss_{}'.format(idx), percep_loss, family='loss', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES]) for idx, percep_loss in enumerate(self.perceptual_loss)]
            _ = tf.summary.scalar('psnr', self.psnr, family='metrics', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            _ = tf.summary.scalar('ssim', self.ssim, family='metrics', collections=['SCALAR_SUMMARY', tf.GraphKeys.SUMMARIES])
            _ = tf.summary.image('random_generated', self.original_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY', 'TRAIN_IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
            _ = tf.summary.image('encoded_recovered', self.recovered_image, max_outputs=1, family='images', collections=['IMAGE_SUMMARY', 'TRAIN_IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
            # _ = tf.summary.image('test_original', self.original_test_image, max_outputs=64, family='images', collections=['IMAGE_SUMMARY', 'TEST_IMAGE_SUMMARY', 'ORIGINAL_IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
            # _ = tf.summary.image('test_recovered', self.recovered_test_image, max_outputs=64, family='images', collections=['IMAGE_SUMMARY', 'TEST_IMAGE_SUMMARY', 'RECOVERED_IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
            self.scalar_summary = tf.summary.merge(tf.get_collection('SCALAR_SUMMARY'))
            self.image_summary = tf.summary.merge(tf.get_collection('IMAGE_SUMMARY'))
            # self.train_image_summary = tf.summary.merge(tf.get_collection('TRAIN_IMAGE_SUMMARY'))
            # self.test_image_summary = tf.summary.merge(tf.get_collection('TEST_IMAGE_SUMMARY'))
            # self.test_original_image_summary = tf.summary.merge(tf.get_collection('ORIGINAL_IMAGE_SUMMARY'))
            # self.test_recovered_image_summary = tf.summary.merge(tf.get_collection('RECOVERED_IMAGE_SUMMARY'))
            self.summary = tf.summary.merge_all()

        # DEFINE OPTIMIZERS
        with tf.name_scope('optimize'):
            encoder_vars = tf.trainable_variables('encoder')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
            gv = optimizer.compute_gradients(loss=self.total_loss, var_list=encoder_vars)
            self.optimize = optimizer.apply_gradients(gv, name='optimize')

        # DEFINE SAVERS
        with tf.name_scope('save'):
            self.saver = tf.train.Saver(var_list=tf.global_variables('encoder'), name='saver')

        self.is_built = True
